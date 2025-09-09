import tempfile
import unittest
import pandas as pd
import numpy as np
import sys
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from EvaluateRules import EvaluateRules


class TestEvaluateRules(unittest.TestCase):
    def setUp(self):
        self.evaluator = EvaluateRules()
        self.quant_df_evaluate_pairs = pd.DataFrame({
            'sample_id': ['S1', 'S2', 'S3', 'S4'],
            'P1': [1, 6, 3, 8],
            'P2': [4, 5, 6, 1]
        })

    def test_vectorize_pair_no_na(self):
        df = pd.DataFrame({
            'P1': [1, 4, 6, 3, 1, 7, 1, 7],
            'P2': [2, 3, 6, 2, 6, 1, 2, 9]
        })
        expected = np.array([False, True, False, True, False, True, False, False])
        result = self.evaluator.vectorize_pair(['P1', 'P2'], df)
        np.testing.assert_array_equal(result, expected)

    def test_vectorize_pair_na_protein_1(self):
        df = pd.DataFrame({
            'P1': [np.nan, 4, 6, np.nan],
            'P2': [2, 3, 6, 1]
        })
        expected = np.array([False, True, False, False])
        result = self.evaluator.vectorize_pair(['P1', 'P2'], df)
        np.testing.assert_array_equal(result, expected)

    def test_vectorize_pair_na_protein_2(self):
        df = pd.DataFrame({
            'P1': [1, 4, 6],
            'P2': [2, np.nan, 6]
        })
        expected = np.array([False, True, False])
        result = self.evaluator.vectorize_pair(['P1', 'P2'], df)
        np.testing.assert_array_equal(result, expected)

    def test_vectorize_pair_na_in_both(self):
        df = pd.DataFrame({
            'P1': [0, 4, 6, 3, np.nan, 7, np.nan, 7],
            'P2': [2, 3, 6, np.nan, 6, np.nan, np.nan, 9]
        })
        expected = np.array([False, True, False, True, False, True, False, False])
        result = self.evaluator.vectorize_pair(['P1', 'P2'], df)
        np.testing.assert_array_equal(result, expected)

    def test_vectorize_pair_same_proteins(self):
        df = pd.DataFrame({'P1': [1, 2, 3, 4]})
        expected = np.array([False, False, False, False])
        result = self.evaluator.vectorize_pair(['P1', 'P1'], df)
        np.testing.assert_array_equal(result, expected)

    def test_score_pair_full_separation_negative_selected(self):
        quant_df = pd.DataFrame({
            'P1': [1, 6, 3, 8],
            'P2': [4, 5, 6, 1]
        })
        pair = ('P1', 'P2')
        meta_df = pd.DataFrame({
            'classification_label': ['H', 'D', 'H', 'D']
        })
        bin_labels = self.evaluator.binarize_labels(meta_df)
        bool_dict = {pair: self.evaluator.vectorize_pair(pair, quant_df)}
        self.evaluator._n_pos = np.sum(bin_labels == 1)
        self.evaluator._n_neg = np.sum(bin_labels == 0)
        score = self.evaluator.score_pair(pair, bool_dict, bin_labels)
        self.assertAlmostEqual(score, 1.0)

    def test_score_pair_no_separation(self):
        quant_df = pd.DataFrame({
            'P1': [1, 2, 3, 4],
            'P2': [4, 3, 2, 1]
        })
        pair = ('P1', 'P2')
        meta_df = pd.DataFrame({'classification_label': ['H', 'D', 'H', 'D']})
        bin_labels = self.evaluator.binarize_labels(meta_df)
        bool_dict = {pair: self.evaluator.vectorize_pair(pair, quant_df)}
        self.evaluator._n_pos = np.sum(bin_labels == 1)
        self.evaluator._n_neg = np.sum(bin_labels == 0)
        score = self.evaluator.score_pair(pair, bool_dict, bin_labels)
        self.assertAlmostEqual(score, 0.0)

    def test_score_pair_half_separation(self):
        quant_df = pd.DataFrame({
            'P1': [1, 6, 4, 7],
            'P2': [5, 5, 4, 8]
        })
        pair = ('P1', 'P2')
        meta_df = pd.DataFrame({'classification_label': ['H', 'H', 'D', 'D']})
        bin_labels = self.evaluator.binarize_labels(meta_df)
        bool_dict = {pair: self.evaluator.vectorize_pair(pair, quant_df)}
        self.evaluator._n_pos = np.sum(bin_labels == 1)
        self.evaluator._n_neg = np.sum(bin_labels == 0)
        score = self.evaluator.score_pair(pair, bool_dict, bin_labels)
        self.assertAlmostEqual(score, 0.5)

    def test_score_pair_full_separation_negative_selected(self):
        quant_df = pd.DataFrame({
            'P1': [1, 2, 3, 4],
            'P2': [4, 3, 2, 1]
        })
        pair = ('P1', 'P2')
        meta_df = pd.DataFrame({'classification_label': ['D', 'D', 'H', 'H']})
        bin_labels = self.evaluator.binarize_labels(meta_df)
        bool_dict = {pair: self.evaluator.vectorize_pair(pair, quant_df)}
        self.evaluator._n_pos = np.sum(bin_labels == 1)
        self.evaluator._n_neg = np.sum(bin_labels == 0)
        score = self.evaluator.score_pair(pair, bool_dict, bin_labels)
        self.assertAlmostEqual(score, 1.0)

    def test_evaluate_pairs_output(self):
        quant_df = pd.DataFrame({
            'P1': [1, 2, 3, 4],
            'P2': [4, 3, 2, 1],
            'P3': [2, 2, 2, 2]
        })
        meta_df = pd.DataFrame({'classification_label': ['H', 'D', 'H', 'D']})
        pairs = [('P1', 'P2'), ('P1', 'P3'), ('P2', 'P3')]
        bool_dict = self.evaluator.vectorize_all_pairs(pairs, quant_df)
        bin_labels = self.evaluator.binarize_labels(meta_df)
        result = self.evaluator.evaluate_pairs(pairs, bool_dict, bin_labels)

        self.assertEqual(len(result), len(pairs))
        for pair, score in result:
            self.assertIsInstance(pair, tuple)
            self.assertEqual(len(pair), 2)
            self.assertIsInstance(score, float)

    def test_randomize_labels(self):
        label_array = np.array(['H', 'D', 'H', 'D', 'H', 'D', 'H'])
        seen = set()
        for _ in range(100):
            shuffled = tuple(self.evaluator.randomize_labels(label_array))
            seen.add(shuffled)
        self.assertGreater(len(seen), 30)

    def test_get_bool_vectors_for_pairs_no_na(self):
        df = pd.DataFrame({
            'P1': [1, 4, 6, 3, 1, 7, 1, 7],
            'P2': [2, 3, 6, 2, 6, 1, 2, 9]
        })
        pairs = [('P1', 'P2')]
        expected_vector = np.array([False, True, False, True, False, True, False, False])
        bool_vectors = self.evaluator.vectorize_all_pairs(pairs, df)
        np.testing.assert_array_equal(bool_vectors[('P1', 'P2')], expected_vector)

    def test_get_proportion_bucket_true_false(self):
        vector = np.array([True, False, True, False, True])
        result = self.evaluator.get_proportion_bucket(vector)
        self.assertEqual(result, 60)

    def test_get_proportion_bucket_all_false(self):
        vector = np.array([False, False, False, False, False])
        result = self.evaluator.get_proportion_bucket(vector)
        self.assertEqual(result, 0)

    def test_get_proportion_bucket_all_true(self):
        vector = np.array([True, True, True, True])
        result = self.evaluator.get_proportion_bucket(vector)
        self.assertEqual(result, 100)

    def test_get_proportion_bucket_places_scores_in_correct_buckets(self):
        test_cases = [([False, False, False, False, False], 0),
                      ([True, False, False, False, False], 20),
                      ([True, True, False, False, False], 40),
                      ([True, True, True, False, False], 60),
                      ([True, True, True, True, False], 80),
                      ([True, True, True, True, True], 100),
                      ([True, True, False, False], 50),
                      ([True, True, True, False], 75),
                      ([True, False, False], 33),
                      ([True, True, True], 100),
                      ([False, False, True], 33), ]

        for bool_vector, proportion in test_cases:
            vector = np.array(bool_vector)
            bucket = self.evaluator.get_proportion_bucket(vector)
            self.assertEqual(bucket, proportion)

    def test_get_proportion_bucket_large_vector(self):
        bool_vector = np.array([True] * 5000 + [False] * 5000)
        result = self.evaluator.get_proportion_bucket(bool_vector)
        self.assertEqual(result, 50)

    def test_get_proportion_bucket_close_rounding(self):
        bool_vector_1 = np.array([True, True, True, False])
        self.assertEqual(self.evaluator.get_proportion_bucket(bool_vector_1), 75)
        bool_vector_2 = np.array([True, True, False])
        self.assertEqual(self.evaluator.get_proportion_bucket(bool_vector_2), 67)
        bool_vector_3 = np.array([True, False])
        self.assertEqual(self.evaluator.get_proportion_bucket(bool_vector_3), 50)

    # To delete.
    def test_get_rule_to_buckets_with_pairs(self):
        pairs = [('P1', 'P2'), ('P1', 'P3'), ('P2', 'P3')]
        bool_dict = {
            ('P1', 'P2'): np.array([True, True, False, False]),
            ('P1', 'P3'): np.array([True, True, True, False]),
            ('P2', 'P3'): np.array([True, False, False, False])
        }

        expected_rule_to_buckets = {
            ('P1', 'P2'): 50,
            ('P1', 'P3'): 75,
            ('P2', 'P3'): 25
        }

        results = self.evaluator.get_rule_to_buckets(pairs, bool_dict)
        self.assertEqual(results, expected_rule_to_buckets)

    def test_get_bucket_to_rules_multiple_buckets(self):
        pairs = [('P1', 'P2'), ('P1', 'P3'), ('P2', 'P3')]
        bool_dict = {
            ('P1', 'P2'): np.array([True, True, False, False]),
            ('P1', 'P3'): np.array([True, True, True, False]),
            ('P2', 'P3'): np.array([True, False, False, False])
        }

        expected_rule_to_buckets = {
            50: [('P1', 'P2')],
            75: [('P1', 'P3')],
            25: [('P2', 'P3')]
        }

        results = self.evaluator.get_bucket_to_rules(pairs, bool_dict)
        self.assertEqual(results, expected_rule_to_buckets)

    def test_get_bucket_to_rules_multiple_rules_same_buckets(self):
        pairs = [('P1', 'P2'), ('P1', 'P3'), ('P2', 'P3')]
        bool_dict = {
            ('P1', 'P2'): np.array([True, False, False, False, True, True, True]),
            ('P1', 'P3'): np.array([True, True, True, True, False, False, False]),
            ('P2', 'P3'): np.array([True, True, True, True, False, False, False]),
        }

        expected_rule_to_buckets = {
            57: [('P1', 'P2'), ('P1', 'P3'), ('P2', 'P3')],
        }

        results = self.evaluator.get_bucket_to_rules(pairs, bool_dict)
        self.assertEqual(results, expected_rule_to_buckets)

    """
    def test_expand_small_null_distributions(self):
        buckets = {57: np.array([0.1, 0.2])}
        bucket_to_rules = {57: [('P1','P2')]}
        binarized_labels = np.array([1, 0, 1, 0])
        bool_dict = {}

        results = self.evaluator.NEWEST_expand_small_null_distributions(buckets, bool_dict, binarized_labels, bucket_to_rules)
        self.assertGreaterEqual(len(results[57]), 100)
    """

    def test_summarize_bucket_stats_score_above_all_nulls(self):
        true_scores = {('P1', 'P2'): 0.9}
        bucket_to_rules = {60: [('P1', 'P2')]}
        buckets = {60: np.array([0.1, 0.2, 0.3])}

        df = self.evaluator.summarize_bucket_stats(true_scores, bucket_to_rules, buckets)
        self.assertEqual(df.iloc[0]['P_Value'], 0.0)

    def test_summarize_bucket_stats_score_below_all_nulls(self):
        true_scores = {('P3', 'P4'): 0.05}
        bucket_to_rules = {80: [('P3', 'P4')]}
        buckets = {80: np.array([0.2, 0.3, 0.4])}

        df = self.evaluator.summarize_bucket_stats(true_scores, bucket_to_rules, buckets)
        self.assertEqual(df.iloc[0]['P_Value'], 1.0)

    def test_summarize_bucket_stats_score_all_equal_null(self):
        true_scores = {('P1', 'P2'): 0.5}
        bucket_to_rules = {60: [('P1', 'P2')]}
        buckets = {60: np.array([0.3, 0.5, 0.7])}

        df = self.evaluator.summarize_bucket_stats(true_scores, bucket_to_rules, buckets)
        p_val = df.loc[df['Gene_Pair'] == ('P1', 'P2'), 'P_Value'].iloc[0]
        self.assertAlmostEqual(p_val, 2 / 3, places=3)

    def test_summarize_bucket_stats_multiple_buckets(self):
        true_scores = {('P1', 'P2'): 0.8, ('P3', 'P4'): 0.1}
        bucket_to_rules = {50: [('P1', 'P2')], 25: [('P3', 'P4')]}
        buckets = {
            50: np.array([0.5, 0.9]),
            25: np.array([0.1, 0.2])
        }

        df = self.evaluator.summarize_bucket_stats(true_scores, bucket_to_rules, buckets)

        self.assertAlmostEqual(df.loc[df['Gene_Pair'] == ('P1', 'P2'), 'P_Value'].iloc[0], 0.5)
        self.assertAlmostEqual(df.loc[df['Gene_Pair'] == ('P3', 'P4'), 'P_Value'].iloc[0], 1.0)

        self.assertCountEqual(df['Bucket'].values, [50, 25])

    def test_filter_and_save_pairs_returns_top_k_non_disjoint_pairs(self):
        self.evaluator = EvaluateRules.__new__(EvaluateRules)
        self.evaluator.disjoint = False

        summary_df = pd.DataFrame({
            "Gene_Pair": [
                ('P1', 'P2'),
                ('P1', 'P3'),
                ('P1', 'P4'),
                ('P1', 'P5'),
                ('P2', 'P3'),
                ('P2', 'P4'),
                ('P2', 'P5'),
                ('P3', 'P4'),
                ('P3', 'P5'),
                ('P4', 'P5')
            ],
            "True_Score": [0.91, 0.88, 0.85, 0.82, 0.80, 0.78, 0.76, 0.74, 0.72, 0.70],
            "Bucket": [60, 60, 80, 80, 60, 60, 80, 80, 60, 80],
            "P_Value": [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055],
            "FDR": [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055]
        })

        with tempfile.NamedTemporaryFile(delete=True) as temp:
            summary = self.evaluator.filter_and_save_rules(summary_df, output_file_path=temp.name, k =5, disjoint=False)

        expected_pairs = [
            ('P1', 'P2'),
            ('P1', 'P3'),
            ('P1', 'P4'),
            ('P1', 'P5'),
            ('P2', 'P3'),
        ]

        actual_pairs = [tuple(x) if not isinstance(x, tuple) else x for x in summary['Gene_Pair']]

        assert actual_pairs== expected_pairs

    def test_filter_and_save_pairs_returns_top_k_disjoint_pairs(self):
        self.evaluator = EvaluateRules.__new__(EvaluateRules)
        self.evaluator.k = 2
        self.evaluator.disjoint = True

        summary_df = pd.DataFrame({
            "Gene_Pair": [
                ('P1', 'P2'),
                ('P1', 'P3'),
                ('P1', 'P4'),
                ('P1', 'P5'),
                ('P2', 'P3'),
                ('P2', 'P4'),
                ('P2', 'P5'),
                ('P3', 'P4'),
                ('P3', 'P5'),
                ('P4', 'P5')
            ],
            "True_Score": [0.91, 0.88, 0.85, 0.82, 0.80, 0.78, 0.76, 0.74, 0.72, 0.70],
            "Bucket": [60, 60, 80, 80, 60, 60, 80, 80, 60, 80],
            "P_Value": [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055],
            "FDR": [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055]
        })

        with tempfile.NamedTemporaryFile(delete=True) as temp:
            summary = self.evaluator.filter_and_save_rules(summary_df, output_file_path=temp.name, k=10)

        expected_pairs = [
            ('P1', 'P2'),
            ('P3', 'P4')
        ]

        actual_pairs = [tuple(x) if not isinstance(x, tuple) else x for x in summary['Gene_Pair']]

        assert actual_pairs == expected_pairs

if __name__ == '__main__':
    unittest.main()
