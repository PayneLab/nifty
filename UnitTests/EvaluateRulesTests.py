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

    def test_score_pair_perfect_separation(self):
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

    def test_permutate_equal_lengths(self):
        quant_df = pd.DataFrame({
            'P1': [1, 2, 3, 4, 5, 4, 10, 2, 3, 4, 5],
            'P2': [4, 3, 2, 1, 6, 3, 4, 1, 2, 3, 4],
            'P3': [2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]
        })
        meta_df = pd.DataFrame({
            'classification_label': ['H', 'D', 'H', 'D', 'D', 'H', 'H', 'D', 'D', 'H', 'D']
        })
        pairs = [('P1', 'P2'), ('P1', 'P3'), ('P2', 'P3')]
        bool_dict = self.evaluator.vectorize_all_pairs(pairs, quant_df)
        bin_labels = self.evaluator.binarize_labels(meta_df)
        results = self.evaluator.permutate(pairs, bool_dict, bin_labels, n_permutations=10)

        self.assertEqual(len(results.index), len(pairs))
        self.assertIn('True_Score', results.columns)

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
        self.assertEqual(result, (3, 2))

    def test_get_proportion_bucket_all_false(self):
        vector = np.array([False, False, False, False, False])
        result = self.evaluator.get_proportion_bucket(vector)
        self.assertEqual(result, (0, 5))

    def test_get_proportion_bucket_empty(self):
        vector = np.array([])
        result = self.evaluator.get_proportion_bucket(vector)
        self.assertEqual(result, (0, 5))

    def test_build_null_buckets_from_permutation(self):
        quant_df = pd.DataFrame({
            'P1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 5, 6],
            'P2': [6, 5, 4, 3, 2, 1, 2, 3, 4, 6, 7, 5],
            'P3': [2, 2, 2, 2, 3, 3, 4, 4, 4, 3, 3, 3],
            'P4': [7, 8, 6, 5, 7, 6, 5, 8, 9, 4, 3, 2],
            'P5': [3, 3, 3, 3, 3, 2, 2, 2, 1, 1, 2, 3],
            'P6': [1, 2, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8]
        })
        meta_df = pd.DataFrame({
            'classification_label': ['H', 'D', 'H', 'D', 'H', 'D', 'H', 'D', 'H', 'D', 'H', 'D']
        })
        pairs = [('P1', 'P2'), ('P1', 'P3'), ('P1', 'P4'), ('P1', 'P5'), ('P1', 'P6'), ('P2', 'P3'), ('P2', 'P4'),
                 ('P2', 'P5'),
                 ('P2', 'P6'), ('P3', 'P4'), ('P3', 'P5'), ('P3', 'P6'), ('P4', 'P5'), ('P4', 'P6'), ('P5', 'P6'), ]
        bool_vectors = self.evaluator.vectorize_all_pairs(pairs, quant_df)
        binarized_labels = self.evaluator.binarize_labels(meta_df)

        buckets = self.evaluator.build_null_buckets_from_permutation(pairs, bool_vectors, binarized_labels)
        # To check we are returning the right structure.
        self.assertIsInstance(buckets, dict)
        for key, value in buckets.items():
            # Having alwasy a (n_true, n_false) key.
            self.assertIsInstance(key, tuple)
            self.assertIsInstance(value, list)
            print(f"Bucket {key}: {value}")
            for score in value:
                self.assertIsInstance(score, float)

if __name__ == '__main__':
    unittest.main()
