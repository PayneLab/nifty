import unittest
import pandas as pd
import numpy as np
import sys
import os
import cProfile
import pstats
import io

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
        # Pair without NA values.
        try:
            df = pd.DataFrame({
                'P1': [1, 4, 6, 3, 1, 7, 1, 7],
                'P2': [2, 3, 6, 2, 6, 1, 2, 9]
            })
            expected_vector = np.array([False, True, False, True, False, True, False, False])
            np.testing.assert_array_equal(self.evaluator.vectorize_pair(['P1', 'P2'], df), expected_vector)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_vectorize_pair_na_protein_1(self):
        # Pair with NA values in protein 1.
        try:
            df = pd.DataFrame({
                'P1': [np.nan, 4, 6, np.nan],
                'P2': [2, 3, 6, 1]
            })
            expected_vector = np.array([False, True, False, False])
            np.testing.assert_array_equal(self.evaluator.vectorize_pair(['P1', 'P2'], df), expected_vector)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_vectorize_pair_na_protein_2(self):
        # Pair with NA values in protein 2.
        try:
            df = pd.DataFrame({
                'P1': [1, 4, 6],
                'P2': [2, np.nan, 6]
            })
            expected_vector = np.array([False, True, False])
            np.testing.assert_array_equal(self.evaluator.vectorize_pair(['P1', 'P2'], df), expected_vector)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_vectorize_pair_na_in_both(self):
        # Pair with NA values in both proteins.
        try:
            df = pd.DataFrame({
                'P1': [0, 4, 6, 3, np.nan, 7, np.nan, 7],
                'P2': [2, 3, 6, np.nan, 6, np.nan, np.nan, 9]
            })
            expected_vector = np.array([False, True, False, True, False, True, False, False])
            np.testing.assert_array_equal(self.evaluator.vectorize_pair(['P1', 'P2'], df), expected_vector)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_vectorize_pair_same_proteins(self):
        try:
            df = pd.DataFrame({
                'P1': [1, 2, 3, 4],
                'P2': [1, 2, 3, 4]
            })
            expected_vector = np.array([False, False, False, False])
            np.testing.assert_array_equal(self.evaluator.vectorize_pair(['P1', 'P1'], df), expected_vector)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_score_pair_perfect_separation(self):
        try:
            # P1 > P2: False, True, False, True.
            meta_df = pd.DataFrame({
                'sample_id': ['S1', 'S2', 'S3', 'S4'],
                'classification_label': ['H', 'D', 'H', 'D']
            })
            expected_score = abs(1.0)
            self.assertAlmostEqual(self.evaluator.score_pair(['P1', 'P2'], self.quant_df_evaluate_pairs, meta_df),
                                   expected_score)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_score_pair_no_separation(self):
        try:
            quant_df = pd.DataFrame({
                'P1': [1, 2, 3, 4],
                'P2': [4, 3, 2, 1]
            })
            meta_df = pd.DataFrame({
                'sample_id': ['S1', 'S2', 'S3', 'S4'],
                'classification_label': ['H', 'D', 'H', 'D']
            })
            expected_score = 0.0
            self.assertAlmostEqual(self.evaluator.score_pair(['P1', 'P2'], quant_df, meta_df), expected_score)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_score_pair_half_separation(self):
        try:
            quant_df = pd.DataFrame({
                'P1': [1, 6, 4, 7],
                'P2': [5, 5, 4, 8]
            })
            meta_df = pd.DataFrame({
                'sample_id': ['S1', 'S2', 'S3', 'S4'],
                'classification_label': ['H', 'H', 'D', 'D']
            })

            expected_score = 0.5
            actual_score = self.evaluator.score_pair(['P1', 'P2'], quant_df, meta_df)
            self.assertAlmostEqual(actual_score, expected_score)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_evaluate_pairs_output(self):
        try:
            quant_df = pd.DataFrame({
                'P1': [1, 2, 3, 4],
                'P2': [4, 3, 2, 1],
                'P3': [2, 2, 2, 2]
            })
            meta_df = pd.DataFrame({
                'sample_id': ['S1', 'S2', 'S3', 'S4'],
                'classification_label': ['H', 'D', 'H', 'D']
            })
            pairs = [('P1', 'P2'), ('P1', 'P3'), ('P2', 'P3')]
            results = self.evaluator.evaluate_pairs(pairs, quant_df, meta_df)

            self.assertEqual(len(results), len(pairs))
            for pair, score in results:
                self.assertIsInstance(pair, tuple)
                self.assertEqual(len(pair), 2)
                self.assertIsInstance(score, float)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_randomize_labels(self):
        try:
            label_array = np.array(['H', 'D', 'H', 'D', 'H', 'D', 'H'])
            seen = set()
            for i in range(100):
                shuffled_labels = self.evaluator.randomize_labels(label_array)
                seen.add(tuple(shuffled_labels))
            self.assertGreater(len(seen), 30)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_permutate_equal_lengths(self):
        try:
            quant_df = pd.DataFrame({
                'P1': [1, 2, 3, 4, 5, 4, 10, 2, 3, 4, 5],
                'P2': [4, 3, 2, 1, 6, 3, 4, 1, 2, 3, 4],
                'P3': [2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]
            })
            meta_df = pd.DataFrame({
                'sample_id': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11'],
                'classification_label': ['H', 'D', 'H', 'D', 'D', 'H', 'H', 'D', 'D', 'H', 'D']
            })
            pairs = [('P1', 'P2'), ('P1', 'P3'), ('P2', 'P3')]

            cProfile
            pr = cProfile.Profile()
            pr.enable()

            results = self.evaluator.permutate(pairs, quant_df, meta_df)

            pr.disable()
            s = io.StringIO()
            pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats()
            print(s.getvalue())

            print(results)
            self.assertEqual(len(results.index), len(pairs))
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")


if __name__ == '__main__':
    unittest.main()
