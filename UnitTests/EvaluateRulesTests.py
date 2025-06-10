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
        quant_df = pd.DataFrame({
            'sample_id': ['S1', 'S2', 'S3', 'S4'],
            'P1': [1, 6, 3, 8],
            'P2': [4, 5, 6, 1]
        })

        # P1 > P2: False, True, False, True.
        meta_df = pd.DataFrame({
            'sample_id': ['S1', 'S2', 'S3', 'S4'],
            'classification_label': ['B', 'A', 'B', 'A']
        })
        expected_scores = abs(1.0)
        self.assertAlmostEqual(self.evaluator.score_pair(['P1', 'P2'], quant_df, meta_df), expected_scores)


if __name__ == '__main__':
    unittest.main()
