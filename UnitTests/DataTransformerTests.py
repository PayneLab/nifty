import sys
import os

import unittest
import pandas as pd
import numpy as np

current_dir = os.path.dirname(__file__)

parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from DataTransformer import DataTransformer


class TestVectorizeAllPairs(unittest.TestCase):

    def setUp(self):
        self.transformer = DataTransformer()

        self.pairs = [('P1', 'P2'), 
                      ('P1', 'P3'), 
                      ('P2', 'P3')]
        
        self.quant_df = pd.DataFrame({
            'sample_id': ['samp1', 'samp2', 'samp3', 'samp4', 'samp5'], 
            'P1': [0.7328, np.nan, 0.4481, np.nan, 0.9125], 
            'P2': [np.nan, 0.1839, np.nan, 0.5921, 0.0197], 
            'P3': [0.7416, 0.2604, np.nan, 0.4973, np.nan]})
        
        self.quant_df.set_index('sample_id', inplace=True)
        
    def test_3_pairs(self):
        expected_bool_matrix = np.array([[True, False, True, False, True], 
                                         [False, False, True, False, True], 
                                         [False, False, False, True, True]])

        bool_matrix = self.transformer.vectorize_all_pairs(self.pairs, self.quant_df)

        self.assertEqual(expected_bool_matrix.shape[0], len(self.pairs))
        self.assertEqual(expected_bool_matrix.shape[1], 5)

        for i, pair in enumerate(self.pairs):
            np.testing.assert_array_equal(expected_bool_matrix[i, :], bool_matrix[i, :])

    def test_2_pairs(self):
        self.pairs.pop(1)

        expected_bool_matrix = np.array([[True, False, True, False, True],  
                                       [False, False, False, True, True]])

        bool_matrix = self.transformer.vectorize_all_pairs(self.pairs, self.quant_df)

        self.assertEqual(expected_bool_matrix.shape[0], len(self.pairs))
        self.assertEqual(expected_bool_matrix.shape[1], 5)

        for i, pair in enumerate(self.pairs):
            np.testing.assert_array_equal(expected_bool_matrix[i, :], bool_matrix[i, :])

    def test_1_pair(self):
        self.pairs.pop(0)
        self.pairs.pop(0)

        expected_bool_matrix = np.array([[False, False, False, True, True]])

        bool_matrix = self.transformer.vectorize_all_pairs(self.pairs, self.quant_df)

        self.assertEqual(expected_bool_matrix.shape[0], len(self.pairs))
        self.assertEqual(expected_bool_matrix.shape[1], 5)

        for i, pair in enumerate(self.pairs):
            np.testing.assert_array_equal(expected_bool_matrix[i, :], bool_matrix[i, :])


class TestVectorizePair(unittest.TestCase):
     
    def setUp(self):
        self.transformer = DataTransformer()

    def test_no_na(self):
            df = pd.DataFrame({
                'P1': [1, 4, 6, 3, 1, 7, 1, 7],
                'P2': [2, 3, 6, 2, 6, 1, 2, 9]
            })
            expected = np.array([False, True, False, True, False, True, False, False])
            result = self.transformer.vectorize_pair(('P1', 'P2'), df)
            np.testing.assert_array_equal(result, expected)

    def test_na_protein_1(self):
        df = pd.DataFrame({
            'P1': [np.nan, 4, 6, np.nan],
            'P2': [2, 3, 6, 1]
        })
        expected = np.array([False, True, False, False])
        result = self.transformer.vectorize_pair(('P1', 'P2'), df)
        np.testing.assert_array_equal(result, expected)

    def test_na_protein_2(self):
        df = pd.DataFrame({
            'P1': [1, 4, 6],
            'P2': [2, np.nan, 6]
        })
        expected = np.array([False, True, False])
        result = self.transformer.vectorize_pair(('P1', 'P2'), df)
        np.testing.assert_array_equal(result, expected)

    def test_na_in_both(self):
        df = pd.DataFrame({
            'P1': [0, 4, 6, 3, np.nan, 7, np.nan, 7],
            'P2': [2, 3, 6, np.nan, 6, np.nan, np.nan, 9]
        })
        expected = np.array([False, True, False, True, False, True, False, False])
        result = self.transformer.vectorize_pair(('P1', 'P2'), df)
        np.testing.assert_array_equal(result, expected)

    def test_same_proteins(self):
        df = pd.DataFrame({'P1': [1, 2, 3, 4]})
        expected = np.array([False, False, False, False])
        result = self.transformer.vectorize_pair(('P1', 'P1'), df)
        np.testing.assert_array_equal(result, expected)


class TestFilterRules(unittest.TestCase):

    def setUp(self):
        self.transformer = DataTransformer()

        self.quant_df = pd.DataFrame({
            'sample_id': ['samp1', 'samp2', 'samp3', 'samp4', 'samp5'], 
            'AAAAA': [0.7328, np.nan, 0.4481, np.nan, 0.9125], 
            'BBBBB': [np.nan, 0.1839, np.nan, 0.5921, 0.0197], 
            'CCCCC': [0.7416, 0.2604, np.nan, 0.4973, np.nan], 
            'DDDDD': [np.nan, np.nan, 0.8231, 0.3795, 0.0028], 
            'EEEEE': [0.5267, np.nan, 0.9089, 0.1442, np.nan], 
            'FFFFF': [np.nan, 0.6834, 0.1978, np.nan, 0.9536] 
        })
        self.quant_df.set_index('sample_id', inplace=True)

        self.feature_df = pd.DataFrame({
            'Protein1': ['AAAAA', 'BBBBB', 'CCCCC'],
            'Protein2': ['DDDDD', 'EEEEE', 'FFFFF'] 
        })

    def test_all_proteins_present(self):
        updated_feature_df = self.transformer.filter_rules(self.feature_df, self.quant_df)

        self.assertTrue(self.feature_df.equals(updated_feature_df))

    def test_some_proteins_absent_Protein1(self):
        self.quant_df.drop('AAAAA', axis=1, inplace=True)

        updated_feature_df = self.transformer.filter_rules(self.feature_df, self.quant_df)

        self.assertFalse(self.feature_df.equals(updated_feature_df))
        self.assertEqual(updated_feature_df['Protein1'].tolist(), ['BBBBB', 'CCCCC'])
        self.assertEqual(updated_feature_df['Protein2'].tolist(), ['EEEEE', 'FFFFF'])

    def test_some_proteins_absent_Protein2(self):
        self.quant_df.drop('EEEEE', axis=1, inplace=True)
        self.quant_df.drop('FFFFF', axis=1, inplace=True)

        updated_feature_df = self.transformer.filter_rules(self.feature_df, self.quant_df)

        self.assertFalse(self.feature_df.equals(updated_feature_df))
        self.assertEqual(updated_feature_df['Protein1'].tolist(), ['AAAAA'])
        self.assertEqual(updated_feature_df['Protein2'].tolist(), ['DDDDD'])

    def test_some_proteins_absent_both(self):
        self.quant_df.drop('AAAAA', axis=1, inplace=True)
        self.quant_df.drop('FFFFF', axis=1, inplace=True)

        updated_feature_df = self.transformer.filter_rules(self.feature_df, self.quant_df)

        self.assertFalse(self.feature_df.equals(updated_feature_df))
        self.assertEqual(updated_feature_df['Protein1'].tolist(), ['BBBBB'])
        self.assertEqual(updated_feature_df['Protein2'].tolist(), ['EEEEE'])

    def test_all_proteins_absent(self):
        self.quant_df.drop('AAAAA', axis=1, inplace=True)
        self.quant_df.drop('EEEEE', axis=1, inplace=True)
        self.quant_df.drop('FFFFF', axis=1, inplace=True)

        with self.assertRaises(SystemExit) as e:
            self.transformer.filter_rules(self.feature_df, self.quant_df)

        self.assertEqual(e.exception.code, 1)


class TestAddMissingProteins(unittest.TestCase):

    def setUp(self):
        self.transformer = DataTransformer()

        self.quant_df = pd.DataFrame({
            'sample_id': ['samp1', 'samp2', 'samp3', 'samp4', 'samp5'], 
            'AAAAA': [0.7328, np.nan, 0.4481, np.nan, 0.9125], 
            'BBBBB': [np.nan, 0.1839, np.nan, 0.5921, 0.0197], 
            'CCCCC': [0.7416, 0.2604, np.nan, 0.4973, np.nan], 
            'DDDDD': [np.nan, np.nan, 0.8231, 0.3795, 0.0028], 
            'EEEEE': [0.5267, np.nan, 0.9089, 0.1442, np.nan], 
            'FFFFF': [np.nan, 0.6834, 0.1978, np.nan, 0.9536] 
        })
        self.quant_df.set_index('sample_id', inplace=True)

        self.feature_df = pd.DataFrame({
            'Protein1': ['AAAAA', 'BBBBB', 'CCCCC'],
            'Protein2': ['DDDDD', 'EEEEE', 'FFFFF'] 
        })

    def test_all_proteins_present(self):
        updated_quant_df = self.transformer.add_missing_proteins(self.feature_df, self.quant_df)

        self.assertTrue(self.quant_df.equals(updated_quant_df))

    def test_some_proteins_absent_Protein1(self):
        self.quant_df.drop('AAAAA', axis=1, inplace=True)

        updated_quant_df = self.transformer.add_missing_proteins(self.feature_df, self.quant_df)

        self.assertFalse(self.quant_df.equals(updated_quant_df))
        self.assertEqual(sorted(updated_quant_df.columns.tolist()), ['AAAAA', 'BBBBB', 'CCCCC', 'DDDDD', 'EEEEE', 'FFFFF'])
        self.assertTrue(updated_quant_df['AAAAA'].isna().all())
        np.testing.assert_allclose(self.quant_df['BBBBB'].tolist(), updated_quant_df['BBBBB'].tolist(), equal_nan=True)
        np.testing.assert_allclose(self.quant_df['CCCCC'].tolist(), updated_quant_df['CCCCC'].tolist(), equal_nan=True)
        np.testing.assert_allclose(self.quant_df['DDDDD'].tolist(), updated_quant_df['DDDDD'].tolist(), equal_nan=True)
        np.testing.assert_allclose(self.quant_df['EEEEE'].tolist(), updated_quant_df['EEEEE'].tolist(), equal_nan=True)
        np.testing.assert_allclose(self.quant_df['FFFFF'].tolist(), updated_quant_df['FFFFF'].tolist(), equal_nan=True)

    def test_some_proteins_absent_Protein2(self):
        self.quant_df.drop('EEEEE', axis=1, inplace=True)
        self.quant_df.drop('FFFFF', axis=1, inplace=True)

        updated_quant_df = self.transformer.add_missing_proteins(self.feature_df, self.quant_df)

        self.assertFalse(self.quant_df.equals(updated_quant_df))
        self.assertEqual(sorted(updated_quant_df.columns.tolist()), ['AAAAA', 'BBBBB', 'CCCCC', 'DDDDD', 'EEEEE', 'FFFFF'])
        self.assertTrue(updated_quant_df['EEEEE'].isna().all())
        self.assertTrue(updated_quant_df['FFFFF'].isna().all())
        np.testing.assert_allclose(self.quant_df['AAAAA'].tolist(), updated_quant_df['AAAAA'].tolist(), equal_nan=True)
        np.testing.assert_allclose(self.quant_df['BBBBB'].tolist(), updated_quant_df['BBBBB'].tolist(), equal_nan=True)
        np.testing.assert_allclose(self.quant_df['CCCCC'].tolist(), updated_quant_df['CCCCC'].tolist(), equal_nan=True)
        np.testing.assert_allclose(self.quant_df['DDDDD'].tolist(), updated_quant_df['DDDDD'].tolist(), equal_nan=True)

    def test_some_proteins_absent_both(self):
        self.quant_df.drop('AAAAA', axis=1, inplace=True)
        self.quant_df.drop('FFFFF', axis=1, inplace=True)

        updated_quant_df = self.transformer.add_missing_proteins(self.feature_df, self.quant_df)

        self.assertFalse(self.quant_df.equals(updated_quant_df))
        self.assertEqual(sorted(updated_quant_df.columns.tolist()), ['AAAAA', 'BBBBB', 'CCCCC', 'DDDDD', 'EEEEE', 'FFFFF'])
        self.assertTrue(updated_quant_df['AAAAA'].isna().all())
        self.assertTrue(updated_quant_df['FFFFF'].isna().all())
        np.testing.assert_allclose(self.quant_df['BBBBB'].tolist(), updated_quant_df['BBBBB'].tolist(), equal_nan=True)
        np.testing.assert_allclose(self.quant_df['CCCCC'].tolist(), updated_quant_df['CCCCC'].tolist(), equal_nan=True)
        np.testing.assert_allclose(self.quant_df['DDDDD'].tolist(), updated_quant_df['DDDDD'].tolist(), equal_nan=True)
        np.testing.assert_allclose(self.quant_df['EEEEE'].tolist(), updated_quant_df['EEEEE'].tolist(), equal_nan=True)

    def test_all_proteins_absent(self):
        self.quant_df.drop('AAAAA', axis=1, inplace=True)
        self.quant_df.drop('BBBBB', axis=1, inplace=True)
        self.quant_df.drop('CCCCC', axis=1, inplace=True)
        self.quant_df.drop('DDDDD', axis=1, inplace=True)
        self.quant_df.drop('EEEEE', axis=1, inplace=True)
        self.quant_df.drop('FFFFF', axis=1, inplace=True)

        with self.assertRaises(SystemExit) as e:
            self.transformer.add_missing_proteins(self.feature_df, self.quant_df)

        self.assertEqual(e.exception.code, 1)


class TestPrepVectorizedPairsForScikitlearn(unittest.TestCase):

    def setUp(self):
        self.transformer = DataTransformer()

        # Feature table defining the order and identity of feature pairs
        self.feature_df = pd.DataFrame({
            'Protein1': ['P1', 'P1', 'P2'],
            'Protein2': ['P2', 'P3', 'P3']
        })
        self.rules = list(zip(self.feature_df['Protein1'].tolist(), self.feature_df['Protein2'].tolist()))

        # Expected ordered list of joined pair strings
        self.expected_column_order = ["P1>P2", "P1>P3", "P2>P3"]

    def test_basic_functionality(self):
        """
        Standard case:
        - bool_dict contains all required pairs
        - values are boolean arrays
        """
        bool_matrix = np.array([
            [True, False, True],
            [False, True, False],
            [True, True, False]
        ])

        df = self.transformer.prep_vectorized_pairs_for_scikitlearn(
            self.rules, bool_matrix
        )

        # Check correct shape
        self.assertEqual(df.shape, (3, 3))

        # Check correct column order
        self.assertEqual(list(df.columns), self.expected_column_order)

        # Check values converted to int
        expected_df = pd.DataFrame({
            "P1>P2": [1, 0, 1],
            "P1>P3": [0, 1, 0],
            "P2>P3": [1, 1, 0]
        })
        pd.testing.assert_frame_equal(df, expected_df)

    def test_empty_feature_df(self):
        """
        No features → return empty DataFrame.
        """
        feature_df = pd.DataFrame({'Protein1': [], 'Protein2': []})
        rules = list(zip(feature_df['Protein1'].tolist(), feature_df['Protein2'].tolist()))
        bool_matrix = np.array([])

        df = self.transformer.prep_vectorized_pairs_for_scikitlearn(
            rules, bool_matrix
        )

        self.assertTrue(df.empty)

    def test_empty_bool_arrays(self):
        """
        If bool arrays are empty, output should be an empty DataFrame with correct columns.
        """
        bool_matrix = np.array([
            [],
            [],
            []
        ])

        df = self.transformer.prep_vectorized_pairs_for_scikitlearn(
            self.rules, bool_matrix
        )

        self.assertEqual(list(df.columns), self.expected_column_order)
        self.assertEqual(df.shape[0], 0)



if __name__ == "__main__":
    unittest.main()