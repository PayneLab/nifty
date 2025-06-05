import unittest
import pandas as pd
import numpy as np
import sys
import os

current_dir = os.path.dirname(__file__)

parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from DataTableChecker import DataTableChecker

class TestDataTableChecker(unittest.TestCase):
    def setUp(self):
        self.checker = DataTableChecker()

    def test_check_meta_file(self):
        # Valid dataframe
        valid_df = pd.DataFrame({'sample_id': ['S1'], 'classification_label': ['A']})
        self.assertEqual(self.checker.check_meta_file(valid_df), 0)

        # Invalid: only one column
        invalid_df1 = pd.DataFrame({'sample_id': ['S1']})
        self.assertEqual(self.checker.check_meta_file(invalid_df1), 1)

        # Invalid: wrong first column label
        invalid_df2 = pd.DataFrame({'SampleID': ['S1'], 'classification_label': ['A']})
        self.assertEqual(self.checker.check_meta_file(invalid_df2), 2)

        # Invalid: wrong second column label
        invalid_df3 = pd.DataFrame({'sample_id': ['S1'], 'label': ['A']})
        self.assertEqual(self.checker.check_meta_file(invalid_df3), 3)

        # Extra spaces but correct column names?

    def test_check_samples(self):
        quant_df = pd.DataFrame({'sample_id': ['S1', 'S2'], 'P1': [1, 2]})
        meta_df = pd.DataFrame({'sample_id': ['S1', 'S2'], 'classification_label': ['A', 'B']})
        self.assertEqual(self.checker.check_samples(quant_df, meta_df), 0)

        # Different sample counts
        meta_df_fewer = meta_df.iloc[:1]
        self.assertEqual(self.checker.check_samples(quant_df, meta_df_fewer), 4)

        # Different sample IDs
        meta_df_wrong_ids = pd.DataFrame({'sample_id': ['X1', 'X2'], 'classification_label': ['A', 'B']})
        self.assertEqual(self.checker.check_samples(quant_df, meta_df_wrong_ids), 5)

        # Different sorting order
        quant_df_wrong_order = pd.DataFrame({'sample_id': ['S2', 'S1'], 'P1': [2, 1]})
        self.assertEqual(self.checker.check_samples(quant_df_wrong_order, meta_df), 0)

        # Wrong column name in quant
        quant_df_wrong_col = pd.DataFrame({'SampleID': ['S1', 'S2'], 'P1': [1, 2]})
        self.assertEqual(self.checker.check_samples(quant_df_wrong_col, meta_df), 2)

    def test_check_quant_data(self):
        # All numeric
        df_numeric = pd.DataFrame({'sample_id': ['S1'], 'P1': [1.0], 'P2': [2.0]})
        self.assertEqual(self.checker.check_quant_data(df_numeric), 0)

        # All NaN
        df_nan = pd.DataFrame({'sample_id': ['S1'], 'P1': [np.nan], 'P2': [np.nan]})
        self.assertEqual(self.checker.check_quant_data(df_nan), 7)

        # Mix valid + NaN
        df_mix = pd.DataFrame({'sample_id': ['S1'], 'P1': [1.0], 'P2': [np.nan]})
        self.assertEqual(self.checker.check_quant_data(df_mix), 0)

        # Invalid string value
        df_invalid = pd.DataFrame({'sample_id': ['S1'], 'P1': [1.0], 'P2': ['abc']})
        self.assertEqual(self.checker.check_quant_data(df_invalid), 8)

        # Invalid character
        df_invalid = pd.DataFrame({'sample_id': ['S1'], 'P1': [1.0], 'P2': ['#']})
        self.assertEqual(self.checker.check_quant_data(df_invalid), 8)

    def test_check_duplicate_proteins(self):
        df_unique = pd.DataFrame({'sample_id': ['S1'], 'P1': [1], 'P2': [2]})
        self.assertEqual(self.checker.check_duplicate_proteins(df_unique), 0)

        # Manually create duplicate columns
        df_dup = pd.DataFrame([[1, 2, 3]], columns=['sample_id', 'P1', 'P1'])
        self.assertEqual(self.checker.check_duplicate_proteins(df_dup), 9)

    def test_check_duplicate_samples(self):
        df_quant = pd.DataFrame({'sample_id': ['S1', 'S2'], 'P1': [1, 2]})
        df_meta = pd.DataFrame({'sample_id': ['S1', 'S2'], 'classification_label': ['A', 'B']})
        self.assertEqual(self.checker.check_duplicate_samples(df_quant, df_meta), 0)

        df_quant_dup = pd.DataFrame({'sample_id': ['S1', 'S1'], 'P1': [1, 2]})
        self.assertEqual(self.checker.check_duplicate_samples(df_quant_dup, df_meta), 10)

        df_meta_dup = pd.DataFrame({'sample_id': ['S1', 'S1'], 'classification_label': ['A', 'B']})
        self.assertEqual(self.checker.check_duplicate_samples(df_quant, df_meta_dup), 10)

    def test_filter_proteins(self):
        df = pd.DataFrame({
            'sample_id': ['S1', 'S2', 'S3'],
            'P1': [1.0, np.nan, np.nan],       
            'P2': [1.0, 2.0, 3.0],            
            'P3': [np.nan, np.nan, np.nan]      
        })

        # Expected output after filtering
        expected_df = pd.DataFrame({
            'sample_id': ['S1', 'S2', 'S3'],
            'P2': [1.0, 2.0, 3.0]
        })

        # Test filtering with 50% threshold
        filtered_df = self.checker.filter_proteins(df, fraction_na=0.5)
        pd.testing.assert_frame_equal(filtered_df.reset_index(drop=True), expected_df.reset_index(drop=True))

        # Test with 90% threshold
        expected_df_90 = pd.DataFrame({
            'sample_id': ['S1', 'S2', 'S3'],
            'P1': [1.0, np.nan, np.nan],
            'P2': [1.0, 2.0, 3.0]
        })
        filtered_df_90 = self.checker.filter_proteins(df, fraction_na=0.9)
        pd.testing.assert_frame_equal(filtered_df_90.reset_index(drop=True), expected_df_90.reset_index(drop=True))

        # Test with 10% threshold (should return all proteins)
        filtered_df_10 = self.checker.filter_proteins(df, fraction_na=0.1)
        pd.testing.assert_frame_equal(filtered_df_10.reset_index(drop=True), expected_df.reset_index(drop=True))

        # Test empty resultant DataFrame
        # All proteins have >50% missing values
        df = pd.DataFrame({
            'sample_id': ['S1', 'S2', 'S3', 'S4'],
            'P1': [np.nan, np.nan, 1.0, np.nan],
            'P2': [np.nan, 2.0, np.nan, np.nan],
            'P3': [np.nan, np.nan, np.nan, 5.0]
        })

        result = self.checker.filter_proteins(df)

        # Shoult return empty df
        self.assertEqual(result, 11)

    def test_check_enough_samples(self):
        df_valid = pd.DataFrame({
            'sample_id': [f'S{i}' for i in range(30)],
            'classification_label': ['A'] * 15 + ['B'] * 15
        })
        self.assertEqual(self.checker.check_enough_samples(df_valid, min_samples=15), 0)

        df_invalid = pd.DataFrame({
            'sample_id': [f'S{i}' for i in range(10)],
            'classification_label': ['A'] * 5 + ['B'] * 5
        })
        self.assertEqual(self.checker.check_enough_samples(df_invalid, min_samples=6), 12)

        # Protein A is good, Protein B is not
        df_invalid = pd.DataFrame({
            'sample_id': [f'S{i}' for i in range(20)],
            'classification_label': ['A'] * 15 + ['B'] * 5
        })
        self.assertEqual(self.checker.check_enough_samples(df_invalid, min_samples=6), 12)

        # B is good, A is not
        df_invalid = pd.DataFrame({
            'sample_id': [f'S{i}' for i in range(20)],
            'classification_label': ['A'] * 5 + ['B'] * 15
        })
        self.assertEqual(self.checker.check_enough_samples(df_invalid, min_samples=6), 12)
        
        # Min Samples is 25
        df_invalid = pd.DataFrame({
            'sample_id': [f'S{i}' for i in range(70)],
            'classification_label': ['A'] * 35 + ['B'] * 35
        })
        self.assertEqual(self.checker.check_enough_samples(df_invalid, min_samples=25), 0)
    
    def test_check_protein_amount(self):
        df_valid = pd.DataFrame({
            'sample_id': ['S1', 'S2'],
            'P1': [1.0, 2.0],
            'P2': [3.0, 4.0]
        })
        self.assertEqual(self.checker.check_protein_amount(df_valid, min_proteins=2), 0)

        df_invalid = pd.DataFrame({
            'sample_id': ['S1', 'S2'],
            'P1': [1.0, 2.0]
        })
        self.assertEqual(self.checker.check_protein_amount(df_invalid, min_proteins=3), 13)

if __name__ == "__main__":
    unittest.main()
