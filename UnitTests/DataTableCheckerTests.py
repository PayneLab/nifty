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
        self.quant_df = pd.DataFrame({'sample_id': ['S1', 'S2'], 'P1': [1, 2]})
        self.meta_df = pd.DataFrame({'sample_id': ['S1', 'S2'], 'classification_label': ['A', 'B']})
        self.base_df = pd.DataFrame({
            'sample_id': ['S1', 'S2', 'S3'],
            'P1': [1.0, np.nan, np.nan],
            'P2': [1.0, 2.0, 3.0],
            'P3': [np.nan, np.nan, np.nan]
        })

    # check_meta_file tests.
    def test_check_meta_file_valid(self):
        try:
            valid_df = pd.DataFrame({'sample_id': ['S1'], 'classification_label': ['A']})
            self.assertEqual(self.checker.check_meta_file(valid_df), 0)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_check_meta_file_only_one_column(self):
        try:
            invalid_df1 = pd.DataFrame({'sample_id': ['S1']})
            self.assertEqual(self.checker.check_meta_file(invalid_df1), 1)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_check_meta_file_wrong_first_column(self):
        try:
            invalid_df2 = pd.DataFrame({'SampleID': ['S1'], 'classification_label': ['A']})
            self.assertEqual(self.checker.check_meta_file(invalid_df2), 2)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_check_meta_file_wrong_second_column(self):
        try:
            invalid_df3 = pd.DataFrame({'sample_id': ['S1'], 'label': ['A']})
            self.assertEqual(self.checker.check_meta_file(invalid_df3), 3)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    # check_samples tests.
    def test_check_samples_valid(self):
        try:
            self.assertEqual(self.checker.check_samples(self.quant_df, self.meta_df), 0)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_check_samples_different_sample_count(self):
        try:
            meta_df_fewer = self.meta_df.iloc[:1]
            self.assertEqual(self.checker.check_samples(self.quant_df, meta_df_fewer), 4)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_check_samples_different_sample_id(self):
        try:
            meta_df_wrong_ids = pd.DataFrame({'sample_id': ['X1', 'X2'], 'classification_label': ['A', 'B']})
            self.assertEqual(self.checker.check_samples(self.quant_df, meta_df_wrong_ids), 5)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_check_samples_different_sorting_order(self):
        try:
            quant_df_wrong_order = pd.DataFrame({'sample_id': ['S2', 'S1'], 'P1': [2, 1]})
            self.assertEqual(self.checker.check_samples(quant_df_wrong_order, self.meta_df), 0)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_check_samples_wrong_column_name(self):
        try:
            quant_df_wrong_col = pd.DataFrame({'SampleID': ['S1', 'S2'], 'P1': [1, 2]})
            self.assertEqual(self.checker.check_samples(quant_df_wrong_col, self.meta_df), 2)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    # sort_data tests.
    def test_sort_quant_data(self):
        try:
            quant_df = pd.DataFrame({'sample_id': ['S2', 'S1'], 'P1': [2, 1]})
            meta_df = pd.DataFrame({'sample_id': ['S2', 'S1'], 'classification_label': ['B', 'A']})
            sorted_quant_df, _ = self.checker.sort_data(quant_df, meta_df)

            # Move correct to SetUp?
            expected_quant = pd.DataFrame({'sample_id': ['S1', 'S2'], 'P1': [1, 2]})
            pd.testing.assert_frame_equal(sorted_quant_df, expected_quant)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_sort_meta_data(self):
        try:
            quant_df = pd.DataFrame({'sample_id': ['S2', 'S1'], 'P1': [2, 1]})
            meta_df = pd.DataFrame({'sample_id': ['S2', 'S1'], 'classification_label': ['B', 'A']})
            _, sorted_meta_df = self.checker.sort_data(quant_df, meta_df)

            expected_meta = pd.DataFrame({'sample_id': ['S1', 'S2'], 'classification_label': ['A', 'B']})
            pd.testing.assert_frame_equal(sorted_meta_df, expected_meta)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_sort_already_sort(self):
        # Data is already sorted.
        try:
            quant_df = pd.DataFrame({'sample_id': ['S1', 'S2', 'S3', 'S4', 'S5'], 'P1': [1, 2, 3, 4, 5]})
            meta_df = pd.DataFrame({'sample_id': ['S1', 'S2', 'S3', 'S4', 'S5'], 'classification_label': ['A', 'B', 'C', 'D', 'E']})
            sorted_quant, sorted_meta = self.checker.sort_data(quant_df, meta_df)

            pd.testing.assert_frame_equal(sorted_quant, quant_df)
            pd.testing.assert_frame_equal(sorted_meta, meta_df)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_sort_quant_sorter_meta_unsorted(self):
        # Quant_df is sorted by sample_id, but meta_df is not sorted.
        try:
            quant_df = pd.DataFrame({'sample_id': ['S1', 'S2', 'S3', 'S4', 'S5'], 'P1': [1, 2, 3, 4, 5]})
            meta_df = pd.DataFrame({'sample_id': ['S3', 'S1', 'S2', 'S4', 'S5'], 'classification_label': ['C', 'A', 'B', 'D', 'E']})
            sorted_quant, sorted_meta = self.checker.sort_data(quant_df, meta_df)

            expected_order_meta = pd.DataFrame({'sample_id': ['S1', 'S2', 'S3', 'S4', 'S5'], 'classification_label': ['A', 'B', 'C', 'D', 'E']})
            pd.testing.assert_frame_equal(sorted_quant, quant_df)
            pd.testing.assert_frame_equal(sorted_meta, expected_order_meta)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    # check_quant_data tests.
    def test_check_quant_data_valid(self):
        try:
            df_numeric = pd.DataFrame({'sample_id': ['S1'], 'P1': [1.0], 'P2': [2.0]})
            self.assertEqual(self.checker.check_quant_data(df_numeric), 0)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_check_quant_data_all_nan(self):
        try:
            df_nan = pd.DataFrame({'sample_id': ['S1'], 'P1': [np.nan], 'P2': [np.nan]})
            self.assertEqual(self.checker.check_quant_data(df_nan), 6)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_check_quant_data_mixed_val_nan(self):
        try:
            df_mix = pd.DataFrame({'sample_id': ['S1'], 'P1': [1.0], 'P2': [np.nan]})
            self.assertEqual(self.checker.check_quant_data(df_mix), 0)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_check_quant_data_string(self):
        try:
            df_invalid = pd.DataFrame({'sample_id': ['S1'], 'P1': [1.0], 'P2': ['abc']})
            self.assertEqual(self.checker.check_quant_data(df_invalid), 7)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_check_quant_data_invalid_char(self):
        try:
            df_invalid = pd.DataFrame({'sample_id': ['S1'], 'P1': [1.0], 'P2': ['#']})
            self.assertEqual(self.checker.check_quant_data(df_invalid), 7)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    # check_duplicate_proteins tests.
    def test_check_duplicate_proteins_valid(self):
        try:
            df_unique = pd.DataFrame({'sample_id': ['S1'], 'P1': [1], 'P2': [2]})
            self.assertEqual(self.checker.check_duplicate_proteins(df_unique), 0)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_check_duplicate_proteins_duplicate_protein(self):
        try:
            # Manually create duplicate columns
            df_dup = pd.DataFrame([[1, 2, 3]], columns=['sample_id', 'P1', 'P1'])
            self.assertEqual(self.checker.check_duplicate_proteins(df_dup), 8)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    # check_duplicate_samples tests.
    def test_check_duplicate_samples_valid(self):
        try:
            self.assertEqual(self.checker.check_duplicate_samples(self.quant_df, self.meta_df), 0)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_check_duplicate_samples_duplicate_sample_ID_quant(self):
        try:
            df_quant_dup = pd.DataFrame({'sample_id': ['S1', 'S1'], 'P1': [1, 2]})
            self.assertEqual(self.checker.check_duplicate_samples(df_quant_dup, self.meta_df), 9)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_check_duplicate_samples_duplicate_sample_ID_meta(self):
        try:
            df_meta_dup = pd.DataFrame({'sample_id': ['S1', 'S1'], 'classification_label': ['A', 'B']})
            self.assertEqual(self.checker.check_duplicate_samples(self.quant_df, df_meta_dup), 14)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    # filter_proteins tests.
    def test_filter_proteins_50(self):
        try:
            expected_df = pd.DataFrame({
                'sample_id': ['S1', 'S2', 'S3'],
                'P2': [1.0, 2.0, 3.0]
            })
            filtered_df = self.checker.filter_proteins(self.base_df, fraction_na=0.5)
            pd.testing.assert_frame_equal(filtered_df.reset_index(drop=True), expected_df.reset_index(drop=True))
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_filter_proteins_90(self):
        try:
            expected_df_90 = pd.DataFrame({
                'sample_id': ['S1', 'S2', 'S3'],
                'P1': [1.0, np.nan, np.nan],
                'P2': [1.0, 2.0, 3.0]
            })
            filtered_df_90 = self.checker.filter_proteins(self.base_df, fraction_na=0.9)
            pd.testing.assert_frame_equal(filtered_df_90.reset_index(drop=True), expected_df_90.reset_index(drop=True))
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_filter_proteins_10(self):
        try:
            expected_df = pd.DataFrame({
                'sample_id': ['S1', 'S2', 'S3'],
                'P2': [1.0, 2.0, 3.0]
            })
            filtered_df_10 = self.checker.filter_proteins(self.base_df, fraction_na=0.1)
            pd.testing.assert_frame_equal(filtered_df_10.reset_index(drop=True), expected_df.reset_index(drop=True))
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_filter_proteins_all_proteins_greater_50_na(self):
        try:
            df = pd.DataFrame({
                'sample_id': ['S1', 'S2', 'S3', 'S4'],
                'P1': [np.nan, np.nan, 1.0, np.nan],
                'P2': [np.nan, 2.0, np.nan, np.nan],
                'P3': [np.nan, np.nan, np.nan, 5.0]
            })
            self.assertEqual(self.checker.filter_proteins(df, 0.5), 10)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    # check_enough_samples tests.
    def test_check_enough_samples_valid(self):
        try:
            df_valid = pd.DataFrame({
                'sample_id': [f'S{i}' for i in range(30)],
                'classification_label': ['A'] * 15 + ['B'] * 15
            })
            self.assertEqual(self.checker.check_enough_samples(df_valid, min_samples=15), 0)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_check_enough_samples_below_min(self):
        try:
            df_invalid = pd.DataFrame({
                'sample_id': [f'S{i}' for i in range(10)],
                'classification_label': ['A'] * 5 + ['B'] * 5
            })
            self.assertEqual(self.checker.check_enough_samples(df_invalid, min_samples=6), 11)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_check_enough_samples_protein_b_not_enough(self):
        try:
            df_invalid = pd.DataFrame({
                'sample_id': [f'S{i}' for i in range(20)],
                'classification_label': ['A'] * 15 + ['B'] * 5
            })
            self.assertEqual(self.checker.check_enough_samples(df_invalid, min_samples=6), 11)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_check_enough_samples_protein_a_enough(self):
        try:
            df_invalid = pd.DataFrame({
                'sample_id': [f'S{i}' for i in range(20)],
                'classification_label': ['A'] * 5 + ['B'] * 15
            })
            self.assertEqual(self.checker.check_enough_samples(df_invalid, min_samples=6), 11)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_check_enough_samples_min_25(self):
        try:
            df_invalid = pd.DataFrame({
                'sample_id': [f'S{i}' for i in range(70)],
                'classification_label': ['A'] * 35 + ['B'] * 35
            })
            self.assertEqual(self.checker.check_enough_samples(df_invalid, min_samples=25), 0)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    # check_protein_amount tests.
    def test_check_protein_amount_valid(self):
        try:
            df_valid = pd.DataFrame({
            'sample_id': ['S1', 'S2'],
            'P1': [1.0, 2.0],
            'P2': [3.0, 4.0]
            })
            self.assertEqual(self.checker.check_protein_amount(df_valid, min_proteins=2), 0)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_check_protein_amount_not_enough_quant(self):
        try:
            df_invalid = pd.DataFrame({
                'sample_id': ['S1', 'S2'],
                'P1': [1.0, 2.0]
            })
            self.assertEqual(self.checker.check_protein_amount(df_invalid, min_proteins=3), 12)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def check_set_quant_index(self):
        try:
            df = pd.DataFrame({
                'sample_id': ['S1', 'S2'],
                'P1': [1.0, 2.0],
                'P2': [3.0, 4.0]
            })
            df_indexed = self.checker.set_quant_index(df)
            self.assertTrue(df_indexed.index.name == 'sample_id')
            self.assertTrue('sample_id' not in df_indexed.columns)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")
    
    
    # filter_proteins_by_class tests.

    def test_filters_high_nan_proteins(self):
        result = self.checker.filter_proteins_by_class(
            self.base_df.set_index("sample_id"),
            self.meta_df.set_index("sample_id"),
            fraction_na=0.5
        )
        self.assertIn("P1", result.columns)   # allowed since ~67% missing but class-wise check may keep
        self.assertIn("P2", result.columns)   # clean
        self.assertNotIn("P3", result.columns) # 100% NaN

    def test_respects_fraction_threshold(self):
        result = self.checker.filter_proteins_by_class(
            self.base_df.set_index("sample_id"),
            self.meta_df.set_index("sample_id"),
            fraction_na=0.25
        )
        self.assertIn("P1", result.columns)   # stays, because Class A passes
        self.assertIn("P2", result.columns)   # clean, always passes
        self.assertNotIn("P3", result.columns) # always fails (100% NaN)


    def test_proteins_to_keep(self):
        result = self.checker.filter_proteins_by_class(
            self.base_df.set_index("sample_id"),
            self.meta_df.set_index("sample_id"),
            fraction_na=0.25,
            proteins_to_keep=["P1"]
        )
        self.assertIn("P1", result.columns)  # forced keep

    def test_returns_10_if_empty(self):
        df_all_nan = pd.DataFrame({
            "sample_id": ["S1", "S2"],
            "P1": [np.nan, np.nan],
            "P2": [np.nan, np.nan]
        }).set_index("sample_id")

        result = self.checker.filter_proteins_by_class(
            df_all_nan,
            self.meta_df.set_index("sample_id"),
            fraction_na=0.0
        )
        self.assertEqual(result, 10)

if __name__ == "__main__":
    unittest.main()
