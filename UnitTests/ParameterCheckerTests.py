import sys
import os

import unittest
from unittest.mock import patch
from pathlib import Path
import pandas as pd

current_dir = os.path.dirname(__file__)

parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from ParameterChecker import ParameterChecker

class TestParameterChecker(unittest.TestCase):
    def setUp(self):
        self.checker = ParameterChecker()
        self.parser = self.checker.set_up_parser()

        # create a quant and meta files that are good
        Path(os.path.join("UnitTests", "quant.tsv")).touch()
        Path(os.path.join("UnitTests", "meta.tsv")).touch()
        # create a quant and meta files that are bad
        Path(os.path.join("UnitTests", "quant.txt")).touch()
        Path(os.path.join("UnitTests", "meta.txt")).touch()
        # create an output path that's good
        Path(os.path.join("UnitTests", "Test_Output")).mkdir(exist_ok=True)

    # check quant file
    @patch('sys.exit')
    def test_bad_quant_path(self, mock_exit):
        try:
            quant_file_path = os.path.join("quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path]
            args = self.parser.parse_args(args)

            self.checker.check_arguments(args)

            mock_exit.assert_called_once_with(1)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    @patch('sys.exit')
    def test_bad_quant_file_extension(self,mock_exit):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.txt")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path]
            args = self.parser.parse_args(args)

            self.checker.check_arguments(args)

            mock_exit.assert_called_once_with(1)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    # check meta file
    @patch('sys.exit')
    def test_bad_meta_path(self, mock_exit):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path]
            args = self.parser.parse_args(args)

            self.checker.check_arguments(args)

            mock_exit.assert_called_once_with(1)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    @patch('sys.exit')
    def test_bad_meta_file_extension(self, mock_exit):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.txt")
            args = ['-q', quant_file_path, '-m', meta_file_path]
            args = self.parser.parse_args(args)

            self.checker.check_arguments(args)

            mock_exit.assert_called_once_with(1)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    # check output directory
    @patch('sys.exit')
    def test_bad_output_path(self, mock_exit):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            output_directory = os.path.join("Test_Output")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-o', output_directory]
            args = self.parser.parse_args(args)

            self.checker.check_arguments(args)

            mock_exit.assert_called_once_with(1)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_good_output_path(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            output_directory = os.path.join("UnitTests", "Test_Output")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-o', output_directory]
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.output, output_directory)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    # check defaults
    def test_defaults(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path]
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.quant, quant_file_path)
            self.assertEqual(args.meta, meta_file_path)
            self.assertEqual(args.output, os.getcwd())
            self.assertEqual(args.k, 50)
            self.assertEqual(args.disjoint, False)
            self.assertEqual(args.mutual_info, True)
            self.assertEqual(args.mi_cutoff, 0.7)
            self.assertEqual(args.missingness_cutoff, 0.5)
            self.assertEqual(args.min_sample_per_class, 15)
            self.assertEqual(args.seed, None)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    # check k
    def test_k_below_1(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-k', '-1']
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.k, 50)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_k_above_50(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-k', '1000']
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.k, 50)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_k_equals_1(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-k', '1']
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.k, 1)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_k_equals_50(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-k', '50']
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.k, 50)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_k_equals_15(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-k', '15']
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.k, 15)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_k_equals_5(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-k', '5']
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.k, 5)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    # check disjoint and mutual info
    def test_disjoint_only(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-d', '-mi']
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.disjoint, True)
            self.assertEqual(args.mutual_info, False)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_mutual_info_only(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path]
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.disjoint, False)
            self.assertEqual(args.mutual_info, True)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_disjoint_and_mutual_info(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-d']
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.disjoint, True)
            self.assertEqual(args.mutual_info, True)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_neither_disjoint_nor_mutual_info(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path,'-mi']
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.disjoint, False)
            self.assertEqual(args.mutual_info, False)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    # check mi cutoff
    def test_mi_cutoff_below_0(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-mic', '-1']
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.mi_cutoff, 0.7)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_mi_cutoff_above_1(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-mic', '2']
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.mi_cutoff, 0.7)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_mi_cutoff_equals_0(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-mic', '0']
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.mi_cutoff, 0)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_mi_cutoff_equals_1(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-mic', '1']
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.mi_cutoff, 1)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_mi_cutoff_equals_03(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-mic', '0.3']
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.mi_cutoff, 0.3)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_mi_cutoff_equals_09(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-mic', '0.9']
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.mi_cutoff, 0.9)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    # check missingness cutoff
    def test_missingness_cutoff_below_0(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-mc', '-1']
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.missingness_cutoff, 0.5)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_missingness_cutoff_above_1(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-mc', '2']
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.missingness_cutoff, 0.5)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_missingness_cutoff_equals_0(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-mc', '0']
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.missingness_cutoff, 0)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_missingness_cutoff_equals_1(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-mc', '1']
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.missingness_cutoff, 1)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_missingness_cutoff_equals_03(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-mc', '0.3']
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.missingness_cutoff, 0.3)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_missingness_cutoff_equals_07(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-mc', '0.7']
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.missingness_cutoff, 0.7)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    # check min samples
    def test_min_samples_below_1(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-ms', '0']
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.min_sample_per_class, 15)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_min_samples_equals_1(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-ms', '1']
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.min_sample_per_class, 1)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_min_samples_equals_10(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-ms', '10']
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.min_sample_per_class, 10)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_min_samples_equals_2000(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-ms', '2000']
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.min_sample_per_class, 2000)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    # check seed
    def test_seed_42(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-s', '42']
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.seed, 42)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_seed_100(self):
        try:
            quant_file_path = os.path.join("UnitTests", "quant.tsv")
            meta_file_path = os.path.join("UnitTests", "meta.tsv")
            args = ['-q', quant_file_path, '-m', meta_file_path, '-s', '100']
            args = self.parser.parse_args(args)

            args = self.checker.check_arguments(args)

            self.assertEqual(args.seed, 100)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")


if __name__ == "__main__":
    unittest.main()
