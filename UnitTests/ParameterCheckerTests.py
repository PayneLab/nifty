import sys
import os

import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
import pickle

current_dir = os.path.dirname(__file__)

parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from ParameterChecker import ParameterChecker

# Test check_arguments()
class TestCheckArguments(unittest.TestCase):

    def setUp(self):
        self.checker = ParameterChecker()

    @patch('sys.exit')
    def test_bad_config_path(self, mock_exit):
        parser = self.checker.set_up_parser()
        args = parser.parse_args()

        self.checker.check_arguments(args)

        mock_exit.assert_called_once_with(1)

    @patch('sys.exit')
    def test_bad_config_extension(self, mock_exit):
        with TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.txt")
            with open(config_path, "w") as config:
                config.write("find_features = true  # required\n")

            parser = self.checker.set_up_parser()
            args = parser.parse_args(['-c', config_path])

            self.checker.check_arguments(args)

            mock_exit.assert_called_once_with(1)

    @patch('sys.exit')
    def test_bad_config_contents(self, mock_exit):
        with TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.toml")
            with open(config_path, "w") as config:
                config.write("find_features =   # required\n")

            parser = self.checker.set_up_parser()
            args = parser.parse_args(['-c', config_path])

            self.checker.check_arguments(args)

            mock_exit.assert_called_once_with(1)

    def test_good_config(self):
        with TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.toml")
            with open(config_path, "w") as config:
                config.write("find_features = true  # required\n")

            parser = self.checker.set_up_parser()
            args = parser.parse_args(['-c', config_path])

            expected = {'find_features': True}

            self.assertEqual(self.checker.check_arguments(args), expected)


# Test read_tsv()
class TestReadTsv(unittest.TestCase):

    def setUp(self):
        self.checker = ParameterChecker()

    @patch('sys.exit')
    def test_parse_error_mismatched_numer_of_tabs(self, mock_exit):
        with TemporaryDirectory() as tmpdir:
            tsv_path = os.path.join(tmpdir, "test.tsv")
            with open(tsv_path, "w") as tsv:
                tsv.write("Header\tHeader2\tHeader3\n")
                tsv.write("Value\tValue2\tValue3\n")
                tsv.write("Value\tValue2\tValue3\tValue4\n")

            self.checker.read_tsv(tsv_path)

            mock_exit.assert_called_once_with(1)

    @patch('sys.exit')
    def test_parse_error_unexpected_line_ending(self, mock_exit):
        with TemporaryDirectory() as tmpdir:
            tsv_path = os.path.join(tmpdir, "test.tsv")
            with open(tsv_path, "w") as tsv:
                tsv.write("Header\tHeader2\t\"Header3\n")
                tsv.write("Value\tValue2\tValue3\n")
                tsv.write("Value\tValue2\tValue3\n")

            self.checker.read_tsv(tsv_path)

            mock_exit.assert_called_once_with(1)

    @patch('sys.exit')
    def test_parse_error_bad_encoding(self, mock_exit):
        with TemporaryDirectory() as tmpdir:
            tsv_path = os.path.join(tmpdir, "test.tsv")
            with open(tsv_path, "w", encoding="utf-16") as tsv:
                tsv.write("Header\tHeader2\tHeader3\n")
                tsv.write("Value\tValue2\tValue3\n")
                tsv.write("Value\tValue2\tValue3\n")

            self.checker.read_tsv(tsv_path)

            mock_exit.assert_called_once_with(1)

    def test_good_tsv(self):
        with TemporaryDirectory() as tmpdir:
            tsv_path = os.path.join(tmpdir, "test.tsv")
            with open(tsv_path, "w") as tsv:
                tsv.write("Header\tHeader2\tHeader3\n")
                tsv.write("Value\tValue2\tValue3\n")
                tsv.write("Value\tValue2\tValue3\n")

            expected = pd.DataFrame({'Header': ['Value', 'Value'], 'Header2': ['Value2', 'Value2'], 'Header3': ['Value3', 'Value3']})

            self.assertEqual(expected.equals(self.checker.read_tsv(tsv_path)), True)


# Test read_pkl()
class TestReadPkl(unittest.TestCase):

    def setUp(self):
        self.checker = ParameterChecker()

    @patch('sys.exit')
    def test_unpickling_error_truncated_pkl_file(self, mock_exit):
        with TemporaryDirectory() as tmpdir:
            pkl_path = os.path.join(tmpdir, "test.pkl")
            with open(pkl_path, "wb") as pkl:
                pkl.write(b"Bad pickle file.")

            self.checker.read_pkl(pkl_path)

            mock_exit.assert_called_once_with(1)

    @patch('sys.exit')
    def test_unpickling_error_modified_pkl_data(self, mock_exit):
        with TemporaryDirectory() as tmpdir:
            data = {'a': 1, 'b': 2}
            pkl_path = os.path.join(tmpdir, "test.pkl")

            with open(pkl_path, "wb") as pkl:
                pickle.dump(data, pkl)

            with open(pkl_path, "r+b") as pkl:
                pkl.seek(10)
                pkl.write(b"ABC123")

            self.checker.read_pkl(pkl_path)

            mock_exit.assert_called_once_with(1)

    def test_good_pkl(self):
        with TemporaryDirectory() as tmpdir:
            data = {'a': 1, 'b': 2}
            pkl_path = os.path.join(tmpdir, "test.pkl")

            with open(pkl_path, "wb") as pkl:
                pickle.dump(data, pkl)

            self.assertEqual(self.checker.read_pkl(pkl_path), data)


# Test check_configurations_project_settings()
class TestCheckConfigurationsProjectSettings(unittest.TestCase):

    def setUp(self):
        self.checker = ParameterChecker()

        self.configs = {'find_features': True, 
                        'train_model': True, 
                        'apply_model': True, 
                        'seed': 'random', 
                        'input_files': 'reference'}

    @patch('sys.exit')
    def test_find_features_not_bool(self, mock_exit):
        self.configs['find_features'] = "not a bool"

        self.checker.check_configurations_project_settings(configs=self.configs)

        mock_exit.assert_called_once_with(1)

    @patch('sys.exit')
    def test_find_features_not_bool2(self, mock_exit):
        self.configs['find_features'] = 0

        self.checker.check_configurations_project_settings(configs=self.configs)

        mock_exit.assert_called_once_with(1)

    @patch('sys.exit')
    def test_train_model_not_bool(self, mock_exit):
        self.configs['train_model'] = "not a bool"

        self.checker.check_configurations_project_settings(configs=self.configs)

        mock_exit.assert_called_once_with(1)

    @patch('sys.exit')
    def test_train_model_not_bool2(self, mock_exit):
        self.configs['train_model'] = 1

        self.checker.check_configurations_project_settings(configs=self.configs)

        mock_exit.assert_called_once_with(1)

    @patch('sys.exit')
    def test_apply_model_not_bool(self, mock_exit):
        self.configs['apply_model'] = "not a bool"

        self.checker.check_configurations_project_settings(configs=self.configs)

        mock_exit.assert_called_once_with(1)

    @patch('sys.exit')
    def test_apply_model_not_bool2(self, mock_exit):
        self.configs['apply_model'] = 2

        self.checker.check_configurations_project_settings(configs=self.configs)

        mock_exit.assert_called_once_with(1)

    @patch('sys.exit')
    def test_find_train_apply_all_false(self, mock_exit):
        self.configs['find_features'] = False
        self.configs['train_model'] = False
        self.configs['apply_model'] = False

        self.checker.check_configurations_project_settings(configs=self.configs)

        mock_exit.assert_called_once_with(1)

    def test_seed_random(self):
        self.checker.check_configurations_project_settings(configs=self.configs)

        self.assertEqual(self.configs['seed'], None)

    def test_seed_42(self):
        self.configs['seed'] = 42

        self.checker.check_configurations_project_settings(configs=self.configs)

        self.assertEqual(self.configs['seed'], 42)

    def test_seed_100(self):
        self.configs['seed'] = 100

        self.checker.check_configurations_project_settings(configs=self.configs)

        self.assertEqual(self.configs['seed'], 100)

    def test_bad_seed(self):
        self.configs['seed'] = "bad seed"

        self.checker.check_configurations_project_settings(configs=self.configs)

        self.assertEqual(self.configs['seed'], None)

    def test_bad_seed2(self):
        self.configs['seed'] = 100.0

        self.checker.check_configurations_project_settings(configs=self.configs)

        self.assertEqual(self.configs['seed'], None)

    @patch('sys.exit')
    def test_bad_input_files(self, mock_exit):
        self.configs['input_files'] = "bas string"

        self.checker.check_configurations_project_settings(configs=self.configs)

        mock_exit.assert_called_once_with(1)

    @patch('sys.exit')
    def test_bad_input_files(self, mock_exit):
        self.configs['input_files'] = 100

        self.checker.check_configurations_project_settings(configs=self.configs)

        mock_exit.assert_called_once_with(1)


# TODO: Test check_configurations_files()
class CheckConfigurationsFiles(unittest.TestCase):

    def setUp(self):
        self.checker = ParameterChecker()

        self.configs = {'find_features': True, 
                        'train_model': True, 
                        'apply_model': True, 
                        'input_files': 'reference'}
        

# TODO: Test check_configurations_feature_selection()
class CheckConfigurationsFeatureSelection(unittest.TestCase):

    def setUp(self):
        self.checker = ParameterChecker()

        self.configs = {'find_features': True, 
                        'train_model': True, 
                        'apply_model': True, 
                        'input_files': 'reference'}


# TODO: Test check_configurations_model_training()
class CheckConfigurationsModelTraining(unittest.TestCase):

    def setUp(self):
        self.checker = ParameterChecker()

        self.configs = {'find_features': True, 
                        'train_model': True, 
                        'apply_model': True, 
                        'input_files': 'reference'}
        

# TODO: Test check_configurations_experimental_classification()
class CheckConfigurationsExperimentalClassification(unittest.TestCase):

    def setUp(self):
        self.checker = ParameterChecker()

        self.configs = {'find_features': True, 
                        'train_model': True, 
                        'apply_model': True, 
                        'input_files': 'reference'}


if __name__ == "__main__":
    unittest.main()
