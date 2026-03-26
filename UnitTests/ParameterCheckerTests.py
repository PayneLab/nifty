import sys
import os

import unittest
from tempfile import TemporaryDirectory
from sklearn.base import BaseEstimator
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

    def test_bad_config_path(self):
        parser = self.checker.set_up_parser()
        args = parser.parse_args()

        with self.assertRaises(SystemExit) as e:
            self.checker.check_arguments(args)

        self.assertEqual(e.exception.code, 1)

    def test_bad_config_extension(self):
        with TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.txt")
            with open(config_path, "w") as config:
                config.write("find_features = true  # required\n")

            parser = self.checker.set_up_parser()
            args = parser.parse_args(['-c', config_path])

            with self.assertRaises(SystemExit) as e:
                self.checker.check_arguments(args)

            self.assertEqual(e.exception.code, 1)

    def test_bad_config_contents(self):
        with TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.toml")
            with open(config_path, "w") as config:
                config.write("find_features =   # required\n")

            parser = self.checker.set_up_parser()
            args = parser.parse_args(['-c', config_path])

            with self.assertRaises(SystemExit) as e:
                self.checker.check_arguments(args)

            self.assertEqual(e.exception.code, 1)

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

    def test_parse_error_mismatched_numer_of_tabs(self):
        with TemporaryDirectory() as tmpdir:
            tsv_path = os.path.join(tmpdir, "test.tsv")
            with open(tsv_path, "w") as tsv:
                tsv.write("Header\tHeader2\tHeader3\n")
                tsv.write("Value\tValue2\tValue3\n")
                tsv.write("Value\tValue2\tValue3\tValue4\n")

            with self.assertRaises(SystemExit) as e:
                self.checker.read_tsv(tsv_path)

            self.assertEqual(e.exception.code, 1)

    def test_parse_error_unexpected_line_ending(self):
        with TemporaryDirectory() as tmpdir:
            tsv_path = os.path.join(tmpdir, "test.tsv")
            with open(tsv_path, "w") as tsv:
                tsv.write("Header\tHeader2\t\"Header3\n")
                tsv.write("Value\tValue2\tValue3\n")
                tsv.write("Value\tValue2\tValue3\n")

            with self.assertRaises(SystemExit) as e:
                self.checker.read_tsv(tsv_path)

            self.assertEqual(e.exception.code, 1)

    def test_parse_error_bad_encoding(self):
        with TemporaryDirectory() as tmpdir:
            tsv_path = os.path.join(tmpdir, "test.tsv")
            with open(tsv_path, "w", encoding="utf-16") as tsv:
                tsv.write("Header\tHeader2\tHeader3\n")
                tsv.write("Value\tValue2\tValue3\n")
                tsv.write("Value\tValue2\tValue3\n")

            with self.assertRaises(SystemExit) as e:
                self.checker.read_tsv(tsv_path)

            self.assertEqual(e.exception.code, 1)

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

    def test_unpickling_error_truncated_pkl_file(self):
        with TemporaryDirectory() as tmpdir:
            pkl_path = os.path.join(tmpdir, "test.pkl")
            with open(pkl_path, "wb") as pkl:
                pkl.write(b"Bad pickle file.")

            with self.assertRaises(SystemExit) as e:
                self.checker.read_pkl(pkl_path)

            self.assertEqual(e.exception.code, 1)

    def test_unpickling_error_modified_pkl_data(self):
        with TemporaryDirectory() as tmpdir:
            data = {'a': 1, 'b': 2}
            pkl_path = os.path.join(tmpdir, "test.pkl")

            with open(pkl_path, "wb") as pkl:
                pickle.dump(data, pkl)

            with open(pkl_path, "r+b") as pkl:
                pkl.seek(10)
                pkl.write(b"ABC123")

            with self.assertRaises(SystemExit) as e:
                self.checker.read_pkl(pkl_path)

            self.assertEqual(e.exception.code, 1)

    def test_good_pkl_bad_contents(self):
        with TemporaryDirectory() as tmpdir:
            data = {'a': 1, 'b': 2}
            pkl_path = os.path.join(tmpdir, "test.pkl")

            with open(pkl_path, "wb") as pkl:
                pickle.dump(data, pkl)

            with self.assertRaises(SystemExit) as e:
                self.checker.read_pkl(pkl_path)

            self.assertEqual(e.exception.code, 1)

    def test_good_pickle_good_contents(self):
        with TemporaryDirectory() as tmpdir:
            data = {'model': BaseEstimator(), 'sklearn_version': 2}
            model = data['model']
            model._sklearn_version = data['sklearn_version']
            pkl_path = os.path.join(tmpdir, "test.pkl")

            with open(pkl_path, "wb") as pkl:
                pickle.dump(data, pkl)

            loaded_model = self.checker.read_pkl(pkl_path)

            self.assertIs(type(model), type(loaded_model))
            self.assertEqual(model.get_params(), loaded_model.get_params())
            self.assertEqual(model._sklearn_version, loaded_model._sklearn_version)


# Test check_configurations_project_settings()
class TestCheckConfigurationsProjectSettings(unittest.TestCase):

    def setUp(self):
        self.checker = ParameterChecker()

        self.configs = {'find_features': True, 
                        'train_model': True, 
                        'apply_model': True, 
                        'seed': 'random', 
                        'input_files': 'reference'}

    def test_find_features_string(self):
        self.configs['find_features'] = "not a bool"

        with self.assertRaises(SystemExit) as e:
            self.checker.check_configurations_project_settings(configs=self.configs)

        self.assertEqual(e.exception.code, 1)

    def test_find_features_int(self):
        self.configs['find_features'] = 0

        with self.assertRaises(SystemExit) as e:
            self.checker.check_configurations_project_settings(configs=self.configs)

        self.assertEqual(e.exception.code, 1)

    def test_train_model_string(self):
        self.configs['train_model'] = "not a bool"

        with self.assertRaises(SystemExit) as e:
            self.checker.check_configurations_project_settings(configs=self.configs)

        self.assertEqual(e.exception.code, 1)

    def test_train_model_int(self):
        self.configs['train_model'] = 1

        with self.assertRaises(SystemExit) as e:
            self.checker.check_configurations_project_settings(configs=self.configs)

        self.assertEqual(e.exception.code, 1)

    def test_apply_model_string(self):
        self.configs['apply_model'] = "not a bool"

        with self.assertRaises(SystemExit) as e:
            self.checker.check_configurations_project_settings(configs=self.configs)

        self.assertEqual(e.exception.code, 1)

    def test_apply_model_int(self):
        self.configs['apply_model'] = 2

        with self.assertRaises(SystemExit) as e:
            self.checker.check_configurations_project_settings(configs=self.configs)

        self.assertEqual(e.exception.code, 1)

    def test_find_train_apply_all_false(self):
        self.configs['find_features'] = False
        self.configs['train_model'] = False
        self.configs['apply_model'] = False

        with self.assertRaises(SystemExit) as e:
            self.checker.check_configurations_project_settings(configs=self.configs)

        self.assertEqual(e.exception.code, 1)

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

    def test_seed_bad_string(self):
        self.configs['seed'] = "bad seed"

        self.checker.check_configurations_project_settings(configs=self.configs)

        self.assertEqual(self.configs['seed'], None)

    def test_seed_float(self):
        self.configs['seed'] = 100.0

        self.checker.check_configurations_project_settings(configs=self.configs)

        self.assertEqual(self.configs['seed'], None)

    def test_seed_bool(self):
        self.configs['seed'] = False

        self.checker.check_configurations_project_settings(configs=self.configs)

        self.assertEqual(self.configs['seed'], None)

    def test_input_files_bad_string(self):
        self.configs['input_files'] = "bad string"

        self.checker.check_configurations_project_settings(configs=self.configs)

        self.assertEqual(self.configs['input_files'], "reference")

    def test_input_files_int(self):
        self.configs['input_files'] = 100

        self.checker.check_configurations_project_settings(configs=self.configs)

        self.assertEqual(self.configs['input_files'], "reference")

    def test_input_files_reference(self):
        self.configs['input_files'] = "reference"

        self.checker.check_configurations_project_settings(configs=self.configs)

        self.assertEqual(self.configs['input_files'], "reference")

    def test_input_files_individual(self):
        self.configs['input_files'] = "individual"

        self.checker.check_configurations_project_settings(configs=self.configs)

        self.assertEqual(self.configs['input_files'], "individual")


# TODO: Test check_configurations_files()
class TestCheckConfigurationsFiles(unittest.TestCase):

    def setUp(self):
        self.checker = ParameterChecker()

        self.configs = {'find_features': False, 
                        'train_model': False, 
                        'apply_model': False, 
                        'input_files': 'reference', 
                        'output_dir': "cwd", 
                        'reference_quant_file': "", 
                        'reference_meta_file': "", 
                        'feature_quant_file': "", 
                        'feature_meta_file': "" , 
                        'feature_file': "", 
                        'train_quant_file': "", 
                        'train_meta_file': "", 
                        'validate_quant_file': "", 
                        'validate_meta_file': "", 
                        'model_file': "", 
                        'experimental_quant_file': "" }
  
    def test_input_files_individual_feature_train_paths_identical(self):
        self.configs['input_files'] = "individual"
        self.configs['train_model'] = True
        self.configs['feature_quant_file'] = "pretend/this/is/a/path"
        self.configs['feature_meta_file'] = "pretend/this/is/a/path"
        self.configs['train_quant_file'] = "pretend/this/is/a/path"
        self.configs['train_meta_file'] = "pretend/this/is/a/path"

        with self.assertRaises(SystemExit) as e:
            self.checker.check_configurations_files(self.configs)

        self.assertEqual(e.exception.code, 1)

    def test_input_files_individual_feature_validate_paths_identical(self):
        self.configs['input_files'] = "individual"
        self.configs['train_model'] = True
        self.configs['feature_quant_file'] = "pretend/this/is/a/path"
        self.configs['feature_meta_file'] = "pretend/this/is/a/path"
        self.configs['validate_quant_file'] = "pretend/this/is/a/path"
        self.configs['validate_meta_file'] = "pretend/this/is/a/path"

        with self.assertRaises(SystemExit) as e:
            self.checker.check_configurations_files(self.configs)

        self.assertEqual(e.exception.code, 1)

    def test_input_files_individual_train_validate_paths_identical(self):
        self.configs['input_files'] = "individual"
        self.configs['train_model'] = True
        self.configs['train_quant_file'] = "pretend/this/is/a/path"
        self.configs['train_meta_file'] = "pretend/this/is/a/path"
        self.configs['validate_quant_file'] = "pretend/this/is/a/path"
        self.configs['validate_meta_file'] = "pretend/this/is/a/path"

        with self.assertRaises(SystemExit) as e:
            self.checker.check_configurations_files(self.configs)

        self.assertEqual(e.exception.code, 1)

    def test_input_files_individual_all_paths_identical(self):
        self.configs['input_files'] = "individual"
        self.configs['train_model'] = True
        self.configs['feature_quant_file'] = "pretend/this/is/a/path"
        self.configs['feature_meta_file'] = "pretend/this/is/a/path"
        self.configs['train_quant_file'] = "pretend/this/is/a/path"
        self.configs['train_meta_file'] = "pretend/this/is/a/path"
        self.configs['validate_quant_file'] = "pretend/this/is/a/path"
        self.configs['validate_meta_file'] = "pretend/this/is/a/path"

        with self.assertRaises(SystemExit) as e:
            self.checker.check_configurations_files(self.configs)

        self.assertEqual(e.exception.code, 1)

    def test_find_features_input_files_reference(self):
        self.configs['find_features'] = True

        with TemporaryDirectory() as tmpdir:
            reference_quant_path = os.path.join(tmpdir, "reference_quant.tsv")
            reference_meta_path = os.path.join(tmpdir, "reference_meta.tsv")

            with open(reference_quant_path, "w") as quant:
                quant.write("sample_id\tProtein1\tProtein2\tProtein3\n")
                quant.write("samp1\t1\t0\t1000\n")
            
            with open(reference_meta_path, "w") as meta:
                meta.write("sample_id\tclassification_label\n")
                meta.write("samp1\tH\n")

            self.configs['reference_quant_file'] = reference_quant_path
            self.configs['reference_meta_file'] = reference_meta_path

            self.checker.check_configurations_files(self.configs)

            self.assertIn('split_for_FS', self.configs)
            self.assertIn('split_for_train', self.configs)
            self.assertIn('split_for_validate', self.configs)

            self.assertEqual(self.configs['split_for_FS'], True)
            self.assertEqual(self.configs['split_for_train'], False)
            self.assertEqual(self.configs['split_for_validate'], False)

    def test_find_features_input_files_individual_bad_quant_path(self):
        self.configs['find_features'] = True
        self.configs['input_files'] = "individual"

        with TemporaryDirectory() as tmpdir:
            feature_quant_path = os.path.join(tmpdir, "reference_quant.tsv")
            feature_meta_path = os.path.join(tmpdir, "reference_meta.tsv")

            with open(feature_quant_path, "w") as quant:
                quant.write("sample_id\tProtein1\tProtein2\tProtein3\n")
                quant.write("samp1\t1\t0\t1000\n")
            
            with open(feature_meta_path, "w") as meta:
                meta.write("sample_id\tclassification_label\n")
                meta.write("samp1\tH\n")

            self.configs['feature_meta_file'] = feature_meta_path

            with self.assertRaises(SystemExit) as e:
                self.checker.check_configurations_files(self.configs)

            self.assertEqual(e.exception.code, 1)

    def test_find_features_input_files_individual_bad_meta_path(self):
        self.configs['find_features'] = True
        self.configs['input_files'] = "individual"

        with TemporaryDirectory() as tmpdir:
            feature_quant_path = os.path.join(tmpdir, "reference_quant.tsv")
            feature_meta_path = os.path.join(tmpdir, "reference_meta.tsv")

            with open(feature_quant_path, "w") as quant:
                quant.write("sample_id\tProtein1\tProtein2\tProtein3\n")
                quant.write("samp1\t1\t0\t1000\n")
            
            with open(feature_meta_path, "w") as meta:
                meta.write("sample_id\tclassification_label\n")
                meta.write("samp1\tH\n")

            self.configs['feature_quant_file'] = feature_quant_path

            with self.assertRaises(SystemExit) as e:
                self.checker.check_configurations_files(self.configs)

            self.assertEqual(e.exception.code, 1)

    def test_find_features_input_files_individual_bad_quant_file_extension(self):
        self.configs['find_features'] = True
        self.configs['input_files'] = "individual"

        with TemporaryDirectory() as tmpdir:
            feature_quant_path = os.path.join(tmpdir, "reference_quant")
            feature_meta_path = os.path.join(tmpdir, "reference_meta.tsv")

            with open(feature_quant_path, "w") as quant:
                quant.write("sample_id\tProtein1\tProtein2\tProtein3\n")
                quant.write("samp1\t1\t0\t1000\n")
            
            with open(feature_meta_path, "w") as meta:
                meta.write("sample_id\tclassification_label\n")
                meta.write("samp1\tH\n")

            self.configs['feature_quant_file'] = feature_quant_path
            self.configs['feature_meta_file'] = feature_meta_path

            with self.assertRaises(SystemExit) as e:
                self.checker.check_configurations_files(self.configs)

            self.assertEqual(e.exception.code, 1)

    def test_find_features_input_files_individual_bad_meta_file_extension(self):
        self.configs['find_features'] = True
        self.configs['input_files'] = "individual"

        with TemporaryDirectory() as tmpdir:
            feature_quant_path = os.path.join(tmpdir, "reference_quant.tsv")
            feature_meta_path = os.path.join(tmpdir, "reference_meta")

            with open(feature_quant_path, "w") as quant:
                quant.write("sample_id\tProtein1\tProtein2\tProtein3\n")
                quant.write("samp1\t1\t0\t1000\n")
            
            with open(feature_meta_path, "w") as meta:
                meta.write("sample_id\tclassification_label\n")
                meta.write("samp1\tH\n")

            self.configs['feature_quant_file'] = feature_quant_path
            self.configs['feature_meta_file'] = feature_meta_path

            with self.assertRaises(SystemExit) as e:
                self.checker.check_configurations_files(self.configs)

            self.assertEqual(e.exception.code, 1)

    def test_find_features_input_files_individual_bad_quant_file_contents(self):
        self.configs['find_features'] = True
        self.configs['input_files'] = "individual"

        with TemporaryDirectory() as tmpdir:
            feature_quant_path = os.path.join(tmpdir, "reference_quant.tsv")
            feature_meta_path = os.path.join(tmpdir, "reference_meta.tsv")

            with open(feature_quant_path, "w") as quant:
                quant.write("sample_id\tProtein1\tProtein2\tProtein3\n")
                quant.write("samp1\t1\t0\t1000\n")
                quant.write("samp2\t1\t0\t1000\t40\n")
            
            with open(feature_meta_path, "w") as meta:
                meta.write("sample_id\tclassification_label\n")
                meta.write("samp1\tH\n")

            self.configs['feature_quant_file'] = feature_quant_path
            self.configs['feature_meta_file'] = feature_meta_path

            with self.assertRaises(SystemExit) as e:
                self.checker.check_configurations_files(self.configs)

            self.assertEqual(e.exception.code, 1)

    def test_find_features_input_files_individual_bad_meta_file_contents(self):
        self.configs['find_features'] = True
        self.configs['input_files'] = "individual"

        with TemporaryDirectory() as tmpdir:
            feature_quant_path = os.path.join(tmpdir, "reference_quant.tsv")
            feature_meta_path = os.path.join(tmpdir, "reference_meta.tsv")

            with open(feature_quant_path, "w") as quant:
                quant.write("sample_id\tProtein1\tProtein2\tProtein3\n")
                quant.write("samp1\t1\t0\t1000\n")
            
            with open(feature_meta_path, "w") as meta:
                meta.write("sample_id\tclassification_label\n")
                meta.write("samp1\tH\n")
                meta.write("samp2\tH\tH\n")

            self.configs['feature_quant_file'] = feature_quant_path
            self.configs['feature_meta_file'] = feature_meta_path

            with self.assertRaises(SystemExit) as e:
                self.checker.check_configurations_files(self.configs)

            self.assertEqual(e.exception.code, 1)

    def test_find_features_false_bad_feature_file_path(self):
        self.configs['input_files'] = "individual"
        self.configs['train_model'] = True

        with self.assertRaises(SystemExit) as e:
            self.checker.check_configurations_files(self.configs)

        self.assertEqual(e.exception.code, 1)

    def test_find_features_false_bad_feature_file_extension(self):
        self.configs['input_files'] = "individual"
        self.configs['train_model'] = True

        with TemporaryDirectory() as tmpdir:
            feature_file_path = os.path.join(tmpdir, "feature_file")

            with open(feature_file_path, "w") as feat:
                feat.write("Protein1\tProtein2\n")
                feat.write("XXXX\tYYYY\n")

            self.configs['feature_file'] = feature_file_path

            with self.assertRaises(SystemExit) as e:
                self.checker.check_configurations_files(self.configs)

            self.assertEqual(e.exception.code, 1)

    def test_find_features_false_bad_feature_file_contents(self):
        self.configs['input_files'] = "individual"
        self.configs['train_model'] = True

        with TemporaryDirectory() as tmpdir:
            feature_file_path = os.path.join(tmpdir, "feature_file.tsv")

            with open(feature_file_path, "w") as feat:
                feat.write("Protein1\tProtein2\n")
                feat.write("XXXX\tYYYY\n")
                feat.write("AAAA\tBBBB\tCCCC\n")

            self.configs['feature_file'] = feature_file_path

            with self.assertRaises(SystemExit) as e:
                self.checker.check_configurations_files(self.configs)

            self.assertEqual(e.exception.code, 1)

    ## TODO: train_model tests
    def test_train_model_input_files_reference(self):
        self.configs['train_model'] = True

        with TemporaryDirectory() as tmpdir:
            reference_quant_path = os.path.join(tmpdir, "reference_quant.tsv")
            reference_meta_path = os.path.join(tmpdir, "reference_meta.tsv")
            feature_file_path = os.path.join(tmpdir, "feature_file.tsv")
            
            with open(reference_quant_path, "w") as quant:
                quant.write("sample_id\tProtein1\tProtein2\tProtein3\n")
                quant.write("samp1\t1\t0\t1000\n")
            
            with open(reference_meta_path, "w") as meta:
                meta.write("sample_id\tclassification_label\n")
                meta.write("samp1\tH\n")

            with open(feature_file_path, "w") as feat:
                feat.write("Protein1\tProtein2\n")
                feat.write("XXXX\tYYYY\n")

            self.configs['reference_quant_file'] = reference_quant_path
            self.configs['reference_meta_file'] = reference_meta_path
            self.configs['feature_file'] = feature_file_path

            self.checker.check_configurations_files(self.configs)

            self.assertIn('split_for_FS', self.configs)
            self.assertIn('split_for_train', self.configs)
            self.assertIn('split_for_validate', self.configs)

            self.assertEqual(self.configs['split_for_FS'], False)
            self.assertEqual(self.configs['split_for_train'], True)
            self.assertEqual(self.configs['split_for_validate'], True)

    ## TODO: apply_model tests
        

class CheckConfigurationsFeatureSelection(unittest.TestCase):

    def setUp(self):
        self.checker = ParameterChecker()

        self.configs = {'find_features': True, 
                        'k_rules': 15, 
                        'missingness_cutoff': 0.5, 
                        'disjoint': False, 
                        'mutual_information': True, 
                        'mutual_information_cutoff': 0.7 }
        
    def test_find_features_false(self):
        self.configs = {'find_features': False, 
                        'k_rules': "15", 
                        'missingness_cutoff': True, 
                        'disjoint': 1.0, 
                        'mutual_information': 0.0, 
                        'mutual_information_cutoff': False }
        
        self.checker.check_configurations_feature_selection(self.configs)

        self.assertDictEqual(self.configs, {'find_features': False, 
                                            'k_rules': "15", 
                                            'missingness_cutoff': True, 
                                            'disjoint': 1.0, 
                                            'mutual_information': 0.0, 
                                            'mutual_information_cutoff': False })
        
    def test_k_rules_string(self):
        self.configs['k_rules'] = "15"

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['k_rules'], 15)

    def test_k_rules_bool(self):
        self.configs['k_rules'] = True

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['k_rules'], 15)

    def test_k_rules_float(self):
        self.configs['k_rules'] = 50.0

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['k_rules'], 15)

    def test_k_rules_negative(self):
        self.configs['k_rules'] = -15

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['k_rules'], 15)

    def test_k_rules_greater_than_max(self):
        self.configs['k_rules'] = 100

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['k_rules'], 15)

    def test_k_rules_0(self):
        self.configs['k_rules'] = 0

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['k_rules'], 15)

    def test_k_rules_1(self):
        self.configs['k_rules'] = 1

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['k_rules'], 1)

    def test_k_rules_50(self):
        self.configs['k_rules'] = 50

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['k_rules'], 50)

    def test_k_rules_15(self):
        self.configs['k_rules'] = 15

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['k_rules'], 15)

    def test_missingness_cutoff_string(self):
        self.configs['missingness_cutoff'] = "0.5"

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['missingness_cutoff'], 0.5)

    def test_missingness_cutoff_bool(self):
        self.configs['missingness_cutoff'] = True

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['missingness_cutoff'], 0.5)

    def test_missingness_cutoff_int(self):
        self.configs['missingness_cutoff'] = 1

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['missingness_cutoff'], 1)

    def test_missingness_cutoff_float(self):
        self.configs['missingness_cutoff'] = 0.8

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['missingness_cutoff'], 0.8)

    def test_missingness_cutoff_negative(self):
        self.configs['missingness_cutoff'] = -0.5

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['missingness_cutoff'], 0.5)

    def test_missingness_cutoff_greater_than_max(self):
        self.configs['missingness_cutoff'] = 15

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['missingness_cutoff'], 0.5)

    def test_missingness_cutoff_0(self):
        self.configs['missingness_cutoff'] = 0.0

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['missingness_cutoff'], 0)

    def test_missingness_cutoff_1(self):
        self.configs['missingness_cutoff'] = 1.0

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['missingness_cutoff'], 1)

    def test_missingness_cutoff_03(self):
        self.configs['missingness_cutoff'] = 0.3

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['missingness_cutoff'], 0.3)

    def test_missingness_cutoff_05(self):
        self.configs['missingness_cutoff'] = 0.5

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['missingness_cutoff'], 0.5)

    def test_disjoint_string(self):
        self.configs['disjoint'] = "True"

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['disjoint'], False)

    def test_disjoint_float(self):
        self.configs['disjoint'] = 0.0

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['disjoint'], False)

    def test_disjoint_int(self):
        self.configs['disjoint'] = 1

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['disjoint'], False)

    def test_disjoint_true(self):
        self.configs['disjoint'] = True

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['disjoint'], True)

    def test_disjoint_false(self):
        self.configs['disjoint'] = False

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['disjoint'], False)

    def test_mutual_information_string(self):
        self.configs['mutual_information'] = "False"

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['mutual_information'], True)

    def test_mutual_information_float(self):
        self.configs['mutual_information'] = 0.0

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['mutual_information'], True)

    def test_mutual_information_int(self):
        self.configs['mutual_information'] = 1

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['mutual_information'], True)

    def test_mutual_information_true(self):
        self.configs['mutual_information'] = True

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['mutual_information'], True)

    def test_mutual_information_false(self):
        self.configs['mutual_information'] = False

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['mutual_information'], False)

    def test_mutual_information_cutoff_string(self):
        self.configs['mutual_information_cutoff'] = "0.5"

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['mutual_information_cutoff'], 0.7)

    def test_mutual_information_cutoff_bool(self):
        self.configs['mutual_information_cutoff'] = True

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['mutual_information_cutoff'], 0.7)

    def test_mutual_information_cutoff_int(self):
        self.configs['mutual_information_cutoff'] = 1

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['mutual_information_cutoff'], 1)

    def test_mutual_information_cutoff_float(self):
        self.configs['mutual_information_cutoff'] = 0.8

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['mutual_information_cutoff'], 0.8)

    def test_mutual_information_cutoff_negative(self):
        self.configs['mutual_information_cutoff'] = -0.5

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['mutual_information_cutoff'], 0.7)

    def test_mutual_information_cutoff_greater_than_max(self):
        self.configs['mutual_information_cutoff'] = 15

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['mutual_information_cutoff'], 0.7)

    def test_mutual_information_cutoff_0(self):
        self.configs['mutual_information_cutoff'] = 0.0

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['mutual_information_cutoff'], 0)

    def test_mutual_information_cutoff_1(self):
        self.configs['mutual_information_cutoff'] = 1.0

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['mutual_information_cutoff'], 1)

    def test_mutual_information_cutoff_05(self):
        self.configs['mutual_information_cutoff'] = 0.5

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['mutual_information_cutoff'], 0.5)

    def test_mutual_information_cutoff_07(self):
        self.configs['mutual_information_cutoff'] = 0.7

        self.checker.check_configurations_feature_selection(self.configs)

        self.assertEqual(self.configs['mutual_information_cutoff'], 0.7)


class TestCheckConfigurationsModelTraining(unittest.TestCase):

    def setUp(self):
        self.checker = ParameterChecker()

        self.configs = {'train_model': True,
                        'impute_NA_missing': True, 
                        'cross_val': 5, 
                        'model_type': "RF", 
                        'autotune_hyperparameters': "", 
                        'autotune_n_iter': 20, 
                        'verbose': 0}
        
    def test_impute_NA_missing_True(self):
        self.configs['impute_NA_missing'] = True

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['impute_NA_missing'], True)

    def test_impute_NA_missing_False(self):
        self.configs['impute_NA_missing'] = False

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['impute_NA_missing'], False)

    def test_impute_NA_missing_float(self):
        self.configs['impute_NA_missing'] = 0.989

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['impute_NA_missing'], True)

    def test_impute_NA_missing_int(self):
        self.configs['impute_NA_missing'] = 34

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['impute_NA_missing'], True)

    def test_impute_NA_missing_string(self):
        self.configs['impute_NA_missing'] = 'False'

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['impute_NA_missing'], True)

    def test_cross_val_5(self):
        self.configs['cross_val'] = 5

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['cross_val'], 5)

    def test_cross_val_10(self):
        self.configs['cross_val'] = 10

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['cross_val'], 10)

    def test_cross_val_greater_than_max(self):
        self.configs['cross_val'] = 1000

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['cross_val'], 5)

    def test_cross_val_negative(self):
        self.configs['cross_val'] = -20

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['cross_val'], 5)

    def test_cross_val_0(self):
        self.configs['cross_val'] = 0

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['cross_val'], 5)

    def test_cross_val_float(self):
        self.configs['cross_val'] = 6.0

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['cross_val'], 5)

    def test_cross_val_bool(self):
        self.configs['cross_val'] = False

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['cross_val'], 5)

    def test_cross_val_string(self):
        self.configs['cross_val'] = '5'

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['cross_val'], 5)

    def test_model_type_RF(self):
        self.configs['model_type'] = 'RF'

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['model_type'], 'RF')

    def test_model_type_SVM(self):
        self.configs['model_type'] = 'SVM'

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['model_type'], 'SVM')

    def test_model_type_bad_string(self):
        self.configs['model_type'] = 'SMV'

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['model_type'], 'RF')

    def test_model_type_float(self):
        self.configs['model_type'] = 1.2

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['model_type'], 'RF')

    def test_model_type_int(self):
        self.configs['model_type'] = 1000

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['model_type'], 'RF')

    def test_model_type_bool(self):
        self.configs['model_type'] = True

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['model_type'], 'RF')

    def test_autotune_hyperparameters_None(self):
        self.configs['autotune_hyperparameters'] = ""

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['autotune_hyperparameters'], None)

    def test_autotune_hyperparameters_grid(self):
        self.configs['autotune_hyperparameters'] = "grid"

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['autotune_hyperparameters'], "grid")

    def test_autotune_hyperparameters_random(self):
        self.configs['autotune_hyperparameters'] = "random"

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['autotune_hyperparameters'], "random")

    def test_autotune_hyperparameters_int(self):
        self.configs['autotune_hyperparameters'] = 88

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['autotune_hyperparameters'], None)

    def test_autotune_hyperparameters_float(self):
        self.configs['autotune_hyperparameters'] = 42.0

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['autotune_hyperparameters'], None)

    def test_autotune_hyperparameters_bool(self):
        self.configs['autotune_hyperparameters'] = False

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['autotune_hyperparameters'], None)

    def test_autotune_hyperparameters_bad_string(self):
        self.configs['autotune_hyperparameters'] = "not correct"

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['autotune_hyperparameters'], None)

    def test_autotune_n_iter_negative(self):
        self.configs['autotune_n_iter'] = -25

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['autotune_n_iter'], 20)

    def test_autotune_n_iter_greater_than_max(self):
        self.configs['autotune_n_iter'] = 200

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['autotune_n_iter'], 20)

    def test_autotune_n_iter_0(self):
        self.configs['autotune_n_iter'] = 0

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['autotune_n_iter'], 20)

    def test_autotune_n_iter_10(self):
        self.configs['autotune_n_iter'] = 10

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['autotune_n_iter'], 10)

    def test_autotune_n_iter_20(self):
        self.configs['autotune_n_iter'] = 20

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['autotune_n_iter'], 20)

    def test_autotune_n_iter_80(self):
        self.configs['autotune_n_iter'] = 80

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['autotune_n_iter'], 80)

    def test_autotune_n_iter_float(self):
        self.configs['autotune_n_iter'] = 150.0

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['autotune_n_iter'], 20)

    def test_autotune_n_iter_string(self):
        self.configs['autotune_n_iter'] = "20"

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['autotune_n_iter'], 20)

    def test_autotune_n_iter_bool(self):
        self.configs['autotune_n_iter'] = True

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['autotune_n_iter'], 20)

    def test_verbose_bool(self):
        self.configs['verbose'] = True

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['verbose'], 0)

    def test_verbose_string(self):
        self.configs['verbose'] = "True"

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['verbose'], False)

    def test_verbose_0(self):
        self.configs['verbose'] = 0

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['verbose'], 0)

    def test_verbose_1(self):
        self.configs['verbose'] = 1

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['verbose'], 1)

    def test_verbose_2(self):
        self.configs['verbose'] = 2

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['verbose'], 2)

    def test_verbose_3(self):
        self.configs['verbose'] = 3

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['verbose'], 3)

    def test_verbose_4(self):
        self.configs['verbose'] = 4

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['verbose'], 4)

    def test_verbose_negative(self):
        self.configs['verbose'] = -3

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['verbose'], 0)

    def test_verbose_greater_than_max(self):
        self.configs['verbose'] = 77

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['verbose'], 0)

    def test_verbose_float(self):
        self.configs['verbose'] = 1.0

        self.checker.check_configurations_model_training(self.configs)

        self.assertEqual(self.configs['verbose'], 0)
        

class TestCheckConfigurationsExperimentalClassification(unittest.TestCase):

    def setUp(self):
        self.checker = ParameterChecker()

        self.configs = {'apply_model': True, 
                        'prediction_format': 'classes'}

    def test_prediction_format_classes(self):
        self.configs['prediction_format'] = 'classes'

        self.checker.check_configurations_experimental_classification(self.configs)
        
        self.assertEqual(self.configs['prediction_format'], 'classes')

    def test_prediction_format_probabilities(self):
        self.configs['prediction_format'] = 'probabilities'

        self.checker.check_configurations_experimental_classification(self.configs)
        
        self.assertEqual(self.configs['prediction_format'], 'probabilities')

    def test_prediction_format_float(self):
        self.configs['prediction_format'] = 0.009

        self.checker.check_configurations_experimental_classification(self.configs)
        
        self.assertEqual(self.configs['prediction_format'], 'classes')

    def test_prediction_format_int(self):
        self.configs['prediction_format'] = 57

        self.checker.check_configurations_experimental_classification(self.configs)
        
        self.assertEqual(self.configs['prediction_format'], 'classes')

    def test_prediction_format_bool(self):
        self.configs['prediction_format'] = False

        self.checker.check_configurations_experimental_classification(self.configs)
        
        self.assertEqual(self.configs['prediction_format'], 'classes')

    def test_prediction_format_bad_string(self):
        self.configs['prediction_format'] = 'this is not right'

        self.checker.check_configurations_experimental_classification(self.configs)
        
        self.assertEqual(self.configs['prediction_format'], 'classes')
        

# TODO: Test run_parameter_checker()
class TestRunParameterChecker(unittest.TestCase):

    def setUp(self):
        self.checker = ParameterChecker()

        self.configs = {}


if __name__ == "__main__":
    unittest.main()
