import sys
import os
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import cloudpickle
import sklearn

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from ModelGenerator import ModelGenerator


class TestOptimizeModelRF(unittest.TestCase):

    def setUp(self):
        self.model_gen = ModelGenerator()
        # Simple training data
        self.X = pd.DataFrame({
            'f1': [0.1, 0.2, 0.3, 0.4],
            'f2': [1.0, 0.9, 0.8, 0.7]
        })
        self.y = [0, 1, 0, 1]

        self.base_configs = {
            'train_meta_table': pd.DataFrame({'classification_label': self.y}),
            'autotune_n_iter': 3,
            'cross_val': 2,
            'seed': 42,
            'autotune_hyperparameters': None,
            'verbose': 0
        }

    def test_rf_grid_search_with_seed(self):
        configs = self.base_configs.copy()
        configs['autotune_hyperparameters'] = "grid"

        with patch('ModelGenerator.GridSearchCV') as mock_gs, \
             patch('ModelGenerator.RandomForestClassifier') as mock_rf:
            rf_instance = MagicMock()
            mock_rf.return_value = rf_instance

            gs_instance = MagicMock()
            gs_instance.best_estimator_ = MagicMock()
            gs_instance.cv_results_ = {
                'mean_test_Accuracy': np.array([0.8, 0.9]),
                'std_test_Accuracy': np.array([0.05, 0.02]),
                'mean_test_Precision': np.array([0.7, 0.85]),
                'std_test_Precision': np.array([0.03, 0.01]),
                'mean_test_Recall': np.array([0.75, 0.88]),
                'std_test_Recall': np.array([0.04, 0.02]),
            }
            gs_instance.best_index_ = 1
            mock_gs.return_value = gs_instance

            best_est, cv_results, best_index = self.model_gen.optimize_model_rf(configs, self.X)

            # check RF called with seed and verbose
            mock_rf.assert_called_once()
            kwargs = mock_rf.call_args.kwargs
            self.assertEqual(kwargs.get('random_state'), 42)
            self.assertEqual(kwargs.get('verbose'), configs['verbose'])

            # GridSearchCV called with RF and cv
            mock_gs.assert_called_once()
            gs_kwargs = mock_gs.call_args.kwargs
            self.assertIs(gs_kwargs['estimator'], rf_instance)
            self.assertEqual(gs_kwargs['cv'], configs['cross_val'])
            self.assertIn('scoring', gs_kwargs)
            self.assertEqual(gs_kwargs['refit'], 'Accuracy')

            # fit called with X, y
            gs_instance.fit.assert_called_once()
            np.testing.assert_array_equal(gs_instance.fit.call_args.args[0], self.X.values)
            np.testing.assert_array_equal(gs_instance.fit.call_args.args[1], np.array(self.y))

            self.assertIs(best_est, gs_instance.best_estimator_)
            self.assertIs(cv_results, gs_instance.cv_results_)
            self.assertEqual(best_index, gs_instance.best_index_)

    def test_rf_random_search_without_seed(self):
        configs = self.base_configs.copy()
        configs['autotune_hyperparameters'] = "random"
        configs['seed'] = None

        with patch('ModelGenerator.RandomizedSearchCV') as mock_rs, \
             patch('ModelGenerator.RandomForestClassifier') as mock_rf:
            rf_instance = MagicMock()
            mock_rf.return_value = rf_instance

            rs_instance = MagicMock()
            rs_instance.best_estimator_ = MagicMock()
            rs_instance.cv_results_ = {'mean_test_Accuracy': np.array([0.5])}
            rs_instance.best_index_ = 0
            mock_rs.return_value = rs_instance

            best_est, cv_results, best_index = self.model_gen.optimize_model_rf(configs, self.X)

            # RF called without random_state
            kwargs = mock_rf.call_args.kwargs
            self.assertNotIn('random_state', kwargs)

            # RandomizedSearchCV called with correct arguments
            mock_rs.assert_called_once()
            rs_kwargs = mock_rs.call_args.kwargs
            self.assertIs(rs_kwargs['estimator'], rf_instance)
            self.assertEqual(rs_kwargs['n_iter'], configs['autotune_n_iter'])
            self.assertEqual(rs_kwargs['cv'], configs['cross_val'])
            self.assertIn('scoring', rs_kwargs)
            self.assertEqual(rs_kwargs['refit'], 'Accuracy')
            self.assertNotIn('random_state', rs_kwargs)

            rs_instance.fit.assert_called_once()
            self.assertIs(best_est, rs_instance.best_estimator_)
            self.assertIs(cv_results, rs_instance.cv_results_)
            self.assertEqual(best_index, rs_instance.best_index_)


class TestOptimizeModelSVM(unittest.TestCase):

    def setUp(self):
        self.model_gen = ModelGenerator()
        self.X = pd.DataFrame({
            'f1': [0.1, 0.2, 0.3, 0.4],
            'f2': [1.0, 0.9, 0.8, 0.7]
        })
        self.y = [0, 1, 0, 1]

        self.base_configs = {
            'train_meta_table': pd.DataFrame({'classification_label': self.y}),
            'autotune_n_iter': 2,
            'cross_val': 2,
            'seed': 123,
            'autotune_hyperparameters': None,
            'verbose': 0
        }

    def test_svm_grid_search_with_seed(self):
        configs = self.base_configs.copy()
        configs['autotune_hyperparameters'] = "grid"

        with patch('ModelGenerator.GridSearchCV') as mock_gs, \
             patch('ModelGenerator.SVC') as mock_svc:
            svm_instance = MagicMock()
            mock_svc.return_value = svm_instance

            gs_instance = MagicMock()
            gs_instance.best_estimator_ = MagicMock()
            gs_instance.cv_results_ = {'mean_test_Accuracy': np.array([0.7])}
            gs_instance.best_index_ = 0
            mock_gs.return_value = gs_instance

            best_est, cv_results, best_index = self.model_gen.optimize_model_svm(configs, self.X)

            mock_svc.assert_called_once()
            kwargs = mock_svc.call_args.kwargs
            self.assertEqual(kwargs.get('random_state'), 123)
            self.assertEqual(kwargs.get('verbose'), configs['verbose'])
            self.assertTrue(kwargs.get('probability'))

            mock_gs.assert_called_once()
            self.assertIs(best_est, gs_instance.best_estimator_)
            self.assertIs(cv_results, gs_instance.cv_results_)
            self.assertEqual(best_index, gs_instance.best_index_)

    def test_svm_random_search_without_seed(self):
        configs = self.base_configs.copy()
        configs['autotune_hyperparameters'] = "random"
        configs['seed'] = None

        with patch('ModelGenerator.RandomizedSearchCV') as mock_rs, \
             patch('ModelGenerator.SVC') as mock_svc:
            svm_instance = MagicMock()
            mock_svc.return_value = svm_instance

            rs_instance = MagicMock()
            rs_instance.best_estimator_ = MagicMock()
            rs_instance.cv_results_ = {'mean_test_Accuracy': np.array([0.6])}
            rs_instance.best_index_ = 0
            mock_rs.return_value = rs_instance

            best_est, cv_results, best_index = self.model_gen.optimize_model_svm(configs, self.X)

            mock_svc.assert_called_once()
            kwargs = mock_svc.call_args.kwargs
            self.assertNotIn('random_state', kwargs)
            self.assertTrue(kwargs.get('probability'))

            mock_rs.assert_called_once()
            rs_kwargs = mock_rs.call_args.kwargs
            self.assertIs(rs_kwargs['estimator'], svm_instance)
            self.assertEqual(rs_kwargs['n_iter'], configs['autotune_n_iter'])
            self.assertEqual(rs_kwargs['cv'], configs['cross_val'])
            self.assertNotIn('random_state', rs_kwargs)

            rs_instance.fit.assert_called_once()
            self.assertIs(best_est, rs_instance.best_estimator_)
            self.assertIs(cv_results, rs_instance.cv_results_)
            self.assertEqual(best_index, rs_instance.best_index_)


class TestTrainModel(unittest.TestCase):

    def setUp(self):
        self.model_gen = ModelGenerator()
        self.X = pd.DataFrame({
            'f1': [0.1, 0.2, 0.3, 0.4],
            'f2': [1.0, 0.9, 0.8, 0.7]
        })
        self.y = [0, 1, 0, 1]
        self.base_configs = {
            'train_meta_table': pd.DataFrame({'classification_label': self.y}),
            'cross_val': 2,
            'seed': 7,
            'autotune_n_iter': 2,
            'autotune_hyperparameters': None,
            'model_type': 'RF',
            'verbose': 0
        }

    @patch('ModelGenerator.cross_validate')
    @patch('ModelGenerator.RandomForestClassifier')
    def test_train_rf_no_autotune(self, mock_rf_cls, mock_cv):
        configs = self.base_configs.copy()
        configs['model_type'] = "RF"
        configs['autotune_hyperparameters'] = None

        # mock cross_validate results
        mock_cv.return_value = {
            'test_Accuracy': np.array([0.8, 0.6]),
            'test_Precision': np.array([0.75, 0.65]),
            'test_Recall': np.array([0.7, 0.55])
        }

        rf_instance = MagicMock()
        rf_instance.get_params.return_value = {'n_estimators': 100}
        mock_rf_cls.return_value = rf_instance

        model, info = self.model_gen.train_model(configs, self.X)

        mock_rf_cls.assert_called_once()
        rf_kwargs = mock_rf_cls.call_args.kwargs
        self.assertEqual(rf_kwargs.get('random_state'), 7)
        self.assertEqual(rf_kwargs.get('verbose'), configs['verbose'])

        mock_cv.assert_called_once()
        rf_instance.fit.assert_called_once()

        self.assertIs(model, rf_instance)
        self.assertIn('cv_scores', info)
        self.assertIn('params', info)
        self.assertAlmostEqual(info['cv_scores']['Accuracy_Mean'], 0.7)
        self.assertAlmostEqual(info['cv_scores']['Precision_Mean'], 0.7)
        self.assertAlmostEqual(info['cv_scores']['Recall_Mean'], 0.625)
        self.assertEqual(info['params']['n_estimators'], 100)

    @patch.object(ModelGenerator, 'optimize_model_rf')
    def test_train_rf_with_autotune(self, mock_optimize_rf):
        configs = self.base_configs.copy()
        configs['model_type'] = "RF"
        configs['autotune_hyperparameters'] = "grid"

        rf_instance = MagicMock()
        cv_results = {
            'mean_test_Accuracy': np.array([0.5, 0.9]),
            'std_test_Accuracy': np.array([0.1, 0.02]),
            'mean_test_Precision': np.array([0.4, 0.85]),
            'std_test_Precision': np.array([0.12, 0.01]),
            'mean_test_Recall': np.array([0.45, 0.88]),
            'std_test_Recall': np.array([0.09, 0.03])
        }
        mock_optimize_rf.return_value = (rf_instance, cv_results, 1)
        rf_instance.get_params.return_value = {'n_estimators': 200}

        model, info = self.model_gen.train_model(configs, self.X)

        mock_optimize_rf.assert_called_once()
        self.assertIs(model, rf_instance)
        self.assertEqual(info['cv_scores']['Accuracy_Mean'], 0.9)
        self.assertEqual(info['cv_scores']['Precision_Mean'], 0.85)
        self.assertEqual(info['cv_scores']['Recall_Mean'], 0.88)
        self.assertEqual(info['params']['n_estimators'], 200)

    @patch('ModelGenerator.cross_validate')
    @patch('ModelGenerator.SVC')
    def test_train_svm_no_autotune(self, mock_svc_cls, mock_cv):
        configs = self.base_configs.copy()
        configs['model_type'] = "SVM"
        configs['autotune_hyperparameters'] = None

        mock_cv.return_value = {
            'test_Accuracy': np.array([0.5, 0.5]),
            'test_Precision': np.array([0.6, 0.6]),
            'test_Recall': np.array([0.7, 0.7])
        }

        svm_instance = MagicMock()
        svm_instance.get_params.return_value = {'C': 1.0}
        mock_svc_cls.return_value = svm_instance

        model, info = self.model_gen.train_model(configs, self.X)

        mock_svc_cls.assert_called_once()
        svc_kwargs = mock_svc_cls.call_args.kwargs
        self.assertEqual(svc_kwargs.get('random_state'), 7)
        self.assertTrue(svc_kwargs.get('probability'))

        mock_cv.assert_called_once()
        svm_instance.fit.assert_called_once()

        self.assertIs(model, svm_instance)
        self.assertEqual(info['cv_scores']['Accuracy_Mean'], 0.5)
        self.assertEqual(info['params']['C'], 1.0)

    @patch.object(ModelGenerator, 'optimize_model_svm')
    def test_train_svm_with_autotune(self, mock_optimize_svm):
        configs = self.base_configs.copy()
        configs['model_type'] = "SVM"
        configs['autotune_hyperparameters'] = "random"

        svm_instance = MagicMock()
        cv_results = {
            'mean_test_Accuracy': np.array([0.6]),
            'std_test_Accuracy': np.array([0.01]),
            'mean_test_Precision': np.array([0.65]),
            'std_test_Precision': np.array([0.02]),
            'mean_test_Recall': np.array([0.7]),
            'std_test_Recall': np.array([0.03])
        }
        mock_optimize_svm.return_value = (svm_instance, cv_results, 0)
        svm_instance.get_params.return_value = {'C': 10.0}

        model, info = self.model_gen.train_model(configs, self.X)

        mock_optimize_svm.assert_called_once()
        self.assertIs(model, svm_instance)
        self.assertEqual(info['cv_scores']['Accuracy_Mean'], 0.6)
        self.assertEqual(info['params']['C'], 10.0)

    @patch('ModelGenerator.cross_validate', side_effect=ValueError("bad data"))
    @patch('ModelGenerator.RandomForestClassifier')
    def test_train_rf_error_raises_system_exit(self, mock_rf_cls, mock_cv):
        configs = self.base_configs.copy()
        configs['model_type'] = "RF"
        configs['autotune_hyperparameters'] = None

        with self.assertRaises(SystemExit) as cm:
            self.model_gen.train_model(configs, self.X)
        self.assertEqual(cm.exception.code, 1)

    @patch('ModelGenerator.cross_validate', side_effect=ValueError("bad data"))
    @patch('ModelGenerator.SVC')
    def test_train_svm_error_raises_system_exit(self, mock_svc_cls, mock_cv):
        configs = self.base_configs.copy()
        configs['model_type'] = "SVM"
        configs['autotune_hyperparameters'] = None

        with self.assertRaises(SystemExit) as cm:
            self.model_gen.train_model(configs, self.X)
        self.assertEqual(cm.exception.code, 1)


class TestValidateModel(unittest.TestCase):

    def setUp(self):
        self.model_gen = ModelGenerator()
        # simple validation data
        self.X_val = pd.DataFrame({
            'f1': [0.1, 0.2, 0.3],
            'f2': [1.0, 0.9, 0.8]
        })
        self.y_val = [0, 1, 1]
        self.configs = {
            'validate_meta_table': pd.DataFrame({'classification_label': self.y_val})
        }

    def test_validate_model_scores(self):
        model = MagicMock()
        # predicted labels
        y_pred = [0, 0, 1]
        model.predict.return_value = y_pred

        scores = self.model_gen.validate_model(self.configs, model, self.X_val)

        # accuracy: 2/3
        self.assertAlmostEqual(scores['Accuracy'], 2/3)
        # we won't be super picky about precision/recall, just check they exist and between 0 and 1
        self.assertIn('Precision', scores)
        self.assertIn('Recall', scores)
        self.assertTrue(0.0 <= scores['Precision'] <= 1.0)
        self.assertTrue(0.0 <= scores['Recall'] <= 1.0)


class TestSaveModel(unittest.TestCase):

    def setUp(self):
        self.model_gen = ModelGenerator()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_save_model_creates_pickle_with_metadata(self):
        model = MagicMock()
        output_path = os.path.join(self.temp_dir, "trained_model.pkl")

        self.model_gen.save_model(model, output_path)

        self.assertTrue(os.path.exists(output_path))

        with open(output_path, "rb") as f:
            loaded = cloudpickle.load(f)

        self.assertIn('model', loaded)
        self.assertIn('sklearn_version', loaded)
        self.assertEqual(loaded['sklearn_version'], sklearn.__version__)
        # model should at least be a sklearn-like object (MagicMock here)
        self.assertTrue(hasattr(loaded['model'], 'predict'))


class TestSaveModelInformation(unittest.TestCase):

    def setUp(self):
        self.model_gen = ModelGenerator()
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.temp_dir, "model_information.txt")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_save_model_information_writes_expected_sections(self):
        metrics = {
            'params': {
                'n_estimators': 100,
                'max_depth': None
            },
            'cv_scores': {
                'Accuracy_Mean': 0.8,
                'Accuracy_Std': 0.05,
                'Precision_Mean': 0.75,
                'Precision_Std': 0.04,
                'Recall_Mean': 0.7,
                'Recall_Std': 0.06
            },
            'val_scores': {
                'Accuracy': 0.85,
                'Precision': 0.8,
                'Recall': 0.78
            }
        }

        self.model_gen.save_model_information(metrics, self.output_path)

        self.assertTrue(os.path.exists(self.output_path))

        with open(self.output_path, "r") as f:
            content = f.read()

        self.assertIn("---MODEL PARAMETERS---", content)
        self.assertIn("n_estimators: 100", content)
        self.assertIn("---TRAIN/TEST CV SCORES---", content)
        self.assertIn("Accuracy_Mean: 0.8", content)
        self.assertIn("---VALIDATION SCORES---", content)
        self.assertIn("Accuracy: 0.85", content)


class TestRunModelGenerator(unittest.TestCase):

    def setUp(self):
        self.model_gen = ModelGenerator()
        self.temp_dir = tempfile.mkdtemp()

        # tiny fake feature_table
        self.feature_table = pd.DataFrame({
            'Protein1': ['P1', 'P1'],
            'Protein2': ['P2', 'P3']
        })

        # tiny quant tables (indexes are sample IDs)
        self.train_quant = pd.DataFrame({
            'sample_id': ['s1', 's2', 's3', 's4'],
            'P1': [0.1, 0.2, 0.3, 0.4],
            'P2': [0.5, 0.6, 0.7, 0.8],
            'P3': [0.9, 1.0, 1.1, 1.2]
        }).set_index('sample_id')

        self.validate_quant = pd.DataFrame({
            'sample_id': ['v1', 'v2', 'v3', 'v4'],
            'P1': [0.2, 0.3, 0.4, 0.5],
            'P2': [0.6, 0.7, 0.8, 0.9],
            'P3': [1.0, 1.1, 1.2, 1.3]
        }).set_index('sample_id')

        self.train_meta = pd.DataFrame({
            'classification_label': [0, 1, 0, 1]
        }, index=self.train_quant.index)

        self.validate_meta = pd.DataFrame({
            'classification_label': [0, 0, 1, 1]
        }, index=self.validate_quant.index)

        self.base_configs = {
            'feature_table': self.feature_table.copy(),
            'train_quant_table': self.train_quant.copy(),
            'validate_quant_table': self.validate_quant.copy(),
            'train_meta_table': self.train_meta.copy(),
            'validate_meta_table': self.validate_meta.copy(),
            'impute_NA_missing': True,
            'model_type': 'RF',
            'autotune_hyperparameters': None,
            'cross_val': 2,
            'seed': 42,
            'autotune_n_iter': 2,
            'verbose': 0,
            'output_dir': self.temp_dir
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch.object(ModelGenerator, 'save_model')
    @patch.object(ModelGenerator, 'save_model_information')
    @patch.object(ModelGenerator, 'validate_model')
    @patch.object(ModelGenerator, 'train_model')
    @patch('ModelGenerator.DataTransformer')
    def test_run_model_generator_impute_true(self, mock_dt_cls, mock_train, mock_validate,
                                             mock_save_info, mock_save_model):
        configs = self.base_configs.copy()
        configs['impute_NA_missing'] = True

        # Mock DataTransformer instance and methods
        dt_instance = MagicMock()
        mock_dt_cls.return_value = dt_instance

        # add_missing_proteins: just return the same table for simplicity
        dt_instance.add_missing_proteins.side_effect = lambda feature_df, quant_df: quant_df

        # transform_df: return a dict mapping pair to bool array (length 4)
        bool_array = np.array([True, False, True, False])
        dt_instance.transform_df.return_value = {('P1', 'P2'): bool_array}

        # prep_vectorized_pairs_for_scikitlearn: return a simple 4x1 DataFrame
        dt_instance.prep_vectorized_pairs_for_scikitlearn.return_value = pd.DataFrame({
            ('P1', 'P2'): [1, 0, 1, 0]
        })

        dummy_model = MagicMock()
        model_info = {
            'cv_scores': {'Accuracy_Mean': 1.0},
            'params': {'dummy_param': 123}
        }
        mock_train.return_value = (dummy_model, model_info)
        mock_validate.return_value = {'Accuracy': 0.9, 'Precision': 0.8, 'Recall': 0.85}

        model, info = self.model_gen.run_model_generator(configs)

        # DataTransformer should be instantiated once
        mock_dt_cls.assert_called_once()

        # add_missing_proteins called for both train and validate
        self.assertEqual(dt_instance.add_missing_proteins.call_count, 2)
        dt_instance.filter_rules.assert_not_called()

        # transform_df and prep_vectorized_pairs_for_scikitlearn called for train and validate
        self.assertEqual(dt_instance.transform_df.call_count, 2)
        self.assertEqual(dt_instance.prep_vectorized_pairs_for_scikitlearn.call_count, 2)

        # train_model called with train_matrix
        mock_train.assert_called_once()
        train_call_kwargs = mock_train.call_args.kwargs
        self.assertIn('train_data', train_call_kwargs)
        self.assertIsInstance(train_call_kwargs['train_data'], pd.DataFrame)
        self.assertEqual(train_call_kwargs['train_data'].shape[0], self.train_quant.shape[0])

        # validate_model called once
        mock_validate.assert_called_once()
        validate_call_kwargs = mock_validate.call_args.kwargs
        self.assertIn('validate_data', validate_call_kwargs)
        self.assertIsInstance(validate_call_kwargs['validate_data'], pd.DataFrame)
        self.assertEqual(validate_call_kwargs['validate_data'].shape[0], self.validate_quant.shape[0])

        # save_model and save_model_information called with correct file names
        mock_save_model.assert_called_once()
        model_output_path = mock_save_model.call_args.kwargs['output_file_path']
        self.assertTrue(model_output_path.endswith("trained_model_and_model_metadata.pkl"))

        mock_save_info.assert_called_once()
        info_output_path = mock_save_info.call_args.kwargs['output_file_path']
        self.assertTrue(info_output_path.endswith("model_information.txt"))

        # returned model & info
        self.assertIs(model, dummy_model)
        self.assertIn('val_scores', info)
        self.assertEqual(info['val_scores']['Accuracy'], 0.9)

    @patch.object(ModelGenerator, 'save_model')
    @patch.object(ModelGenerator, 'save_model_information')
    @patch.object(ModelGenerator, 'validate_model')
    @patch.object(ModelGenerator, 'train_model')
    @patch('ModelGenerator.DataTransformer')
    def test_run_model_generator_filter_rules_false(self, mock_dt_cls, mock_train, mock_validate,
                                                    mock_save_info, mock_save_model):
        configs = self.base_configs.copy()
        configs['impute_NA_missing'] = False

        dt_instance = MagicMock()
        mock_dt_cls.return_value = dt_instance

        # filter_rules: just return the same feature_table
        dt_instance.filter_rules.side_effect = lambda feature_df, quant_df: feature_df

        bool_array = np.array([True, True, False, False])
        dt_instance.transform_df.return_value = {('P1', 'P2'): bool_array}
        dt_instance.prep_vectorized_pairs_for_scikitlearn.return_value = pd.DataFrame({
            ('P1', 'P2'): [1, 1, 0, 0]
        })

        dummy_model = MagicMock()
        model_info = {
            'cv_scores': {'Accuracy_Mean': 0.8},
            'params': {'dummy_param': 999}
        }
        mock_train.return_value = (dummy_model, model_info)
        mock_validate.return_value = {'Accuracy': 0.88, 'Precision': 0.82, 'Recall': 0.8}

        model, info = self.model_gen.run_model_generator(configs)

        # filter_rules should be used instead of add_missing_proteins
        self.assertEqual(dt_instance.filter_rules.call_count, 2)
        dt_instance.add_missing_proteins.assert_not_called()

        self.assertIs(model, dummy_model)
        self.assertIn('val_scores', info)
        self.assertAlmostEqual(info['val_scores']['Accuracy'], 0.88)


if __name__ == "__main__":
    unittest.main()
