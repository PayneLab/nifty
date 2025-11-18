import sys
import os
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from ExperimentalClassifier import ExperimentalClassifier


class TestPredictClasses(unittest.TestCase):
    """
    Tests for ExperimentalClassifier.predict_classes
    """

    def setUp(self):
        self.cls = ExperimentalClassifier()
        # Simple dummy experimental matrix (3 samples x 2 features)
        self.experimental_matrix = pd.DataFrame({
            'f1': [0.1, 0.2, 0.3],
            'f2': [1.0, 0.9, 0.8]
        }, index=['s1', 's2', 's3'])

    def test_predict_classes_format_classes(self):
        """
        When prediction_format == 'classes', use model.predict().
        """
        model = MagicMock()
        model.predict.return_value = np.array([0, 1, 0])

        configs = {
            'prediction_format': 'classes',
            'model': model
        }

        preds = self.cls.predict_classes(configs, self.experimental_matrix)

        model.predict.assert_called_once()
        pd.testing.assert_frame_equal(
            self.experimental_matrix,
            model.predict.call_args.args[0]
        )
        np.testing.assert_array_equal(preds, np.array([0, 1, 0]))

    def test_predict_classes_format_probabilities(self):
        """
        When prediction_format == 'probabilities', use model.predict_proba().
        """
        model = MagicMock()
        model.predict_proba.return_value = np.array([
            [0.2, 0.8],
            [0.6, 0.4],
            [0.1, 0.9]
        ])

        configs = {
            'prediction_format': 'probabilities',
            'model': model
        }

        preds = self.cls.predict_classes(configs, self.experimental_matrix)

        model.predict_proba.assert_called_once()
        pd.testing.assert_frame_equal(
            self.experimental_matrix,
            model.predict_proba.call_args.args[0]
        )
        np.testing.assert_array_equal(
            preds,
            np.array([[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
        )


class TestFormatPredictions(unittest.TestCase):
    """
    Tests for ExperimentalClassifier.format_predictions
    """

    def setUp(self):
        self.cls = ExperimentalClassifier()
        # Fake experimental quant table (index = sample IDs)
        self.experimental_quant = pd.DataFrame({
            'P1': [0.1, 0.2, 0.3],
            'P2': [0.4, 0.5, 0.6]
        }, index=['s1', 's2', 's3'])

    def test_format_predictions_classes(self):
        """
        For prediction_format == 'classes', should return a DataFrame with
        index sample_id and column 'classification_label'.
        """
        configs = {
            'prediction_format': 'classes',
            'experimental_quant_table': self.experimental_quant
        }
        predictions = np.array([0, 1, 1])

        formatted = self.cls.format_predictions(configs, predictions)

        expected = pd.DataFrame({
            'classification_label': [0, 1, 1]
        }, index=['s1', 's2', 's3'])
        # match the index name set by format_predictions
        expected.index.name = 'sample_id'

        pd.testing.assert_frame_equal(formatted, expected)

    def test_format_predictions_probabilities(self):
        """
        For prediction_format == 'probabilities', create two probability columns:
        classification_probability_<class0> and classification_probability_<class1>.
        """
        class_labels = np.array(['KO', 'WT'])
        model = MagicMock()
        model.classes_ = class_labels

        configs = {
            'prediction_format': 'probabilities',
            'experimental_quant_table': self.experimental_quant,
            'model': model
        }

        predictions = np.array([
            [0.1, 0.9],
            [0.7, 0.3],
            [0.4, 0.6]
        ])

        formatted = self.cls.format_predictions(configs, predictions)

        expected = pd.DataFrame({
            f'classification_probability_{class_labels[0]}': [0.1, 0.7, 0.4],
            f'classification_probability_{class_labels[1]}': [0.9, 0.3, 0.6]
        }, index=['s1', 's2', 's3'])
        expected.index.name = 'sample_id'

        pd.testing.assert_frame_equal(formatted, expected)


class TestSavePredictions(unittest.TestCase):
    """
    Tests for ExperimentalClassifier.save_predictions
    """

    def setUp(self):
        self.cls = ExperimentalClassifier()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_save_predictions_writes_tsv_with_index(self):
        """
        save_predictions should write a tab-separated file with index included.
        """
        df = pd.DataFrame({
            'classification_label': [0, 1, 0]
        }, index=['s1', 's2', 's3'])
        df.index.name = 'sample_id'

        output_path = os.path.join(self.temp_dir, "predicted_classes.tsv")

        self.cls.save_predictions(df, output_path)

        self.assertTrue(os.path.exists(output_path))

        loaded = pd.read_csv(output_path, sep='\t', index_col=0)
        loaded.index.name = 'sample_id'

        pd.testing.assert_frame_equal(loaded, df)


class TestRunExperimentalClassifier(unittest.TestCase):
    """
    Tests for ExperimentalClassifier.run_experimental_classifier
    """

    def setUp(self):
        self.cls = ExperimentalClassifier()
        self.temp_dir = tempfile.mkdtemp()

        self.feature_table = pd.DataFrame({
            'Protein1': ['P1'],
            'Protein2': ['P2']
        })

        self.experimental_quant = pd.DataFrame({
            'sample_id': ['s1', 's2', 's3'],
            'P1': [0.1, 0.2, 0.3],
            'P2': [0.4, 0.5, 0.6]
        }).set_index('sample_id')

        self.model = MagicMock()

        self.base_configs = {
            'feature_table': self.feature_table.copy(),
            'experimental_quant_table': self.experimental_quant.copy(),
            'model': self.model,
            'prediction_format': 'classes',
            'output_dir': self.temp_dir
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch.object(ExperimentalClassifier, 'save_predictions')
    @patch.object(ExperimentalClassifier, 'format_predictions')
    @patch.object(ExperimentalClassifier, 'predict_classes')
    @patch('ExperimentalClassifier.DataTransformer')
    def test_run_experimental_classifier_flow(self, mock_dt_cls, mock_predict,
                                              mock_format, mock_save):
        """
        Verify full data flow with mocks.
        """
        configs = self.base_configs.copy()

        dt_instance = MagicMock()
        mock_dt_cls.return_value = dt_instance

        dt_instance.add_missing_proteins.side_effect = lambda feature_df, quant_df: quant_df
        dt_instance.transform_df.return_value = {('P1', 'P2'): np.array([True, False, True])}
        dt_instance.prep_vectorized_pairs_for_scikitlearn.return_value = pd.DataFrame({
            ('P1', 'P2'): [1, 0, 1]
        }, index=['s1', 's2', 's3'])

        mock_predict.return_value = np.array([0, 1, 0])

        formatted_df = pd.DataFrame({
            'classification_label': [0, 1, 0]
        }, index=['s1', 's2', 's3'])
        formatted_df.index.name = 'sample_id'
        mock_format.return_value = formatted_df

        result_df = self.cls.run_experimental_classifier(configs)

        mock_dt_cls.assert_called_once()

        dt_instance.add_missing_proteins.assert_called_once()
        dt_instance.transform_df.assert_called_once()
        dt_instance.prep_vectorized_pairs_for_scikitlearn.assert_called_once()

        mock_predict.assert_called_once()
        pred_args, pred_kwargs = mock_predict.call_args
        self.assertIs(pred_args[0], configs)
        self.assertIsInstance(pred_args[1], pd.DataFrame)

        mock_format.assert_called_once()
        fmt_args, fmt_kwargs = mock_format.call_args
        self.assertIs(fmt_kwargs['configs'], configs)
        np.testing.assert_array_equal(fmt_kwargs['predictions'], np.array([0, 1, 0]))

        mock_save.assert_called_once()
        save_args, save_kwargs = mock_save.call_args
        self.assertIs(save_args[0], formatted_df)
        self.assertTrue(save_args[1].endswith("predicted_classes.tsv"))

        pd.testing.assert_frame_equal(result_df, formatted_df)

    @patch('ExperimentalClassifier.DataTransformer')
    def test_run_experimental_classifier_integration_minimal(self, mock_dt_cls):
        """
        Light integration test with a fake model.
        """
        configs = self.base_configs.copy()
        configs['prediction_format'] = 'probabilities'

        dt_instance = MagicMock()
        mock_dt_cls.return_value = dt_instance

        dt_instance.add_missing_proteins.side_effect = lambda feature_df, quant_df: quant_df
        dt_instance.transform_df.return_value = {('P1', 'P2'): np.array([True, True, True])}
        dt_instance.prep_vectorized_pairs_for_scikitlearn.return_value = pd.DataFrame({
            ('P1', 'P2'): [1, 1, 1]
        }, index=['s1', 's2', 's3'])

        class FakeModel:
            def __init__(self):
                self.classes_ = np.array(['KO', 'WT'])

            def predict(self, X):
                return np.array(['KO'] * len(X))

            def predict_proba(self, X):
                return np.tile(np.array([0.3, 0.7]), (len(X), 1))

        configs['model'] = FakeModel()

        result_df = self.cls.run_experimental_classifier(configs)

        output_path = os.path.join(configs['output_dir'], "predicted_classes.tsv")
        self.assertTrue(os.path.exists(output_path))

        expected_cols = {
            'classification_probability_KO',
            'classification_probability_WT'
        }
        self.assertEqual(set(result_df.columns), expected_cols)
        self.assertEqual(list(result_df.index), ['s1', 's2', 's3'])
        np.testing.assert_allclose(
            result_df['classification_probability_KO'].values,
            np.array([0.3, 0.3, 0.3])
        )
        np.testing.assert_allclose(
            result_df['classification_probability_WT'].values,
            np.array([0.7, 0.7, 0.7])
        )


if __name__ == "__main__":
    unittest.main()
