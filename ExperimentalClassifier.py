import sys
import os

import pandas as pd

from Colors import Colors
from DataTransformer import DataTransformer

class ExperimentalClassifier:

    def __init__(self):
        pass

    def predict_classes(self, configs, experimental_data):
        if configs['prediction_format'] == 'classes':
            predictions = configs['model'].predict(experimental_data)
        else:
            predictions = configs['model'].predict_proba(experimental_data)
        return predictions

    def format_predictions(self, configs, predictions):
        index = configs['experimental_quant_table'].index.copy()

        if configs['prediction_format'] == 'classes':
            formatted_predictions = pd.DataFrame({
                'sample_id': index, 
                'classification_label': predictions
            })
            formatted_predictions.set_index('sample_id', inplace=True)
        elif configs['prediction_format'] == "probabilities":
            classes = configs['model'].classes_
            class1_col_name = f'classification_probability_{classes[0]}'
            class2_col_name = f'classification_probability_{classes[1]}'

            formatted_predictions = pd.DataFrame({
                'sample_id': index, 
                class1_col_name: [prediction[0] for prediction in predictions], 
                class2_col_name: [prediction[1] for prediction in predictions]
            })
            formatted_predictions.set_index('sample_id', inplace=True)
        return formatted_predictions

    def save_predictions(self, predictions_df, output_file_path):
        predictions_df.to_csv(output_file_path, index=True, sep='\t')
        print(f"{Colors.INFO}INFO: Classifications saved to '{output_file_path}'.{Colors.END}", file=sys.stderr, flush=True)

    def run_experimental_classifier(self, configs):
        data_transformer = DataTransformer()
        print("ADDING MISSING PROTEINS AND IMPUTING NA", file=sys.stderr, flush=True)
        configs['experimental_quant_table'] = data_transformer.add_missing_proteins(feature_df=configs['feature_table'], quant_df=configs['experimental_quant_table'])

        print("TRANSFORMING DATA", file=sys.stderr, flush=True)
        experimental_bool_dict = data_transformer.transform_df(feature_df=configs['feature_table'], quant_df=configs['experimental_quant_table'])
        experimental_matrix = data_transformer.prep_vectorized_pairs_for_scikitlearn(feature_df=configs['feature_table'], bool_dict=experimental_bool_dict)
        experimental_matrix.index = configs['experimental_quant_table'].index.copy()

        print("CLASSIFYING SAMPLES", file=sys.stderr, flush=True)
        predictions = self.predict_classes(configs, experimental_matrix)

        print("SAVING CLASSIFICATIONS", file=sys.stderr, flush=True)
        formatted_predictions = self.format_predictions(configs=configs, predictions=predictions)

        predictions_output_path = os.path.join(configs['output_dir'], "predicted_classes.tsv")
        self.save_predictions(formatted_predictions, predictions_output_path)
        
        return formatted_predictions
