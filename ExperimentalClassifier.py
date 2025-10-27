import sys
import os

from Colors import Colors
from DataTransformer import DataTransformer

class ExperimentalClassifier:

    def __init__(self):
        pass

    def predict_classes(self, model, experimental_data):
        predictions = model.predict(experimental_data)
        return predictions

    def format_predictions(self, predictions):
        # TODO: write a function that takes the predicted classes and formats them like a meta_df 
        #       (sample_id as index, classification_label column with the classifications)
        pass

    def save_predictions(self, predictions_df, output_file_path):
        predictions_df.to_csv(output_file_path, index=True, sep='\t')
        print(f"{Colors.INFO}INFO: Classifications saved to '{output_file_path}'.{Colors.END}", file=sys.stderr, flush=True)

    def run_experimental_classifier(self, configs):
        data_transformer = DataTransformer()
        print("ADDING MISSING PROTEINS AND IMPUTING NA", file=sys.stderr, flush=True)
        configs['experimental_quant_table'] = data_transformer.add_missing_proteins(feature_df=configs['feature_table'], quant_df=configs['experimental_quant_table'])

        print("TRANSFORMING DATA", file=sys.stderr, flush=True)
        experimental_bool_dict = data_transformer.transform_df(feature_df=configs['feature_table'], quant_df=configs['experimental_quant_table'])
        experimental_matrix = data_transformer.prep_vectorized_pairs_for_scikitlearn(experimental_bool_dict)

        print("CLASSIFYING SAMPLES", file=sys.stderr, flush=True)
        predictions = self.predict_classes(configs['model'], experimental_matrix)

        print("SAVING CLASSIFICATIONS", file=sys.stderr, flush=True)
        # TODO: format predictions
        formatted_predictions = self.format_predictions(predictions)

        predictions_output_path = os.path.join(configs['output_dir'], "predicted_classes.tsv")
        self.save_predictions(formatted_predictions, predictions_output_path)
        
        return formatted_predictions
