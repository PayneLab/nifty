import sys
import os

from Colors import Colors
from DataTransformer import DataTransformer

class ExperimentalClassifier:

    def __init__(self):
        pass

    def predict_classes(self):
        pass

    def format_predictions(self):
        # TODO: write a function that takes the predicted classes and formats them like a meta_df 
        #       (sample_id index, classification_label column with the classifications)
        pass

    def save_predictions(self, predictions_df):
        pass

    def run_experimental_classifier(self, configs):
        data_transformer = DataTransformer()
        print("ADDING MISSING PROTEINS AND IMPUTING NA", file=sys.stderr, flush=True)
        configs['experimental_quant_table'] = data_transformer.add_missing_proteins(feature_df=configs['feature_table'], quant_df=configs['experimental_quant_table'])

        print("TRANSFORMING DATA", file=sys.stderr, flush=True)
        experimental_bool_dict = data_transformer.transform_df(feature_df=configs['feature_table'], quant_df=configs['experimental_quant_table'])

        experimental_matrix = data_transformer.prep_vectorized_pairs_for_scikitlearn(experimental_bool_dict)

        # TODO: predict classes

        # TODO: save predictions to "predicted_classes.tsv" in the specified output dir (Experimental Classifier)
        pass
