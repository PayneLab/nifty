import sys
import os
import pickle

from Colors import Colors
from DataTransformer import DataTransformer

class ModelGenerator:

    def __init__(self):
        # TODO
        pass

    def optimize_model(self):
        # TODO
        pass

    def train_model(self):
        # TODO
        pass

    def validate_model(self):
        # TODO
        pass

    def save_model(self, model, output_file_path):
        with open(output_file_path, "wb") as f:
            pickle.dump(model, f)
        print(f"{Colors.INFO}INFO: Model saved to '{output_file_path}'.{Colors.END}", file=sys.stderr, flush=True)

    def save_performance_metrics(self, metrics, output_file_path):
        # TODO
        print(f"{Colors.INFO}INFO: Model performance metrics saved to '{output_file_path}'.{Colors.END}", file=sys.stderr, flush=True)

    def run_model_generator(self, configs):
        data_transformer = DataTransformer()
        if configs['impute_NA_missing']:
            print("ADDING MISSING PROTEINS AND IMPUTING NA", file=sys.stderr, flush=True)
            configs['train_quant_table'] = data_transformer.add_missing_proteins(feature_df=configs['feature_table'], quant_df=configs['train_quant_table'])
            configs['validate_quant_table'] = data_transformer.add_missing_proteins(feature_df=configs['feature_table'], quant_df=configs['validate_quant_table'])
        else:
            print("FILTERING OUT RULES CONTAINING MISSING PROTEINS")
            configs['feature_table'] = data_transformer.filter_rules(feature_df=configs['feature_table'], quant_df=configs['train_quant_table'])
            configs['feature_table'] = data_transformer.filter_rules(feature_df=configs['feature_table'], quant_df=configs['validate_quant_table'])
    
        print("TRANSFORMING DATA", file=sys.stderr, flush=True)
        train_bool_dict = data_transformer.transform_df(feature_df=configs['feature_table'], quant_df=configs['train_quant_table'])
        validate_bool_dict = data_transformer.transform_df(feature_df=configs['feature_table'], quant_df=configs['validate_quant_table'])

        train_matrix = data_transformer.prep_vectorized_pairs_for_scikitlearn(train_bool_dict)
        validate_matrix = data_transformer.prep_vectorized_pairs_for_scikitlearn(validate_bool_dict)

        # TODO: train model, 
        #       validate model

        # Save model to "trained_model.pkl" in the specified output dir (configs['output_dir'])
        print("SAVING MODEL", file=sys.stderr, flush=True)
        model_output_path = os.path.join(configs['output_dir'], "trained_model.pkl")
        self.save_model(model=model, output_file_path=model_output_path)

        # Save train/validate information to "model_performance_metrics.???" in the specified output dir
        print("SAVING MODEL PERFORMANCE METRICS", file=sys.stderr, flush=True)
        metrics_output_path = os.path.join(configs['output_dir'], "model_performance_metrics.txt")  # TODO: fix file extension when function is written
        self.save_performance_metrics(metrics=performance_metrics, output_file_path=metrics_output_path)
        
        return model
