import sys

from DataTransformer import DataTransformer

class ModelGenerator:

    def __init__(self):
        pass

    def optimize_model(self):
        pass

    def train_model(self):
        pass

    def validate_model(self):
        pass

    def save_model(self):
        pass

    def save_performance_metrics(self):
        pass

    def run_model_generator(self, configs):
        data_transformer = DataTransformer()
        if configs['impute_NA_missing']:
            print("ADDING MISSING PROTEINS AND IMPUTING NA", file=sys.stderr, flush=True)
            configs['train_quant_table'] = data_transformer.add_null_proteins(feature_df=configs['feature_table'], quant_df=configs['train_quant_table'])
            configs['validate_quant_table'] = data_transformer.add_null_proteins(feature_df=configs['feature_table'], quant_df=configs['validate_quant_table'])
        else:
            print("FILTERING OUT RULES CONTAINING MISSING PROTEINS")
            configs['feature_table'] = data_transformer.filter_rules(feature_df=configs['feature_table'], quant_df=configs['train_quant_table'])
            configs['feature_table'] = data_transformer.filter_rules(feature_df=configs['feature_table'], quant_df=configs['validate_quant_table'])
    
        print("TRANSFORMING DATA", file=sys.stderr, flush=True)
        # TODO: take proteins from feature table and make list of tuples needed for vectorization
        
        train_bool_dict = data_transformer.transform_df(feature_df=configs['feature_table'], quant_df=configs['train_quant_table'])
        validate_bool_dict = data_transformer.transform_df(feature_df=configs['feature_table'], quant_df=configs['validate_quant_table'])

        # TODO: turn binary dict into df for model training

        # TODO: train and save model to "trained_model.pkl" in the specified output dir, 
        #       validate model, 
        #       save train/validate information to "model_performance_metrics.???" in the specified output dir, (ModelGenerator) 
        
        return model
