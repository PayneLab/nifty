## TODO

class DataTransformer:

    def __init__(self):
        pass

    def transform_df(self, feature_df, quant_df):
        # TODO: prepare a quant table for model training --> make feature table
        #       move code from EvaluateRules here and have EvaluateRules call this

        return vectorized_pairs

    def filter_rules(self, feature_df, quant_df):
        # TODO: filter out rules that don't have proteins in the quant_df
        
        return updated_feature_df

    def add_null_proteins(self, feature_df, quant_df):
        # TODO: add in missing proteins and populate with NA in preparation for transform
        
        return updated_quant_df
