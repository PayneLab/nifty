## TODO
import sys
import pandas as pd
import numpy as np

class DataTransformer:

    def __init__(self):
        pass

    def vectorize_all_pairs(self, pairs: list, quant_df) -> dict:
        '''Vectorizes all pairs of proteins and returns a dictionary of boolean vectors.
        Rows are indexed by the string representation of each pair.'''
        bool_dict = {}
        for pair in pairs:
            bool_vector = self.vectorize_pair(pair, quant_df)
            bool_dict[pair] = bool_vector
        return bool_dict

    def vectorize_pair(self, pair: list, quant_df) -> np.ndarray:
        '''Gets all values for two proteins of a pair, compares them and returns a boolean vector'''
        prot1_values = quant_df[pair[0]].to_numpy(copy=True)
        prot2_values = quant_df[pair[1]].to_numpy(copy=True)

        # Create masks for NaN combinations
        mask1_nan = np.isnan(prot1_values)
        mask2_nan = np.isnan(prot2_values)

        both_nan = mask1_nan & mask2_nan
        only1_nan = mask1_nan & ~mask2_nan
        only2_nan = mask2_nan & ~mask1_nan

        # Apply replacement logic
        prot1_values[only1_nan] = 0
        prot2_values[only1_nan] = 10

        prot2_values[only2_nan] = 0
        prot1_values[only2_nan] = 10

        prot1_values[both_nan] = 0
        prot2_values[both_nan] = 10

        bool_vector = prot1_values > prot2_values

        return bool_vector

    def transform_df(self, feature_df, quant_df):
        rules = list(zip(feature_df['Protein1'].tolist(), feature_df['Protein2'].tolist()))
        vectorized_pairs = self.vectorize_all_pairs(rules, quant_df)

        return vectorized_pairs

    def filter_rules(self, feature_df, quant_df):
        # TODO: filter out rules that don't have proteins in the quant_df
        
        return updated_feature_df

    def add_null_proteins(self, feature_df, quant_df):
        # TODO: add in missing proteins and populate with NA in preparation for transform
        
        return updated_quant_df
