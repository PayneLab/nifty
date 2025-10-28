import sys
import pandas as pd
import numpy as np

from Colors import Colors

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

    def vectorize_pair(self, pair: tuple, quant_df) -> np.ndarray:
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
        prot2_values[both_nan] = 0

        bool_vector = prot1_values > prot2_values

        return bool_vector

    def transform_df(self, feature_df, quant_df):
        rules = list(zip(feature_df['Protein1'].tolist(), feature_df['Protein2'].tolist()))
        vectorized_pairs = self.vectorize_all_pairs(rules, quant_df)

        return vectorized_pairs

    def filter_rules(self, feature_df, quant_df):
        proteins = {}
        proteins.update(feature_df['Protein1'])
        proteins.update(feature_df['Protein2'])

        updated_feature_df = feature_df

        for protein in proteins:
            if protein not in quant_df.columns:
                updated_feature_df = updated_feature_df[~((updated_feature_df['Protein1'] == protein) | (updated_feature_df['Protein2'] == protein))]

        if len(updated_feature_df) < 1:
            print(f"{Colors.ERROR}ERROR: All rules filtered out due to missing proteins in the quant table.{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)
        
        return updated_feature_df

    def add_missing_proteins(self, feature_df, quant_df):
        proteins = {}
        proteins.update(feature_df['Protein1'])
        proteins.update(feature_df['Protein2'])

        all_missing = True
        for protein in proteins:
            if protein not in quant_df.columns:
                quant_df[protein] = np.nan
            else:
                all_missing = False

        if all_missing:
            print(f"{Colors.ERROR}ERROR: All proteins in rules are missing in the quant table.{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)
        
        return quant_df
    
    def prep_vectorized_pairs_for_scikitlearn(self, bool_dict):
        # TODO - takes bool_dict from vectorize pairs output and turns it into what it needs to be turned into for ML
        pass
