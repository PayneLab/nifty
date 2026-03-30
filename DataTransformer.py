import sys
import pandas as pd
import numpy as np

from Colors import Colors

class DataTransformer:

    def __init__(self):
        pass

    def vectorize_all_pairs(self, pairs: list, quant_df) -> tuple[dict, np.ndarray]:
        """Vectorizes all pairs of proteins and returns a matrix of boolean vectors.
        Rows are indexed by the string representation of each pair."""
        
        filled_df = quant_df.fillna(-np.inf)
        quant_matrix = filled_df.to_numpy() 
        
        col_to_idx = {col: i for i, col in enumerate(quant_df.columns)}
        
        idx1 = [col_to_idx[p[0]] for p in pairs]
        idx2 = [col_to_idx[p[1]] for p in pairs]
        
        prot1_matrix = quant_matrix[:, idx1].T 
        prot2_matrix = quant_matrix[:, idx2].T 
        
        final_matrix = (prot1_matrix > prot2_matrix).astype(np.int8)
                
        return final_matrix

    def filter_rules(self, feature_df, quant_df):
        proteins = set()
        proteins.update(feature_df['Protein1'].tolist())
        proteins.update(feature_df['Protein2'].tolist())

        updated_feature_df = feature_df.copy()

        for protein in proteins:
            if protein not in quant_df.columns:
                updated_feature_df = updated_feature_df[~((updated_feature_df['Protein1'] == protein) | (updated_feature_df['Protein2'] == protein))]

        if len(updated_feature_df) < 1:
            print(f"{Colors.ERROR}ERROR: All rules filtered out due to missing proteins in the quant table.{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)
        
        return updated_feature_df
    
    def create_feature_table_from_model(self, model):
        protein1 = [feature.split(">")[0] for feature in model.feature_names_in_]
        protein2 = [feature.split(">")[1] for feature in model.feature_names_in_]
        feature_df = pd.DataFrame({
            "Protein1": protein1, 
            "Protein2": protein2
        })
        return feature_df

    def add_missing_proteins(self, feature_df, quant_df):
        proteins = set()
        proteins.update(feature_df['Protein1'].tolist())
        proteins.update(feature_df['Protein2'].tolist())

        updated_quant_df = quant_df.copy()

        all_missing = True
        for protein in proteins:
            if protein not in quant_df.columns:
                updated_quant_df[protein] = np.nan
            else:
                all_missing = False

        if all_missing:
            print(f"{Colors.ERROR}ERROR: All proteins in rules are missing in the quant table.{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)
        
        return updated_quant_df
    
    def prep_vectorized_pairs_for_scikitlearn(self, rules, bool_matrix):
        # Convert dict values (bool arrays) to int arrays
        pairs = [">".join(rule) for rule in rules]
        bool_df = pd.DataFrame()

        for i, rule in enumerate(rules):
            pair = ">".join(rule)
            bool_df[pair] = [int(eval) for eval in bool_matrix[i, :]]

        bool_df = bool_df[pairs]

        return bool_df
