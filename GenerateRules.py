from itertools import combinations
import pandas as pd

class GenerateRules:
    def __init__(self):
        pass

    def get_protein_list(self, quant_df):
        """Extracts the list of proteins from the quantification DataFrame."""

        protein_list = list(quant_df.columns)
        return protein_list
    
    def generate_rule_pairs(self, quant_df):
        """Generates all possible pairs of proteins from the quant df."""

        protein_list = self.get_protein_list(quant_df)
        rule_pairs = list(combinations(protein_list, 2))
        print(f"INFO: {len(rule_pairs)} rules generated from {len(quant_df.columns)} proteins.", file=sys.stderr, flush=True)
        return rule_pairs
    