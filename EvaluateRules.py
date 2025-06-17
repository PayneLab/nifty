import numpy as np
import pandas as pd


class EvaluateRules:
    def __init__(self):
        pass

    def vectorize_pair(self, pair: list, quant_df) -> np.ndarray:
        '''Gets all values for two proteins of a pair, compares them and returns a boolean vector'''
        prot1 = pair[0]
        prot2 = pair[1]

        prot1_values = quant_df[prot1].values
        prot2_values = quant_df[prot2].values

        #Check for NA values in either prot1_values or prot2_values using numpy
        # Copy the values to avoid modifying the original DataFrame
        prot1_values = prot1_values.copy()
        prot2_values = prot2_values.copy()

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
    
    def score_pair(self, pair: list, quant_df, meta_df) -> float:
        '''Scores a pair of proteins based on how well they separate the classes in the meta data'''
        bool_vector = EvaluateRules.vectorize_pair(self, pair, quant_df)
        class_labels = meta_df['classification_label'].values

        class_labels = np.array([1 if label == class_labels[0] else 0 for label in class_labels])

        # Find TP and FP values
        TP = np.sum((bool_vector == 1) & (class_labels == 1))
        FP = np.sum((bool_vector == 1) & (class_labels == 0))

        # Find proportion of TP and FP
        TP_prop = TP / np.sum(class_labels == 1)
        FP_prop = FP / np.sum(class_labels == 0)

        # Calculate Score
        score = abs(TP_prop - FP_prop)

        return score
    
    def evaluate_pairs(self, pairs: list, quant_df, meta_df) -> list:
        '''Evaluates all pairs of proteins and returns a list of tuples with the pair and its score'''
        scored_pairs = []
        for pair in pairs:
            score = self.score_pair(pair, quant_df, meta_df)
            scored_pairs.append((pair, score))
        # Another var with permutation base probability.
        return scored_pairs

    def randomize_labels(self, meta_df) -> pd.DataFrame:
        '''Randomizes the labels in the metadata and returns a new DataFrame.'''
        randomized_meta_df = meta_df.copy()
        randomized_meta_df['classification_label'] = np.random.permutation(meta_df['classification_label'].values)
        return randomized_meta_df
