import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.stats.multitest as ssm

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
    
    def score_pair(self, pair: list, quant_np: np.ndarray, binarized_labels: np.ndarray) -> float:
        '''Scores a pair of proteins based on how well they separate the classes in the metadata'''
        bool_vector = self.vectorize_pair(pair, quant_np)

        TP = np.sum((bool_vector == 1) & (binarized_labels == 1))
        FP = np.sum((bool_vector == 1) & (binarized_labels == 0))

        TP_prop = TP / self._n_pos if self._n_pos > 0 else 0
        FP_prop = FP / self._n_neg if self._n_neg > 0 else 0

        return abs(TP_prop - FP_prop)
    
    '''def evaluate_pairs(self, pairs: list, quant_df, meta_df) -> list:
        #Evaluates all pairs of proteins and returns a list of tuples with the pair and its score
        scored_pairs = []
        #permutated_score_pairs = []

        class_labels = meta_df['classification_label'].to_numpy()
        first_label = class_labels[0]
        binarized_labels = (class_labels == first_label).astype(int)

        for pair in pairs:
            score = self.score_pair(pair, quant_df, binarized_labels)
            scored_pairs.append((pair, score))
        # Another var with permutation base probability.
        return scored_pairs'''
    
    def evaluate_pairs(self, pairs: list, quant_df, meta_df) -> list:
        '''Evaluates all pairs of proteins and returns a list of tuples with the pair and its score'''

        # Keep this as a DataFrame so vectorize_pair can work with column names and NaN handling
        class_labels = meta_df['classification_label'].to_numpy()
        first_label = class_labels[0]
        binarized_labels = (class_labels == first_label).astype(int)

        # Precompute denominators
        self._n_pos = np.sum(binarized_labels == 1)
        self._n_neg = np.sum(binarized_labels == 0)

        # Loop is still fast since vectorize_pair is lean
        scored_pairs = [(pair, self.score_pair(pair, quant_df, binarized_labels)) for pair in pairs]

        return scored_pairs


    def randomize_labels(self, labels: np.ndarray) -> np.ndarray:
        '''Randomizes the labels in the metadata and returns a new DataFrame.'''
        #randomized_meta_df = meta_df.copy()
        return np.random.permutation(labels)

    def permutate(self, pairs: list, quant_df, meta_df, n_permutations=100):
        ''' Runs a permutation test on all pairs to see how significant their classification
        score is under a null distribution '''
        true_scores = dict(self.evaluate_pairs(pairs, quant_df, meta_df))
        permuted_scores = {}
        for pair in pairs:
            permuted_scores[pair] = []

        labels = meta_df['classification_label'].to_numpy()

        for i in range(n_permutations):
            shuffled_labels = self.randomize_labels(labels)
            randomized_meta_df = meta_df.copy()
            randomized_meta_df['classification_label'] = shuffled_labels
            scores = self.evaluate_pairs(pairs, quant_df, randomized_meta_df)
            for pair, score in scores:
                permuted_scores[pair].append(score)
        summary_df = self.summarize_stats(true_scores, permuted_scores)
        return summary_df

    def summarize_stats(self, true_scores, permuted_scores) -> pd.DataFrame:
        ''' Summarizes permutation test results for all evaluated feature pairs.'''
        permuted_df = pd.DataFrame.from_dict(permuted_scores, orient='index')
        # Transpose it.
        permuted_df = permuted_df.T
        pairs = list(permuted_df.columns)
        means = permuted_df.mean(axis=0)
        stds = permuted_df.std(axis=0)
        summary_data = []

        for pair in pairs:
            true = true_scores[pair]
            mean = means[pair]
            std = stds[pair]
            if std == 0 or np.isnan(std):
                # Temporary solution to return something.
                z_score = 0.0
                # So not significant per default.
                p_value = 1.0
            else:
                z_score = (true - mean) / std
                p_value = norm.sf(abs(z_score)) * 2
            # p_val = stats.norm.sf(abs(z_score)) * 2
            summary_data.append((pair, true, mean, std, z_score, p_value))

        summary_df = pd.DataFrame(summary_data, columns=['Gene_Pair', 'True_Score', 'Mean', 'Std', 'Z-Score', 'P_Value'])
        significant_pairs_df = self.get_significant_pairs(summary_df)
        return significant_pairs_df

    def get_significant_pairs(self, summary_df) -> pd.DataFrame:
        # Adjust p-values for multiple testing
        summary_df['FDR'] = ssm.fdrcorrection(summary_df.P_Value)[1]
        # summary_df['FDR'] = ssm.fdrcorrection(summary_df.P_Value)[1]
        return summary_df