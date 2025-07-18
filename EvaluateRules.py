import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.stats.multitest as ssm
from collections import defaultdict

class EvaluateRules:
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

    def score_pair(self, pair: list, bool_dict, binarized_labels: np.ndarray) -> float:
        '''Scores a pair of proteins based on how well they separate the classes in the metadata'''
        bool_vector = bool_dict[pair]

        TP = np.sum((bool_vector == 1) & (binarized_labels == 1))
        FP = np.sum((bool_vector == 1) & (binarized_labels == 0))

        TP_prop = TP / self._n_pos if self._n_pos > 0 else 0
        FP_prop = FP / self._n_neg if self._n_neg > 0 else 0

        return abs(TP_prop - FP_prop)

    def binarize_labels(self, meta_df) -> np.ndarray:
        '''Binarizes the labels in the metadata and computes denominators. Returns binarized labels'''
        class_labels = meta_df['classification_label'].to_numpy()
        first_label = class_labels[0]

        binarized_labels = (class_labels == first_label).astype(int)

        # precompute denominators for score calculation
        self._n_pos = np.sum(binarized_labels == 1)
        self._n_neg = np.sum(binarized_labels == 0)

        return binarized_labels

    def evaluate_pairs(self, pairs: list, bool_dict, binarized_labels) -> list:
        '''Evaluates all pairs of proteins and returns a list of tuples with the pair and its score'''
        # score pairs
        scored_pairs = [(pair, self.score_pair(pair, bool_dict, binarized_labels)) for pair in pairs]

        return scored_pairs

    def randomize_labels(self, labels: np.ndarray) -> np.ndarray:
        '''Randomizes the labels in the metadata and returns a new DataFrame.'''
        #randomized_meta_df = meta_df.copy()
        return np.random.permutation(labels)

    def permutate(self, pairs: list, bool_dict, binarized_labels,n_permutations=100):
        ''' Runs a permutation test on all pairs to see how significant their classification
        score is under a null distribution '''

        true_scores = dict(self.evaluate_pairs(pairs, bool_dict, binarized_labels))

        permuted_scores = defaultdict(list)

        for i in range(n_permutations):
            shuffled_labels = self.randomize_labels(binarized_labels)
            scores = self.evaluate_pairs(pairs, bool_dict, shuffled_labels)
            for pair, score in scores:
                permuted_scores[pair].append(score)
        summary_df = self.summarize_stats(true_scores, permuted_scores)
        return summary_df

    def evaluate_permutate_wrapper(self, pairs: list, quant_df, meta_df, n_permutations=100):
        ''' A wrapper function that evaluates pairs and runs permutation test on them.'''
        # Evaluate pairs
        bool_dict = self.vectorize_all_pairs(pairs, quant_df)
        binarized_labels = self.binarize_labels(meta_df)

        # Get true scores
        true_scores = dict(self.evaluate_pairs(pairs, bool_dict, binarized_labels))

        # Run permutation test
        summary_df = self.permutate(pairs, bool_dict, binarized_labels, n_permutations)

        return true_scores, summary_df

    def summarize_stats(self, true_scores, permuted_scores) -> pd.DataFrame:
        '''Uses vectorization to summarize permutation test results for all evaluated feature pairs.'''
        # Create a DataFrame of permuted scores (samples x pairs)
        permuted_df = pd.DataFrame.from_dict(permuted_scores, orient='index').T

        # Compute mean and std across permutations for each pair
        means = permuted_df.mean()
        stds = permuted_df.std()

        # Convert true_scores to a pandas Series for alignment
        true_scores_series = pd.Series(true_scores)

        # Align index (important if true_scores may be a superset/subset)
        true_scores_series = true_scores_series[permuted_df.columns]

        # Vectorized z-score calculation with safe handling of std = 0
        z_scores = (true_scores_series - means) / stds.replace(0, np.nan)
        z_scores = z_scores.fillna(0.0)

        # Two-tailed p-values
        p_values = norm.sf(z_scores)
        #p_values = norm.sf(np.abs(z_scores)) * 2

        # Build summary DataFrame
        summary_df = pd.DataFrame({
            'True_Score': true_scores_series.values,
            'Mean': means.values,
            'Std': stds.values,
            'Z-Score': z_scores.values,
            'P_Value': p_values
        }, index=z_scores.index)
        summary_df.index.name = 'Gene_Pair'


        # Filter significant pairs
        return self.get_significant_pairs(summary_df)

    def get_significant_pairs(self, summary_df) -> pd.DataFrame:
        # Adjust p-values for multiple testing
        summary_df['FDR'] = ssm.fdrcorrection(summary_df.P_Value)[1]
        # summary_df['FDR'] = ssm.fdrcorrection(summary_df.P_Value)[1]
        return summary_df

    def get_proportion_bucket(self, bool_vector: np.ndarray) -> tuple:
        '''Returns the proportions of true and false of each bucket for the vector of the rules'''
        n_true = int(np.sum(bool_vector))
        n_false = len(bool_vector) - n_true
        return tuple([n_true, n_false])






