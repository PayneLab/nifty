import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.stats.multitest as ssm
from collections import defaultdict

class EvaluateRules:
    def __init__(self):
        pass

    # Turn this into vectorize all pairs, call it only once outside of all loops. Return a matrix of bools.
    def vectorize_all_pairs_blake(self, pairs: list, quant_df) -> dict:
        #TODO possibly make this a dictionary of bool vectors, where keys are pairs. Returns a dict instead of DF
        # TODO make sure keys are the same as the keys in the other dicts (permutate)
        '''Vectorizes all pairs of proteins and returns a DataFrame of boolean vectors.
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

    def score_pair(self, pair: list, quant_np: np.ndarray, binarized_labels: np.ndarray) -> float:
        '''Scores a pair of proteins based on how well they separate the classes in the metadata'''
        bool_vector = self.vectorize_pair(pair, quant_np)

        TP = np.sum((bool_vector == 1) & (binarized_labels == 1))
        FP = np.sum((bool_vector == 1) & (binarized_labels == 0))

        TP_prop = TP / self._n_pos if self._n_pos > 0 else 0
        FP_prop = FP / self._n_neg if self._n_neg > 0 else 0

        return abs(TP_prop - FP_prop)
    
    def score_pair_2_blake(self, pair: list, bool_dict, binarized_labels: np.ndarray) -> float:
        '''Scores a pair of proteins based on how well they separate the classes in the metadata'''
        bool_vector = bool_dict[pair]

        TP = np.sum((bool_vector == 1) & (binarized_labels == 1))
        FP = np.sum((bool_vector == 1) & (binarized_labels == 0))

        TP_prop = TP / self._n_pos if self._n_pos > 0 else 0
        FP_prop = FP / self._n_neg if self._n_neg > 0 else 0

        return abs(TP_prop - FP_prop)
    
    def binarize_labels_blake(self, meta_df) -> np.ndarray:
        '''Binarizes the labels in the metadata and computes denominators. Returns binarized labels'''
        class_labels = meta_df['classification_label'].to_numpy()
        first_label = class_labels[0]

        binarized_labels = (class_labels == first_label).astype(int)

        # precompute denominators for score calculation
        self._n_pos = np.sum(binarized_labels == 1)
        self._n_neg = np.sum(binarized_labels == 0)

        return binarized_labels
    
    def evaluate_pairs_2_blake(self, pairs: list, bool_dict, binarized_labels) -> list:
        '''Evaluates all pairs of proteins and returns a list of tuples with the pair and its score'''
        # score pairs
        scored_pairs = [(pair, self.score_pair_2_blake(pair, bool_dict, binarized_labels)) for pair in pairs]

        return scored_pairs
    
    def evaluate_pairs(self, pairs: list, quant_df, meta_df) -> list:
        '''Evaluates all pairs of proteins and returns a list of tuples with the pair and its score'''

        # binarize the labels
        class_labels = meta_df['classification_label'].to_numpy()
        first_label = class_labels[0]
        binarized_labels = (class_labels == first_label).astype(int)

        # precompute denominators for score calculation
        self._n_pos = np.sum(binarized_labels == 1)
        self._n_neg = np.sum(binarized_labels == 0)

        # score pairs
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

    # TODO Make a wrapper (main) function that does everything, permutation just permutates
    def permutate_2_blake(self, pairs: list, quant_df, meta_df, n_permutations=100):
        ''' Runs a permutation test on all pairs to see how significant their classification
        score is under a null distribution '''
        bool_dict = self.vectorize_all_pairs_blake(pairs, quant_df)
        binarized_labels = self.binarize_labels_blake(meta_df)

        true_scores = dict(self.evaluate_pairs_2_blake(pairs, bool_dict, binarized_labels))
        
        permuted_scores = defaultdict(list)
        
        for i in range(n_permutations):
            shuffled_labels = self.randomize_labels(binarized_labels)
            scores = self.evaluate_pairs_2_blake(pairs, bool_dict, shuffled_labels)
            for pair, score in scores:
                permuted_scores[pair].append(score)
        summary_df = self.summarize_stats_2_blake(true_scores, permuted_scores)
        return summary_df
    
    def permutate_2_blake_final(self, pairs: list, bool_dict, binarized_labels,n_permutations=100):
        ''' Runs a permutation test on all pairs to see how significant their classification
        score is under a null distribution '''

        true_scores = dict(self.evaluate_pairs_2_blake(pairs, bool_dict, binarized_labels))
        
        permuted_scores = defaultdict(list)
        
        for i in range(n_permutations):
            shuffled_labels = self.randomize_labels(binarized_labels)
            scores = self.evaluate_pairs_2_blake(pairs, bool_dict, shuffled_labels)
            for pair, score in scores:
                permuted_scores[pair].append(score)
        summary_df = self.summarize_stats_2_blake(true_scores, permuted_scores)
        return summary_df
    
    def evaluate_permutate_blake(self, pairs: list, quant_df, meta_df, n_permutations=100):
        ''' A wrapper function that evaluates pairs and runs permutation test on them.'''
        # Evaluate pairs
        bool_dict = self.vectorize_all_pairs_blake(pairs, quant_df)
        binarized_labels = self.binarize_labels_blake(meta_df)

        # Get true scores
        true_scores = dict(self.evaluate_pairs_2_blake(pairs, bool_dict, binarized_labels))

        # Run permutation test
        summary_df = self.permutate_2_blake_final(pairs, bool_dict, binarized_labels, n_permutations)
        
        return true_scores, summary_df

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
    
    def summarize_stats_2_blake(self, true_scores, permuted_scores) -> pd.DataFrame:
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
        p_values = norm.sf(np.abs(z_scores)) * 2

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

    ################################################################################################################
    #####################################################################TEST######################################
    ################################################################################################################
    def Ben_get_bool_vectors_for_pairs(self, pairs: list, quant_df):
        '''Precomputes and returns boolean vectors for each protein pair.'''
        bool_vector_dictionary = {}
        for pair in pairs:
            bool_vector_dictionary[pair] = self.Ben_vectorize_pair(pair, quant_df)
        return bool_vector_dictionary

    def Ben_vectorize_pair(self, pair: list, quant_df) -> np.ndarray:
        '''Gets all values for two proteins of a pair, compares them and returns a boolean vector'''
        prot1_values = quant_df[pair[0]].to_numpy(copy=True)
        prot2_values = quant_df[pair[1]].to_numpy(copy=True)

        mask1_nan = np.isnan(prot1_values)
        mask2_nan = np.isnan(prot2_values)

        both_nan = mask1_nan & mask2_nan
        only1_nan = mask1_nan & ~mask2_nan
        only2_nan = mask2_nan & ~mask1_nan

        prot1_values[only1_nan] = 0
        prot2_values[only1_nan] = 10

        prot2_values[only2_nan] = 0
        prot1_values[only2_nan] = 10

        prot1_values[both_nan] = 0
        prot2_values[both_nan] = 10

        bool_vector = prot1_values > prot2_values

        return bool_vector

    def Ben_score_pair(self, bool_vector: np.ndarray, binarized_labels: np.ndarray, n_pos: int, n_neg: int) -> float:
        '''Scores a pair of proteins based on how well they separate the classes in the metadata'''
        TP = np.sum((bool_vector == 1) & (binarized_labels == 1))
        FP = np.sum((bool_vector == 1) & (binarized_labels == 0))

        TP_prop = TP / n_pos if n_pos > 0 else 0
        FP_prop = FP / n_neg if n_neg > 0 else 0

        return abs(TP_prop - FP_prop)

    def Ben_binarize_labels(self, meta_df):
        class_labels = meta_df['classification_label'].to_numpy()
        first_label = class_labels[0]
        binarized_labels = (class_labels == first_label).astype(int)
        return binarized_labels

    def Ben_evaluate_pairs(self, pairs: list, bool_dict, binarized_labels) -> list:
        n_pos = np.sum(binarized_labels == 1)
        n_neg = np.sum(binarized_labels == 0)
        scored_pairs = [(pair, self.Ben_score_pair(bool_vector, binarized_labels, n_pos, n_neg)) for pair, bool_vector in bool_dict.items()]
        return scored_pairs

    def Ben_get_true_scores(self, pairs: list, bool_vectors: dict, meta_df):
        binarized_labels = self.Ben_binarize_labels(meta_df)
        scored_pairs = self.Ben_evaluate_pairs(pairs, bool_vectors, binarized_labels)
        return dict(scored_pairs), binarized_labels

    def Ben_randomize_labels(self, labels: np.ndarray) -> np.ndarray:
        '''Randomizes the labels in the metadata and returns a new DataFrame.'''
        return np.random.permutation(labels)

    def Ben_permutate(self, pairs: list, bool_vectors: dict, binarized_labels: np.ndarray, true_scores: dict, n_permutations=100):
        ''' Runs a permutation test on all pairs to see how significant their classification
        score is under a null distribution '''
        permuted_scores_dic = {}
        for pair in pairs:
            permuted_scores_dic[pair] = []

        n_pos = np.sum(binarized_labels == 1)
        n_neg = np.sum(binarized_labels == 0)

        for i in range(n_permutations):
            shuffled_labels = self.Ben_randomize_labels(binarized_labels)
            for pair, bool_vector in bool_vectors.items():
                score = self.Ben_score_pair(bool_vector, shuffled_labels, n_pos, n_neg)
                permuted_scores_dic[pair].append(score)
        summary_df = self.Ben_summarize_stats(true_scores, permuted_scores_dic)
        return summary_df

<<<<<<< HEAD
    def evaluate_permutate_Ben(self, pairs: list, quant_df, meta_df, n_permutations=100):
        ''' A wrapper function that evaluates pairs and runs permutation test on them.'''
        # Get boolean vectors for pairs
        bool_vectors = self.get_bool_vectors_for_pairs(pairs, quant_df)

        # Get true scores
        true_scores = self.get_true_scores(bool_vectors, meta_df)

        # Run permutation test
        summary_df = self.permutate_Ben(pairs, bool_vectors, meta_df, true_scores, n_permutations)
        
        return true_scores, summary_df
=======
    def Ben_summarize_stats(self, true_scores, permuted_scores) -> pd.DataFrame:
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
                p_value = norm.sf(z_score)
                #p_value = norm.sf(abs(z_score)) * 2
            summary_data.append((pair, true, mean, std, z_score, p_value))

        summary_df = pd.DataFrame(summary_data, columns=['Gene_Pair', 'True_Score', 'Mean', 'Std', 'Z-Score', 'P_Value'])
        significant_pairs_df = self.Ben_get_significant_pairs(summary_df)
        return significant_pairs_df

    def Ben_get_significant_pairs(self, summary_df) -> pd.DataFrame:
        summary_df['FDR'] = ssm.fdrcorrection(summary_df.P_Value)[1]
        return summary_df


>>>>>>> 898d662545754edb6d2e911c93559839e30008e6
