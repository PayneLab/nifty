import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.stats.multitest as ssm
from collections import defaultdict


class EvaluateRules:
    def __init__(self):
        pass

    def NEW_Bm_vectorize_all_pairs(self, pairs: list, quant_df) -> dict:
        # Preload columns as arrays
        protein_arrays = {col: quant_df[col].to_numpy(copy=True) for col in quant_df.columns if col != "sample_id"}

        bool_dict = {}
        for prot1, prot2 in pairs:
            arr1 = protein_arrays[prot1].copy()
            arr2 = protein_arrays[prot2].copy()

            # NaN logic
            mask1_nan = np.isnan(arr1)
            mask2_nan = np.isnan(arr2)
            both_nan = mask1_nan & mask2_nan
            only1_nan = mask1_nan & ~mask2_nan
            only2_nan = mask2_nan & ~mask1_nan

            arr1[only1_nan] = 0;
            arr2[only1_nan] = 10
            arr2[only2_nan] = 0;
            arr1[only2_nan] = 10
            arr1[both_nan] = 0;
            arr2[both_nan] = 10

            bool_dict[(prot1, prot2)] = arr1 > arr2

        return bool_dict

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

    def NEW_Bm_score_pair(self, pair: tuple, bool_dict: dict, binarized_labels: np.ndarray) -> float:
        bool_vector = bool_dict[pair]
        # These are all boolean arrays, so bitwise logic and dot product works fast
        TP = np.dot(bool_vector, binarized_labels)
        FP = np.dot(bool_vector, 1 - binarized_labels)

        TP_prop = TP / self._n_pos if self._n_pos else 0
        FP_prop = FP / self._n_neg if self._n_neg else 0

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

    def NEW_Bm_evaluate_pairs(self, pairs: list, bool_dict: dict, binarized_labels: np.ndarray) -> list:
        scored = []
        for pair in pairs:
            score = self.NEW_Bm_score_pair(pair, bool_dict, binarized_labels)
            scored.append((pair, score))
        return scored

    def batch_score_all_pairs_Bm(self, pairs: list, bool_dict: dict, binarized_labels: np.ndarray) -> list:
        """Vectorized version of evaluate_pairs."""
        # Step 1: Create bool matrix
        bool_matrix = np.vstack([bool_dict[pair] for pair in pairs])  # shape (n_pairs, n_samples)

        # Step 2: Compute TP and FP via dot products
        TP = bool_matrix @ binarized_labels  # vector of TP for each pair
        FP = bool_matrix @ (1 - binarized_labels)

        # Step 3: Convert to proportions
        TP_prop = TP / self._n_pos if self._n_pos > 0 else np.zeros_like(TP)
        FP_prop = FP / self._n_neg if self._n_neg > 0 else np.zeros_like(FP)

        scores = np.abs(TP_prop - FP_prop)

        # Step 4: Package back into list of (pair, score)
        return list(zip(pairs, scores))

    def randomize_labels(self, labels: np.ndarray) -> np.ndarray:
        '''Randomizes the labels in the metadata and returns a new DataFrame.'''
        #randomized_meta_df = meta_df.copy()
        return np.random.permutation(labels)

    def permutate(self, pairs: list, bool_dict, binarized_labels, n_permutations=100):
        ''' Runs a permutation test on all pairs to see how significant their classification
        score is under a null distribution '''

        true_scores = dict(self.evaluate_pairs(pairs, bool_dict, binarized_labels))

        permuted_scores = defaultdict(list)

        for i in range(n_permutations):
            shuffled_labels = self.randomize_labels(binarized_labels)
            scores = self.evaluate_pairs(pairs, bool_dict, shuffled_labels)
            for pair, score in scores:
                permuted_scores[pair].append(score)
        summary_df = self.summarize_permutation_stats(true_scores, permuted_scores)
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

    def summarize_permutation_stats(self, true_scores, permuted_scores) -> pd.DataFrame:
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
        '''Adjust p-values for multiple testing'''
        summary_df['FDR'] = ssm.fdrcorrection(summary_df.P_Value)[1]
        return summary_df

    def get_proportion_bucket(self, bool_vector: np.ndarray) -> int:
        '''Returns the proportions of true and false of each bucket for the vector of the rules'''
        # print(bool_vector)
        n_true = int(np.sum(bool_vector))
        n_false = len(bool_vector) - n_true
        #return tuple([n_true, n_false])
        return round((n_true / (n_true + n_false)) * 100)

    def get_rule_to_buckets(self, pairs, bool_dict) -> dict:
        rule_to_buckets = {}
        # ('P1', 'P2'): (7, 5)
        for pair in pairs:
            bool_vector = bool_dict[pair]
            bucket = self.get_proportion_bucket(bool_vector)
            rule_to_buckets[pair] = bucket
        return rule_to_buckets

    def create_null_distributions_for_p_values_testing(self, pairs, bool_dict, binarized_labels, rule_to_buckets):
        '''Creates buckets for permutation test results based on the proportion of true and false in each rule'''
        shuffled_labels = self.randomize_labels(binarized_labels)
        scores = self.evaluate_pairs(pairs, bool_dict, shuffled_labels)
        bucket_lists = defaultdict(list)

        for pair, score in scores:
            bucket_key = rule_to_buckets[pair]
            bucket_lists[bucket_key].append(score)
            #{(3, 2): [0.1, 0.2, 0.3], (2, 3): [0.4, 0.5],...}

        # Convert all buckets into numpy array here.
        buckets = {}
        for key, val in bucket_lists.items():
            buckets[key] = np.array(val)
        return buckets

    def summarize_bucket_stats(self, true_scores: dict, rule_to_buckets: dict, buckets) -> pd.DataFrame:
        '''Get the p-values for each rule comparing its true score with the null distribution corresponding to its
        bucket (n_sum, n_false)'''
        data = []
        for rule, true_score in true_scores.items():
            bucket_key = rule_to_buckets[rule]
            null_distribution = buckets.get(bucket_key)  # empty should never happen

            if null_distribution is None:
                # should never get into here
                p_value = np.nan
            else:
                count = np.sum(null_distribution >= true_score)
                p_value = count / len(null_distribution)

            data.append({
                "Gene_Pair": rule,
                "True_Score": true_score,
                "Bucket": bucket_key,
                "P_Value": p_value
            })

        summary_df = pd.DataFrame(data)
        return self.get_significant_pairs(summary_df)

    def evaluate_buckets_wrapper(self, pairs: list, quant_df, meta_df):
        ''' A wrapper function that evaluates pairs, builds null buckets by n_true and n_false and calculate p-values
        based on bucket distribution.'''
        # Evaluate pairs
        bool_dict = self.vectorize_all_pairs(pairs, quant_df)
        binarized_labels = self.binarize_labels(meta_df)
        true_scores = dict(self.evaluate_pairs(pairs, bool_dict, binarized_labels))

        # Generate buckets and assign distributions after one permutation.
        rule_to_buckets = self.get_rule_to_buckets(pairs, bool_dict)
        buckets = self.create_null_distributions_for_p_values_testing(pairs, bool_dict, binarized_labels,
                                                                      rule_to_buckets)

        # Get p-values using the buckets.
        summary_df = self.summarize_bucket_stats(true_scores, rule_to_buckets, buckets)
        return true_scores, summary_df

    def NEW_get_bucket_to_rules(self, pairs, bool_dict) -> dict:
        '''Group pairs into buckets'''
        bucket_to_rules = {}
        for pair in pairs:
            bool_vector = bool_dict[pair]
            bucket = self.get_proportion_bucket(bool_vector)
            if bucket not in bucket_to_rules:
                bucket_to_rules[bucket] = []
            bucket_to_rules[bucket].append(pair)
        return bucket_to_rules

    def NEW_create_null_distributions_for_p_values_testing(self, pairs, bool_dict, binarized_labels, bucket_to_rules):
        '''Creates buckets for permutation test results based on the proportion of true and false in each rule'''
        shuffled_labels = self.randomize_labels(binarized_labels)
        buckets = {}

        for bucket in bucket_to_rules:
            rules = bucket_to_rules[bucket]
            scores_with_rules = self.evaluate_pairs(rules, bool_dict, shuffled_labels)
            scores = []
            for pair, score in scores_with_rules:
                scores.append(score)
            buckets[bucket] = np.array(scores)
        return buckets

    def NEW_Bm_create_null_distributions_for_p_values_testing(self, pairs, bool_dict, binarized_labels,
                                                              bucket_to_rules):
        shuffled_labels = self.randomize_labels(binarized_labels)
        buckets = {}

        # Reuse score_pair logic efficiently
        for bucket, rules in bucket_to_rules.items():
            scores = [self.score_pair(pair, bool_dict, shuffled_labels) for pair in rules]
            buckets[bucket] = np.array(scores)

        return buckets

    def NEW_summarize_bucket_stats(self, true_scores: dict, bucket_to_rules: dict, buckets) -> pd.DataFrame:
        data = []
        for bucket, rules in bucket_to_rules.items():
            null_distribution = buckets[bucket]

            null_distribution_sorted = np.sort(null_distribution)
            null_distribution_len = len(null_distribution)

            for rule in rules:
                true_score = true_scores[rule]
                index = np.searchsorted(null_distribution_sorted, true_score, side='left')
                count = null_distribution_len - index
                p_value = count / null_distribution_len

                data.append({
                    "Gene_Pair": rule,
                    "True_Score": true_score,
                    "Bucket": bucket,
                    #"P_Value": p_value
                })
        summary_df = pd.DataFrame(data)
        #summary_df = self.get_significant_pairs(summary_df)
        return summary_df

    def NEW_filter_and_save_rules(self, summary_df: pd.DataFrame, output_file_path='output.tsv'):
        df = summary_df.sort_values(by=['FDR'])
        used = set()
        filtered = []

        for i, row in df.iterrows():
            p1, p2 = row['Gene_Pair']
            if self.disjoint and (p1 in used or p2 in used):
                continue
            filtered.append(row)
            if self.disjoint:
                used.update([p1, p2])
            if len(filtered) >= self.k:
                break

        filtered_df = pd.DataFrame(filtered).reset_index(drop=True)
        if 'P_Value' in filtered_df.columns:
            filtered_df = filtered_df.drop(columns=['P_Value'])

        filtered_df.to_csv(output_file_path, index=False, sep='\t')
        return filtered_df
