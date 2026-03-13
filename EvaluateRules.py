import sys
import os

import numpy as np
import pandas as pd
import statsmodels.stats.multitest as ssm
from sklearn.metrics import normalized_mutual_info_score

from Colors import Colors
from DataTransformer import DataTransformer

class EvaluateRules:
    def __init__(self, seed=None):
        self.seed = seed

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

    def evaluate_pairs(self, pairs: list, bool_matrix, binarized_labels) -> list:
        """Evaluates all pairs of proteins and returns a list of tuples with the pair and its score"""
        
        # score pairs
        inverse = 1 - binarized_labels

        TP = bool_matrix.dot(binarized_labels)
        FP = bool_matrix.dot(inverse)

        TP_prop = TP / self._n_pos if self._n_pos > 0 else 0
        FP_prop = FP / self._n_neg if self._n_neg > 0 else 0

        final_scores = np.abs(TP_prop - FP_prop)
        
        scored_pairs = zip(pairs, final_scores)

        return scored_pairs

    def evaluate_null(self, bool_matrix, binarized_labels) -> list:
        """Evaluates all pairs of proteins and returns a list of tuples with the pair and its score"""
        
        # score pairs
        inverse = 1 - binarized_labels

        TP = bool_matrix.dot(binarized_labels)
        FP = bool_matrix.dot(inverse)

        TP_prop = TP / self._n_pos if self._n_pos > 0 else 0
        FP_prop = FP / self._n_neg if self._n_neg > 0 else 0

        final_scores = np.abs(TP_prop - FP_prop)

        return final_scores
    
    def get_proportion_bucket_list(self, bool_matrix) -> np.ndarray:
        """
        Takes the full (N_pairs, N_samples) matrix and returns a 1D array of buckets.
        """
        n_trues = np.sum(bool_matrix, axis=1)
        total_samples = bool_matrix.shape[1]
        percentages = (n_trues / total_samples) * 100
        buckets = np.round(percentages).astype(int)
        return buckets

    
    def bookkeeping(self, true_scores, null_scores, bool_matrix) -> tuple[dict,dict]:
        """
        make score-key rule
        make score-key null score lists
        """
        bucket_list = self.get_proportion_bucket_list(bool_matrix)
        bucket_to_rules = {}
        bucket_to_null_scores = {}
        
        zip(true_scores, null_scores, bucket_list)
        for pair, true_score, null_score, bucket in zip(true_scores.keys(), true_scores.values(), null_scores, bucket_list):
            # assign pair to bucket
            if bucket not in bucket_to_rules:
                bucket_to_rules[bucket] = []
            bucket_to_rules[bucket].append(pair)
            # assign true score to pair
            true_scores[pair] = true_score
            # assign null score to pair
            if bucket not in bucket_to_null_scores:
                bucket_to_null_scores[bucket] = []
            bucket_to_null_scores[bucket].append(null_score)
        return bucket_to_rules, bucket_to_null_scores

    def randomize_labels(self, labels: np.ndarray) -> np.ndarray:
        '''Randomizes the labels in the metadata and returns a new DataFrame.'''
        if hasattr(self, "seed") and self.seed is not None:
            rng = np.random.default_rng(self.seed)
            shuffled = rng.permutation(labels)
            self.seed += 1
            return shuffled
        else:
            return np.random.permutation(labels)
    
    def create_null_matrix(self, bool_matrix, binarized_labels) -> np.ndarray:
        shuffled_labels = self.randomize_labels(binarized_labels)
        null_matrix = self.evaluate_null(bool_matrix, shuffled_labels)
        return null_matrix

    def expand_small_null_distributions(self, buckets, bool_dict, binarized_labels, bucket_to_rules) -> dict:
        for bucket, scores in buckets.items():
            n = len(scores)
            if n < 100:
                needed_permutations = int(np.ceil(100 / n))
                scores_all = list(scores)

                for i in range(needed_permutations - 1):
                    shuffled_labels = self.randomize_labels(binarized_labels)
                    additional_scores = [self.score_pair(pair, bool_dict, shuffled_labels) for pair in
                                         bucket_to_rules[bucket]]
                    scores_all.extend(additional_scores)
                buckets[bucket] = np.array(scores_all)
        return buckets

    def summarize_bucket_stats(self, true_scores: dict, bucket_to_rules: dict, buckets) -> pd.DataFrame:
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
                    "P_Value": p_value
                })
        summary_df = pd.DataFrame(data)
        return summary_df

    def filter_rules(self, summary_df, bool_vectors, k, mutual_info, mi_cutoff, disjoint):
        df = summary_df.sort_values(['P_Value', 'True_Score'],
                                    ascending=[True, False])
        used_rules = set()
        used_proteins = set()
        filtered = []

        if disjoint and mutual_info:
            for _, row in df.iterrows():
                rule = row['Gene_Pair']
                p1 = rule[0]
                p2 = rule[1]

                if p1 in used_proteins or p2 in used_proteins:
                    continue

                if not used_rules:
                    used_rules.add(rule)
                    used_proteins.update([p1, p2])
                    filtered.append(row.to_dict())
                    continue

                redundant = False
                for kept in used_rules:
                    mi = self.calculate_mutual_information(rule, kept, bool_vectors)
                    if mi >= mi_cutoff:
                        redundant = True
                        break
                if not redundant:
                    used_rules.add(rule)
                    used_proteins.update([p1, p2])
                    filtered.append(row.to_dict())
                if len(filtered) >= k:
                    break
        elif disjoint and not mutual_info:
            for _, row in df.iterrows():
                p1, p2 = row['Gene_Pair']
                if p1 in used_proteins or p2 in used_proteins:
                    continue
                filtered.append(row.to_dict())
                used_proteins.update([p1, p2])
                if len(filtered) >= k:
                    break
        elif mutual_info and not disjoint:
            for _, row in df.iterrows():
                rule = row['Gene_Pair']
                if not used_rules:
                    used_rules.add(rule)
                    filtered.append(row.to_dict())
                    continue
                redundant = False
                for kept in used_rules:
                    mi = self.calculate_mutual_information(rule, kept, bool_vectors)
                    if mi >= mi_cutoff:
                        redundant = True
                        break
                if not redundant:
                    used_rules.add(rule)
                    filtered.append(row.to_dict())
                if len(filtered) >= k:
                    break
        else:
            filtered = df.head(k).to_dict('records')

        filtered_df = pd.DataFrame(filtered).reset_index(drop=True)

        if disjoint and mutual_info and len(filtered_df) < k:
            print(f"{Colors.WARNING}WARNING: Only {len(filtered_df)} disjoint pairs with low mutual information available (requested {k}).{Colors.END}",
                  file=sys.stderr, flush=True)
        elif mutual_info and not disjoint and len(filtered_df) < k:
            print(f"{Colors.WARNING}WARNING: Only {len(filtered_df)} pairs with low mutual information available (requested {k}).{Colors.END}",
                  file=sys.stderr, flush=True)
        elif disjoint and not mutual_info and len(filtered_df) < k:
            print(f"{Colors.WARNING}WARNING: Only {len(filtered_df)} disjoint pairs available (requested {k}).{Colors.END}",
                  file=sys.stderr, flush=True)

        return filtered_df

    def calculate_mutual_information(self, pair1, pair2, bool_vectors):
        vec1 = bool_vectors[pair1]
        vec2 = bool_vectors[pair2]
        mi = normalized_mutual_info_score(vec1, vec2)
        return mi

    def save_rules(self, filtered_df: pd.DataFrame, output_file_path: str):
        filtered_df['Protein1'] = filtered_df['Gene_Pair'].apply(lambda x: x[0])
        filtered_df['Protein2'] = filtered_df['Gene_Pair'].apply(lambda x: x[1])

        filtered_df.rename(columns={'Gene_Pair': 'Protein_Pair', 'True_Score': 'Score'}, inplace=True)
        output_filtered_df = filtered_df.drop('Bucket', axis=1)
        output_filtered_df = output_filtered_df.iloc[:, [0, 3, 4, 1, 2]]

        output_filtered_df.to_csv(output_file_path, index=False, sep='\t')
        print(f"{Colors.INFO}INFO: Rules saved to '{output_file_path}'.{Colors.END}", file=sys.stderr, flush=True)

    #Wrapper:
    def run_rule_evaluator(self, configs, pairs: list, quant_df, meta_df):
        ''' A wrapper function that evaluates pairs, builds null buckets by n_true and n_false and calculate p-values
        based on bucket distribution.'''

        print(" - GENERATING RULE TABLE", file=sys.stderr, flush=True)
        data_transformer = DataTransformer()
        bool_dict, bool_matrix = data_transformer.vectorize_all_pairs(pairs, quant_df)

        print(" - BINARIZING LABELS", file=sys.stderr, flush=True)
        binarized_labels = self.binarize_labels(meta_df)

        print(" - SCORING RULES", file=sys.stderr, flush=True)
        true_scores = dict(self.evaluate_pairs(pairs, bool_matrix, binarized_labels))

        print("EVALUATING SCORES", file=sys.stderr, flush=True)
        null_scores = self.create_null_matrix(bool_matrix, binarized_labels)
        bucket_to_rules, bucket_to_null_scores = self.bookkeeping(true_scores,null_scores,bool_matrix)
        expanded_buckets = self.expand_small_null_distributions(bucket_to_null_scores, bool_dict, binarized_labels, bucket_to_rules)

        summary_df = self.summarize_bucket_stats(true_scores, bucket_to_rules, expanded_buckets)

        print("FILTERING RULES", file=sys.stderr, flush=True)
        filtered_df = self.filter_rules(summary_df, bool_dict, k=configs['k_rules'], mutual_info=configs['mutual_information'], mi_cutoff=configs['mutual_information_cutoff'], disjoint=configs['disjoint'])

        print("SAVING RULES", file=sys.stderr, flush=True)
        output_file_path = os.path.join(configs['output_dir'], "selected_features.tsv")
        self.save_rules(filtered_df, output_file_path)

        return true_scores, summary_df, filtered_df

