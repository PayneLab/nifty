import numpy as np
import pandas as pd
import statsmodels.stats.multitest as ssm
# import networkx as nx
from sklearn.metrics import normalized_mutual_info_score


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

    def TEST_vectorize_all_pairs(self, pairs: list, quant_df) -> dict:
        '''Vectorizes all pairs of proteins and returns a dictionary of boolean vectors.
        Rows are indexed by the string representation of each pair.'''
        bool_dict = {}
        for pair in pairs:
            bool_vector = self.TEST_vectorize_pairs(pair, quant_df)
            bool_dict[pair] = bool_vector
        return bool_dict

    def TEST_vectorize_pairs(self, pair: list, quant_df) -> np.ndarray:
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
        prot2_values[both_nan] = 0

        bool_table = np.where(prot1_values > prot2_values, 1, np.where(prot1_values < prot2_values, 0, 2))

        return bool_table

    def get_percentage(self, bool_table):
        total = 0
        total_ties = 0
        for _, arrays in bool_table.items():
            total += len(arrays)
            total_ties += np.sum(arrays == 2)
        return (total_ties / total) * 100

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

    def get_bucket_to_rules(self, pairs, bool_dict) -> dict:
        '''Group pairs into buckets'''
        bucket_to_rules = {}
        for pair in pairs:
            bool_vector = bool_dict[pair]
            bucket = self.get_proportion_bucket(bool_vector)
            if bucket not in bucket_to_rules:
                bucket_to_rules[bucket] = []
            bucket_to_rules[bucket].append(pair)
        return bucket_to_rules

    def create_null_distributions_for_p_values_testing(self, bool_dict, binarized_labels, bucket_to_rules):
        shuffled_labels = self.randomize_labels(binarized_labels)
        buckets = {}

        # Reuse score_pair logic efficiently
        for bucket, rules in bucket_to_rules.items():
            scores = [self.score_pair(pair, bool_dict, shuffled_labels) for pair in rules]
            buckets[bucket] = np.array(scores)
        return buckets

    def expand_small_null_distributions(self, buckets, bool_dict, binarized_labels, bucket_to_rules):
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
        #summary_df = self.get_significant_pairs(summary_df)
        return summary_df

    def add_mutual_information(self, summary_df, bool_vectors, min_threshold=0.9, max_rules=200):
        rules = list(summary_df.nsmallest(max_rules, "P_Value")['Gene_Pair'])
        pval_map = dict(zip(summary_df['Gene_Pair'], summary_df['P_Value']))

        edges = []
        for i in range(len(rules)):
            v1 = bool_vectors[rules[i]]
            for j in range(i + 1, len(rules)):
                v2 = bool_vectors[rules[j]]
                mi = normalized_mutual_info_score(v1, v2)
                if mi >= min_threshold:
                    edges.append({
                        "Source_Rule": rules[i],
                        "Target_Rule": rules[j],
                        "MI_Score": mi,
                        "Source_P_Value": pval_map[rules[i]],
                        "Target_P_Value": pval_map[rules[j]],
                    })
        return pd.DataFrame(edges)

    '''
    def cluster_by_mi_and_filter(self, summary_df, edges_df):
        graph = nx.Graph()

        for _, row in summary_df.iterrows():
            rule_identifier = row['Gene_Pair']
            p_value_for_rule = row['P_Value']
            graph.add_node(rule_identifier, p_value=p_value_for_rule)

        for _, row in edges_df.iterrows():
            source_rule = row['Source_Rule']
            target_rule = row['Target_Rule']
            mi_score = row['MI_Score']
            graph.add_edge(source_rule, target_rule, weight=mi_score)

        #print('Nodes with p-values')
        #print(graph.nodes(data=True))

        #print('Edges with MI scores')
        #print(graph.edges(data=True))

        winners = []
        already_selected = set()

        for cluster in nx.connected_components(graph):
            if cluster & already_selected:
                continue

            best_rule = None
            best_p_value = float('inf')

            for rule_identifier in cluster:
                p_value_for_rule = graph.nodes[rule_identifier]['p_value']
                if p_value_for_rule < best_p_value:
                    best_rule = rule_identifier
                    best_p_value = p_value_for_rule
            winners.append(best_rule)
            already_selected.add(best_rule)
        return winners
    '''

    def filter_and_save_rules(self, summary_df: pd.DataFrame, k: int,
                              disjoint=True, output_file_path='output.tsv'):
        # Sort by p-value ASC, then True_Score DESC
        df = summary_df.sort_values(['P_Value', 'True_Score'],
                                    ascending=[True, False])
        used = set()
        filtered = []

        if disjoint:
            for _, row in df.iterrows():
                p1, p2 = row['Gene_Pair']
                if p1 in used or p2 in used:
                    continue
                filtered.append(row.to_dict())
                used.update([p1, p2])
                if len(filtered) >= k:
                    break
        else:
            filtered = df.head(k).to_dict('records')

        filtered_df = pd.DataFrame(filtered).reset_index(drop=True)
        if 'P_Value' in filtered_df.columns:
            filtered_df = filtered_df.drop(columns=['P_Value'])
        if disjoint and len(filtered_df) < k:
            print(f"Only {len(filtered_df)} disjoint pairs available (requested {k}).", flush=True)
        filtered_df.to_csv(output_file_path, index=False, sep='\t')
        return filtered_df

    #Wrappers:

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

    def evaluate_buckets_wrapper(self, pairs: list, quant_df, meta_df, mi_threshold=0.9):
        ''' A wrapper function that evaluates pairs, builds null buckets by n_true and n_false and calculate p-values
        based on bucket distribution.'''
        bool_dict = self.vectorize_all_pairs(pairs, quant_df)
        binarized_labels = self.binarize_labels(meta_df)
        true_scores = dict(self.evaluate_pairs(pairs, bool_dict, binarized_labels))
        bucket_to_rules = self.get_bucket_to_rules(pairs, bool_dict)
        buckets = self.create_null_distributions_for_p_values_testing(bool_dict, binarized_labels, bucket_to_rules)
        expanded_buckets = self.expand_small_null_distributions(buckets, bool_dict, binarized_labels, bucket_to_rules)

        summary_df = self.summarize_bucket_stats(true_scores, bucket_to_rules, expanded_buckets)
        edges_df = self.add_mutual_information(summary_df, bool_dict, min_threshold=mi_threshold)
        return true_scores, summary_df, edges_df
