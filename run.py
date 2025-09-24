import pandas as pd
from sklearn.model_selection import train_test_split

from EvaluateRules import EvaluateRules
from GenerateRules import GenerateRules

def split_training_data(df, feature_selection_ratio=0.13, validation_ratio=0.2, random_state=42):
    try:
        # Assume label column is the last column
        label_col = df.columns[-1]
        y = df[label_col]

        # Step 1: Feature Selection split
        FS_df, rest_df = train_test_split(
            df,
            test_size=(1 - feature_selection_ratio),
            stratify=y,
            random_state=random_state
        )

        # Step 2: Train/Test vs Validation split
        rest_y = rest_df[label_col]
        adjusted_validation_ratio = validation_ratio / (1 - feature_selection_ratio)

        train_test_df, validation_df = train_test_split(
            rest_df,
            test_size=adjusted_validation_ratio,
            stratify=rest_y,
            random_state=random_state
        )

        return FS_df, train_test_df, validation_df

    except Exception as e:
        raise ValueError(f"Error splitting data: {e}")


def load_data(file_path, _sep=','):
    """Loads a CSV file into a Pandas DataFrame."""
    try:
        df = pd.read_csv(file_path, sep=_sep, low_memory=False, index_col=0)
        df = df.transpose()
        return df
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")


file_path = "data/na-allow-testing-dfs/proteinXsinglecell_unimputed_labeled_corrected.csv"
sep = ","

df = load_data(file_path, _sep=sep)
FS_df, train_test_df, validation_df = split_training_data(df)

FS_df.index.name = "sample_ID"
meta_df = FS_df.iloc[:, -1:]  # Assuming the last column is the label
FS_df = FS_df.iloc[:, :-1]    # All columns except the last one
FS_df = FS_df.apply(pd.to_numeric, errors='coerce')
meta_df = meta_df.apply(pd.to_numeric, errors='coerce')
meta_df.rename(columns={'label': 'classification_label'}, inplace=True)

rule_gen = GenerateRules()
pairs = rule_gen.generate_rule_pairs(FS_df)

evaluator = EvaluateRules()

bool_vectors = evaluator.TEST_vectorize_all_pairs(pairs, FS_df)
percentage = evaluator.get_percentage(bool_vectors)
print(f"Tie percentage overall: {percentage:.2f}%")



# print(type(bool_vectors))
#
#
# binarized_labels = evaluator.binarize_labels(meta_df)
# true_scores = dict(evaluator.evaluate_pairs(pairs, bool_vectors, binarized_labels))
#
# duplicate_cols = {}
# for pair, vec in bool_vectors.items():
#     int_vec = vec.astype(int)
#     str_vec = [str(x) for x in int_vec]
#
#     vec = "".join(str_vec)
#
#     if vec in duplicate_cols:
#         if duplicate_cols[vec][1] < true_scores[pair]:
#             duplicate_cols[vec] = (pair, true_scores[pair])
#     else:
#         duplicate_cols[vec] = (pair, true_scores[pair])
#
# total_rules = len(duplicate_cols)
# print(f"Rules kept after filtering: {total_rules}")



# for vec, count in duplicate_cols.items():
#     if count > 1:
#         total_rules -= (count - 1)
#
# print(f"Rules after filtering: {total_rules}")



'''
buckets_to_rule = evaluator.get_bucket_to_rules(pairs, bool_vectors)
buckets = evaluator.create_null_distributions_for_p_values_testing(bool_vectors, binarized_labels, buckets_to_rule)
expanded_buckets = evaluator.expand_small_null_distributions(buckets, bool_vectors, binarized_labels, buckets)


summary_df = evaluator.summarize_bucket_stats(true_scores, buckets_to_rule, expanded_buckets)
edges_df = evaluator.add_mutual_information(summary_df, bool_vectors, min_threshold=0.9)

pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)

print(edges_df.head(20).to_string())
'''
#print(edges_df)