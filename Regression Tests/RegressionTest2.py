# REgression test 2 - Large Dataset with NA values
import pandas as pd
import cProfile
import numpy as np
import sys
import os
import pstats
import time

current_dir = os.path.dirname(__file__)

parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from GenerateRules import GenerateRules
from EvaluateRules import EvaluateRules
from DataTableChecker import DataTableChecker


def test_large_imbalanced_NA(num_samples=500, num_proteins=1000):


    sample_ids = [f'Sample {i+1}' for i in range(num_samples)]

    # assign labels
    labels_H = np.random.choice(['H'], size=470)
    labels_D = np.random.choice(['D'], size=30)
    labels = np.concatenate((labels_H, labels_D))
    np.random.shuffle(labels)

    meta_df = pd.DataFrame({
        'sample_id': sample_ids,
        'classification_label': labels
    })

    quant_data = {
        'sample_id': sample_ids
    }

    # handle randomly generated proteins
    excluded = {10, 20, 30, 40}
    for i in range(1, num_proteins + 1):
        if i in excluded:
            continue
        values = np.random.randint(0, 150, size=num_samples).astype(float)
        # Add NaNs randomly
        nan_indices = np.random.choice(num_samples, size=int(0.05 * num_samples), replace=False)
        values[nan_indices] = np.nan
        quant_data[f'Protein {i}'] = values

    # perfect 20/30 pair for score of 1.0
    prot_20, prot_30 = [], []
    for label in labels:
        base = np.random.randint(30, 120)
        delta = np.random.randint(5, 20)
        if label == 'H':
            prot_20.append(base)
            prot_30.append(base - delta)
        else:
            prot_30.append(base)
            prot_20.append(base - delta)

    prot_20 = np.array(prot_20, dtype=float)
    prot_30 = np.array(prot_30, dtype=float)

    quant_data['Protein 20'] = prot_20
    quant_data['Protein 30'] = prot_30

    # generate pair 10/40 (poor separation)
    prot_10 = np.zeros(num_samples, dtype=float)
    prot_40 = np.zeros(num_samples, dtype=float)

    H_indices = np.where(labels == 'H')[0]
    D_indices = np.where(labels == 'D')[0]

    def assign_balanced_rule(indices):
        #enforces that exactly half of the indices will enforce one rule and the other half the opposite
        #in an even-number of proteins, will give us a score of 0.0
        n = len(indices)
        half = n // 2
        for i, idx in enumerate(indices):
            base = np.random.randint(30, 120)
            delta = np.random.randint(5, 20)
            if i < half:
                prot_10[idx] = base
                prot_40[idx] = base + delta
            else:
                prot_10[idx] = base + delta
                prot_40[idx] = base

    assign_balanced_rule(H_indices)
    assign_balanced_rule(D_indices)

    quant_data['Protein 10'] = prot_10
    quant_data['Protein 40'] = prot_40

    # build df
    quant_df = pd.DataFrame(quant_data)

    # checks
    checker = DataTableChecker()
    checker.check_meta_file(meta_df)
    checker.check_samples(meta_df, quant_df)
    checker.check_quant_data(quant_df)
    checker.check_duplicate_proteins(quant_df)

    # evaluate
    rule_gen = GenerateRules()
    pairs = rule_gen.generate_rule_pairs(quant_df)

    evaluator = EvaluateRules()
    results = evaluator.evaluate_pairs(pairs, quant_df, meta_df)

    perm_results = evaluator.permutate(pairs, quant_df, meta_df, n_permutations=10)
    print(perm_results)

    result_dict = dict(results)

    # assertions
    score_20_30 = result_dict.get(('Protein 20', 'Protein 30'), None)
    if score_20_30 < 1.0:
        raise AssertionError(f"Expected score 1.0 for ('Protein 20', 'Protein 30'), got {score_20_30}")
    else:
        print("Test passed: ('Protein 20', 'Protein 30') has score near 1.0.")
        print(f"Score for ('Protein 20', 'Protein 30'): {score_20_30}")

    score_10_40 = result_dict.get(('Protein 10', 'Protein 40'), None)
    if score_10_40 is None:
        raise AssertionError("Expected pair ('Protein 10', 'Protein 40') not found in results.")
    if score_10_40 > 0.0:
        raise AssertionError(f"Expected score ~0.0 for ('Protein 10', 'Protein 40'), got {score_10_40}")
    else:
        print("Test passed: ('Protein 10', 'Protein 40') has score near 0.0.")
        print(f"Score for ('Protein 10', 'Protein 40'): {score_10_40}")

def test_large_imbalanced_NA_2(num_samples=500, num_proteins=1000):

    sample_ids = [f'Sample {i+1}' for i in range(num_samples)]

    # assign labels
    n_H = int(0.94 * num_samples)
    n_D = int(0.06 * num_samples)
    labels_H = np.random.choice(['H'], size= n_H)
    labels_D = np.random.choice(['D'], size= n_D)
    labels = np.concatenate((labels_H, labels_D))
    np.random.shuffle(labels)

    meta_df = pd.DataFrame({
        'sample_id': sample_ids,
        'classification_label': labels
    })

    quant_data = {
        'sample_id': sample_ids
    }

    # handle randomly generated proteins
    excluded = {10, 20, 30, 40}
    for i in range(1, num_proteins + 1):
        if i in excluded:
            continue
        values = np.random.randint(0, 150, size=num_samples).astype(float)
        # Add NaNs randomly
        nan_indices = np.random.choice(num_samples, size=int(0.05 * num_samples), replace=False)
        values[nan_indices] = np.nan
        quant_data[f'Protein {i}'] = values

    # perfect 20/30 pair for score of 1.0
    prot_20, prot_30 = [], []
    for label in labels:
        base = np.random.randint(30, 120)
        delta = np.random.randint(5, 20)
        if label == 'H':
            prot_20.append(base)
            prot_30.append(base - delta)
        else:
            prot_30.append(base)
            prot_20.append(base - delta)

    prot_20 = np.array(prot_20, dtype=float)
    prot_30 = np.array(prot_30, dtype=float)

    quant_data['Protein 20'] = prot_20
    quant_data['Protein 30'] = prot_30

    # generate pair 10/40 (poor separation)
    prot_10 = np.zeros(num_samples, dtype=float)
    prot_40 = np.zeros(num_samples, dtype=float)

    H_indices = np.where(labels == 'H')[0]
    D_indices = np.where(labels == 'D')[0]

    def assign_balanced_rule(indices):
        #enforces that exactly half of the indices will enforce one rule and the other half the opposite
        #in an even-number of proteins, will give us a score of 0.0
        n = len(indices)
        half = n // 2
        for i, idx in enumerate(indices):
            base = np.random.randint(30, 120)
            delta = np.random.randint(5, 20)
            if i < half:
                prot_10[idx] = base
                prot_40[idx] = base + delta
            else:
                prot_10[idx] = base + delta
                prot_40[idx] = base

    assign_balanced_rule(H_indices)
    assign_balanced_rule(D_indices)

    quant_data['Protein 10'] = prot_10
    quant_data['Protein 40'] = prot_40

    # build df
    quant_df = pd.DataFrame(quant_data)

    # checks
    checker = DataTableChecker()
    checker.check_meta_file(meta_df)
    checker.check_samples(meta_df, quant_df)
    checker.check_quant_data(quant_df)
    checker.check_duplicate_proteins(quant_df)

    # evaluate
    rule_gen = GenerateRules()
    pairs = rule_gen.generate_rule_pairs(quant_df)

    evaluator = EvaluateRules()

    true_scores, perm_results = evaluator.evaluate_permutate_wrapper(pairs, quant_df, meta_df, n_permutations=100)
    print(perm_results)

    # assertions
    score_20_30 = perm_results.loc[('Protein 20', 'Protein 30'), 'True_Score']
    if score_20_30 is None:
        raise AssertionError("Expected pair ('Protein 20', 'Protein 30') not found in results.")
    if score_20_30 < 1.0:
        raise AssertionError(f"Expected score 1.0 for ('Protein 20', 'Protein 30'), got {score_20_30}")
    else:
        print("Test passed: ('Protein 20', 'Protein 30') has score near 1.0.")
        print(f"Score for ('Protein 20', 'Protein 30'): {score_20_30}")

    score_10_40 = perm_results.loc[('Protein 10', 'Protein 40'), 'True_Score']
    if score_10_40 is None:
        raise AssertionError("Expected pair ('Protein 10', 'Protein 40') not found in results.")
    if score_10_40 > 0.0:
        raise AssertionError(f"Expected score ~0.0 for ('Protein 10', 'Protein 40'), got {score_10_40}")
    else:
        print("Test passed: ('Protein 10', 'Protein 40') has score near 0.0.")
        print(f"Score for ('Protein 10', 'Protein 40'): {score_10_40}")

def test_large_imbalanced_NA_Ben(num_samples=500, num_proteins=1000):

    sample_ids = [f'Sample {i+1}' for i in range(num_samples)]

    # assign labels
    n_H = int(0.94 * num_samples)
    n_D = int(0.06 * num_samples)
    labels_H = np.random.choice(['H'], size= n_H)
    labels_D = np.random.choice(['D'], size= n_D)
    labels = np.concatenate((labels_H, labels_D))
    np.random.shuffle(labels)

    meta_df = pd.DataFrame({
        'sample_id': sample_ids,
        'classification_label': labels
    })

    quant_data = {
        'sample_id': sample_ids
    }

    # handle randomly generated proteins
    excluded = {10, 20, 30, 40}
    for i in range(1, num_proteins + 1):
        if i in excluded:
            continue
        values = np.random.randint(0, 150, size=num_samples).astype(float)
        # Add NaNs randomly
        nan_indices = np.random.choice(num_samples, size=int(0.05 * num_samples), replace=False)
        values[nan_indices] = np.nan
        quant_data[f'Protein {i}'] = values

    # perfect 20/30 pair for score of 1.0
    prot_20, prot_30 = [], []
    for label in labels:
        base = np.random.randint(30, 120)
        delta = np.random.randint(5, 20)
        if label == 'H':
            prot_20.append(base)
            prot_30.append(base - delta)
        else:
            prot_30.append(base)
            prot_20.append(base - delta)

    prot_20 = np.array(prot_20, dtype=float)
    prot_30 = np.array(prot_30, dtype=float)

    quant_data['Protein 20'] = prot_20
    quant_data['Protein 30'] = prot_30

    # generate pair 10/40 (poor separation)
    prot_10 = np.zeros(num_samples, dtype=float)
    prot_40 = np.zeros(num_samples, dtype=float)

    H_indices = np.where(labels == 'H')[0]
    D_indices = np.where(labels == 'D')[0]

    def assign_balanced_rule(indices):
        #enforces that exactly half of the indices will enforce one rule and the other half the opposite
        #in an even-number of proteins, will give us a score of 0.0
        n = len(indices)
        half = n // 2
        for i, idx in enumerate(indices):
            base = np.random.randint(30, 120)
            delta = np.random.randint(5, 20)
            if i < half:
                prot_10[idx] = base
                prot_40[idx] = base + delta
            else:
                prot_10[idx] = base + delta
                prot_40[idx] = base

    assign_balanced_rule(H_indices)
    assign_balanced_rule(D_indices)

    quant_data['Protein 10'] = prot_10
    quant_data['Protein 40'] = prot_40

    # build df
    quant_df = pd.DataFrame(quant_data)

    # checks
    checker = DataTableChecker()
    checker.check_meta_file(meta_df)
    checker.check_samples(meta_df, quant_df)
    checker.check_quant_data(quant_df)
    checker.check_duplicate_proteins(quant_df)

    # evaluate
    rule_gen = GenerateRules()
    pairs = rule_gen.generate_rule_pairs(quant_df)

    evaluator = EvaluateRules()

    true_scores, perm_results = evaluator.evaluate_permutate_Ben(pairs, quant_df, meta_df, n_permutations=100)
    print(perm_results)

def test_new_method(num_samples=500, num_proteins=1000):

    sample_ids = [f'Sample {i+1}' for i in range(num_samples)]

    # assign labels
    n_H = int(0.94 * num_samples)
    n_D = int(0.06 * num_samples)
    labels_H = np.random.choice(['H'], size= n_H)
    labels_D = np.random.choice(['D'], size= n_D)
    labels = np.concatenate((labels_H, labels_D))
    np.random.shuffle(labels)

    meta_df = pd.DataFrame({
        'sample_id': sample_ids,
        'classification_label': labels
    })

    quant_data = {
        'sample_id': sample_ids
    }

    # handle randomly generated proteins
    excluded = {10, 20, 30, 40}
    for i in range(1, num_proteins + 1):
        if i in excluded:
            continue
        values = np.random.randint(0, 150, size=num_samples).astype(float)
        # Add NaNs randomly
        nan_indices = np.random.choice(num_samples, size=int(0.05 * num_samples), replace=False)
        values[nan_indices] = np.nan
        quant_data[f'Protein {i}'] = values

    # perfect 20/30 pair for score of 1.0
    prot_20, prot_30 = [], []
    for label in labels:
        base = np.random.randint(30, 120)
        delta = np.random.randint(5, 20)
        if label == 'H':
            prot_20.append(base)
            prot_30.append(base - delta)
        else:
            prot_30.append(base)
            prot_20.append(base - delta)

    prot_20 = np.array(prot_20, dtype=float)
    prot_30 = np.array(prot_30, dtype=float)

    quant_data['Protein 20'] = prot_20
    quant_data['Protein 30'] = prot_30

    # generate pair 10/40 (poor separation)
    prot_10 = np.zeros(num_samples, dtype=float)
    prot_40 = np.zeros(num_samples, dtype=float)

    H_indices = np.where(labels == 'H')[0]
    D_indices = np.where(labels == 'D')[0]

    def assign_balanced_rule(indices):
        #enforces that exactly half of the indices will enforce one rule and the other half the opposite
        #in an even-number of proteins, will give us a score of 0.0
        n = len(indices)
        half = n // 2
        for i, idx in enumerate(indices):
            base = np.random.randint(30, 120)
            delta = np.random.randint(5, 20)
            if i < half:
                prot_10[idx] = base
                prot_40[idx] = base + delta
            else:
                prot_10[idx] = base + delta
                prot_40[idx] = base

    assign_balanced_rule(H_indices)
    assign_balanced_rule(D_indices)

    quant_data['Protein 10'] = prot_10
    quant_data['Protein 40'] = prot_40

    # build df
    quant_df = pd.DataFrame(quant_data)

    # checks
    checker = DataTableChecker()
    checker.check_meta_file(meta_df)
    checker.check_samples(meta_df, quant_df)
    #checker.check_quant_data(quant_df) TODO fix this bc it fails here
    checker.check_duplicate_proteins(quant_df)

    # evaluate
    rule_gen = GenerateRules()
    pairs = rule_gen.generate_rule_pairs(quant_df)

    evaluator = EvaluateRules()

    bool_vectors = evaluator.vectorize_all_pairs(pairs, quant_df)
    binarized_labels = evaluator.binarize_labels(meta_df)
    true_scores = dict(evaluator.evaluate_pairs(pairs, bool_vectors, binarized_labels))
    rule_to_buckets = evaluator.get_rule_to_buckets(pairs, bool_vectors)
    #buckets = evaluator.create_null_distributions_for_p_values_testing(pairs, bool_vectors, binarized_labels, rule_to_buckets)

    buckets_to_rule = evaluator.NEW_get_bucket_to_rules(pairs, bool_vectors)
    buckets = evaluator.NEW_create_null_distributions_for_p_values_testing(pairs, bool_vectors, binarized_labels, buckets_to_rule)

    for keys, values in buckets.items():
        print(f"{keys} → {len(values)} null scores, sample: {values[:5]}")
    print()

    #summary_df = evaluator.summarize_bucket_stats(true_scores, rule_to_buckets, buckets)
    summary_df = evaluator.NEW_summarize_bucket_stats(true_scores, buckets_to_rule, buckets)
    print(summary_df)

def test_newest_method(num_samples=500, num_proteins=1000):

    sample_ids = [f'Sample {i+1}' for i in range(num_samples)]

    # assign labels
    n_H = int(0.94 * num_samples)
    n_D = int(0.06 * num_samples)
    labels_H = np.random.choice(['H'], size= n_H)
    labels_D = np.random.choice(['D'], size= n_D)
    labels = np.concatenate((labels_H, labels_D))
    np.random.shuffle(labels)

    meta_df = pd.DataFrame({
        'sample_id': sample_ids,
        'classification_label': labels
    })

    quant_data = {
        'sample_id': sample_ids
    }

    # handle randomly generated proteins
    excluded = {10, 20, 30, 40}
    for i in range(1, num_proteins + 1):
        if i in excluded:
            continue
        values = np.random.randint(0, 150, size=num_samples).astype(float)
        # Add NaNs randomly
        nan_indices = np.random.choice(num_samples, size=int(0.05 * num_samples), replace=False)
        values[nan_indices] = np.nan
        quant_data[f'Protein {i}'] = values

    # perfect 20/30 pair for score of 1.0
    prot_20, prot_30 = [], []
    for label in labels:
        base = np.random.randint(30, 120)
        delta = np.random.randint(5, 20)
        if label == 'H':
            prot_20.append(base)
            prot_30.append(base - delta)
        else:
            prot_30.append(base)
            prot_20.append(base - delta)

    prot_20 = np.array(prot_20, dtype=float)
    prot_30 = np.array(prot_30, dtype=float)

    quant_data['Protein 20'] = prot_20
    quant_data['Protein 30'] = prot_30

    # generate pair 10/40 (poor separation)
    prot_10 = np.zeros(num_samples, dtype=float)
    prot_40 = np.zeros(num_samples, dtype=float)

    H_indices = np.where(labels == 'H')[0]
    D_indices = np.where(labels == 'D')[0]

    def assign_balanced_rule(indices):
        #enforces that exactly half of the indices will enforce one rule and the other half the opposite
        #in an even-number of proteins, will give us a score of 0.0
        n = len(indices)
        half = n // 2
        for i, idx in enumerate(indices):
            base = np.random.randint(30, 120)
            delta = np.random.randint(5, 20)
            if i < half:
                prot_10[idx] = base
                prot_40[idx] = base + delta
            else:
                prot_10[idx] = base + delta
                prot_40[idx] = base

    assign_balanced_rule(H_indices)
    assign_balanced_rule(D_indices)

    quant_data['Protein 10'] = prot_10
    quant_data['Protein 40'] = prot_40

    # build df
    quant_df = pd.DataFrame(quant_data)

    # checks
    checker = DataTableChecker()
    checker.check_meta_file(meta_df)
    checker.check_samples(meta_df, quant_df)
    #checker.check_quant_data(quant_df) TODO fix this bc it fails here
    checker.check_duplicate_proteins(quant_df)

    # evaluate
    rule_gen = GenerateRules()
    pairs = rule_gen.generate_rule_pairs(quant_df)

    evaluator = EvaluateRules()

    bool_vectors = evaluator.NEW_Bm_vectorize_all_pairs(pairs, quant_df)
    binarized_labels = evaluator.binarize_labels(meta_df)
    #true_scores = dict(evaluator.NEW_Bm_evaluate_pairs(pairs, bool_vectors, binarized_labels))
    true_scores = dict(evaluator.batch_score_all_pairs_Bm(pairs, bool_vectors, binarized_labels))
    rule_to_buckets = evaluator.get_rule_to_buckets(pairs, bool_vectors)
    #buckets = evaluator.create_null_distributions_for_p_values_testing(pairs, bool_vectors, binarized_labels, rule_to_buckets)

    buckets_to_rule = evaluator.NEW_get_bucket_to_rules(pairs, bool_vectors)
    buckets = evaluator.NEW_Bm_create_null_distributions_for_p_values_testing(pairs, bool_vectors, binarized_labels, buckets_to_rule)

    for keys, values in buckets.items():
        print(f"{keys} → {len(values)} null scores, sample: {values[:5]}")
    print()

    #summary_df = evaluator.summarize_bucket_stats(true_scores, rule_to_buckets, buckets)
    summary_df = evaluator.NEW_summarize_bucket_stats(true_scores, buckets_to_rule, buckets)
    print(summary_df)

    '''
    # assertions
    score_20_30 = perm_results.loc[('Protein 20', 'Protein 30'), 'True_Score']
    if score_20_30 is None:
        raise AssertionError("Expected pair ('Protein 20', 'Protein 30') not found in results.")
    if score_20_30 < 1.0:
        raise AssertionError(f"Expected score 1.0 for ('Protein 20', 'Protein 30'), got {score_20_30}")
    else:
        print("Test passed: ('Protein 20', 'Protein 30') has score near 1.0.")
        print(f"Score for ('Protein 20', 'Protein 30'): {score_20_30}")

    score_10_40 = perm_results.loc[('Protein 10', 'Protein 40'), 'True_Score']
    if score_10_40 is None:
        raise AssertionError("Expected pair ('Protein 10', 'Protein 40') not found in results.")
    if score_10_40 > 0.0:
        raise AssertionError(f"Expected score ~0.0 for ('Protein 10', 'Protein 40'), got {score_10_40}")
    else:
        print("Test passed: ('Protein 10', 'Protein 40') has score near 0.0.")
        print(f"Score for ('Protein 10', 'Protein 40'): {score_10_40}")
    '''

if __name__ == "__main__":
    num_samples = 1000
    num_proteins = 2500
    print(f"Running test with {num_samples} samples and {num_proteins} proteins...")
    start_time = time.time()
    profiler = cProfile.Profile()
    profiler.enable()
    test_newest_method(num_samples=num_samples, num_proteins=num_proteins)
    profiler.disable()
    end_time = time.time()
    total = end_time - start_time

    # print top 10 lines
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(25)
    print(f"Time: {total:.2f} seconds")