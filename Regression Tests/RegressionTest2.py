# REgression test 2 - Large Dataset with NA values
import copy

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

def add_pair_tsp(quant_data, labels, p_i, p_j,
                 effect=12, jitter=5, violation_rate=0.0,
                 missing=(0.0, 0.0), direction='H_gt',
                 base_low=30, base_high=120):
    """
    Adds two protein columns (p_i, p_j) whose ordering is class-dependent.
      H_gt: H -> p_i>p_j, D -> p_i<p_j  (D_gt flips)
    violation_rate = fraction of samples that violate intended ordering.
    missing = (m_i, m_j) NaN rates per protein.
    Anchor-safe: if p_i already exists, it is reused and not overwritten.
    """
    n = len(labels)

    # If anchor already exists, reuse it; otherwise generate it.
    if p_i in quant_data:
        a = np.asarray(quant_data[p_i], dtype=float).copy()
        # Do NOT apply missingness[0] again — we preserve the existing anchor.
        reuse_anchor = True
    else:
        reuse_anchor = False
        a = np.empty(n, dtype=float)

    b = np.empty(n, dtype=float)

    rng = np.random

    for idx, lab in enumerate(labels):
        want_i_gt = ((direction == 'H_gt' and lab == 'H') or
                     (direction == 'D_gt' and lab == 'D'))

        delta = effect + (rng.randint(0, jitter+1) if jitter > 0 else 0)

        if reuse_anchor:
            # Keep a[idx] fixed; place b relative to existing a[idx]
            ai = a[idx]
            # Intended order: ai > bi if want_i_gt else ai < bi
            bi = ai - delta if want_i_gt else ai + delta

            # Inject violations by flipping which side b goes on
            if rng.rand() < violation_rate:
                bi = ai + delta if want_i_gt else ai - delta

        else:
            # Generate both from scratch
            base  = rng.randint(base_low, base_high)
            ai, bi = (base + delta, base) if want_i_gt else (base, base + delta)

            if rng.rand() < violation_rate:
                ai, bi = bi, ai

            a[idx] = ai

        b[idx] = bi

    # Apply missingness:
    if not reuse_anchor and missing[0] > 0:
        miss = rng.choice(n, int(missing[0]*n), replace=False); a[miss] = np.nan
    if missing[1] > 0:
        miss = rng.choice(n, int(missing[1]*n), replace=False); b[miss] = np.nan

    # Write back
    quant_data[p_i] = a
    quant_data[p_j] = b

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
    buckets = evaluator.NEW_Bm_create_null_distributions_for_p_values_testing(bool_vectors, binarized_labels,
                                                                              buckets_to_rule)
    # So we can compare them.
    buckets_copy = copy.deepcopy(buckets)
    filtered_buckets = evaluator.NEWEST_expand_small_null_distributions(buckets_copy, bool_vectors, binarized_labels,
                                                                        buckets_to_rule)
    print('Before filtering:')
    for bucket_key, values in buckets.items():
        print(f"{bucket_key} → {len(values)} null scores, sample: {values[:5]}")

    print()
    print('After filtering:')
    for bucket_key, values in filtered_buckets.items():
        print(f"{bucket_key} → {len(values)} null scores, sample: {values[:5]}")
        if len(values) < 100:
            print(f'{bucket_key} has less than 100 null scores, skipping.')
    print()

    #summary_df = evaluator.summarize_bucket_stats(true_scores, rule_to_buckets, buckets)
    summary_df = evaluator.NEW_summarize_bucket_stats(true_scores, buckets_to_rule, buckets)

    filtered_df = evaluator.NEW_filter_and_save_rules_BM(summary_df, k=5000)
    print(filtered_df)

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

    # Start new IDs beyond your random range to avoid collisions
    cursor = num_proteins + 1
    def nid():
        nonlocal cursor
        s = f'Protein {cursor}'; cursor += 1; return s

    # Good (not perfect) disjoint rules (~85–90% correct)
    for _ in range(3): #(N+1,N+2), (N+3,N+4), (N+5,N+6)  → e.g., (1001,1002), (1003,1004), (1005,1006)
        p_i, p_j = nid(), nid()
        add_pair_tsp(quant_data, labels, p_i, p_j, effect=12, jitter=5, violation_rate=0.12, direction='H_gt')

    # Meh rules (~60–70% correct)
    for _ in range(2): # IDs minted (N+7,N+8), (N+9,N+10)  → e.g., (1007,1008), (1009,1010)
        p_i, p_j = nid(), nid()
        add_pair_tsp(quant_data, labels, p_i, p_j, effect=10, jitter=8, violation_rate=0.35, direction='H_gt')
    #Assert both pairs are present, but 1007/1008 and 1009/1010 are below (1001,1002), (1003,1004), (1005,1006)

    # Dependent / non-disjoint (share an anchor protein)
    # Pairs: (N+11,N+12) [stronger], (N+11,N+13) [weaker]  → e.g., (1011,1012), (1011,1013)
    p_anchor = nid()
    p_j1, p_j2 = nid(), nid()
    add_pair_tsp(quant_data, labels, p_anchor, p_j1, effect=13, violation_rate=0.08, direction='H_gt')  # stronger
    add_pair_tsp(quant_data, labels, p_anchor, p_j2, effect=11, violation_rate=0.18, direction='H_gt')  # weaker
    #Assert both are present in summary_df, but only 1011/1012 is in filtered_df

    # High/asymmetric missingness examples
    # IDs minted (N+14,N+15), (N+16,N+17)  → e.g., (1014,1015), (1016,1017)
    p_i, p_j = nid(), nid()
    add_pair_tsp(quant_data, labels, p_i, p_j, effect=14, violation_rate=0.10, missing=(0.45, 0.10))
    p_i, p_j = nid(), nid()
    add_pair_tsp(quant_data, labels, p_i, p_j, effect=10, violation_rate=0.20, missing=(0.60, 0.05))


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
    buckets_to_rule = evaluator.NEW_get_bucket_to_rules(pairs, bool_vectors)
    buckets = evaluator.NEW_Bm_create_null_distributions_for_p_values_testing(bool_vectors, binarized_labels, buckets_to_rule)
    filtered_buckets = evaluator.NEWEST_expand_small_null_distributions(buckets, bool_vectors, binarized_labels,
                                                                        buckets_to_rule)

    #summary_df = evaluator.summarize_bucket_stats(true_scores, rule_to_buckets, buckets)
    summary_df = evaluator.NEW_summarize_bucket_stats(true_scores, buckets_to_rule, buckets)
    #print(summary_df.tail(10).to_string(index=False))
    
    filtered_df = evaluator.NEW_filter_and_save_rules_BM(summary_df, k = 5000)
    #print(filtered_df.sort_values(by="True_Score", ascending=False))

    final_df = evaluator.add_mutual_information(filtered_df, bool_vectors, binarized_labels)

    print('Before filtering:')
    for bucket_key, values in filtered_buckets.items():
        print(f"{bucket_key} → {len(values)} null scores, sample: {values[:5]}")

    print()

    print(filtered_df)
    print(final_df)

    print(type(summary_df['Gene_Pair'].iloc[0]))
    print(summary_df['P_Value'].dtype)

    # assertions
    filtered_df = filtered_df.set_index('Gene_Pair')
    score_20_30 = filtered_df.at[('Protein 20','Protein 30'), 'True_Score']

    if score_20_30 is None:
        raise AssertionError("Expected pair ('Protein 20', 'Protein 30') not found in results.")
    if score_20_30 < 1.0:
        raise AssertionError(f"Expected score 1.0 for ('Protein 20', 'Protein 30'), got {score_20_30}")
    else:
        print("Test passed: ('Protein 20', 'Protein 30') has score near 1.0.")
        print(f"Score for ('Protein 20', 'Protein 30'): {score_20_30}")

    score_10_40 = summary_df.loc[
    summary_df['Gene_Pair'].isin([('Protein 10','Protein 40'), ('Protein 40','Protein 10')]),
    'True_Score'].iloc[0]
    if score_10_40 is None:
        raise AssertionError("Expected pair ('Protein 10', 'Protein 40') not found in results.")
    if score_10_40 > 0.0:
        raise AssertionError(f"Expected score ~0.0 for ('Protein 10', 'Protein 40'), got {score_10_40}")
    else:
        print("Test passed: ('Protein 10', 'Protein 40') has score near 0.0.")
        print(f"Score for ('Protein 10', 'Protein 40'): {score_10_40}")

    # --- Assertions for good vs meh synthetic pairs (use summary_df) ---

    # helper to get a score (order-agnostic) or raise if missing
    def _score_or_fail(df, a, b):
        """
        Return True_Score for pair (a,b), trying both orders.
        Works whether Gene_Pair is a column or the (tuple or MultiIndex) index.
        Raises AssertionError if the pair isn't found.
        """
        pair, rev = (a, b), (b, a)

        # Case 1: Gene_Pair is a column
        if 'Gene_Pair' in df.columns:
            mask = df['Gene_Pair'].isin([pair, rev])
            s = df.loc[mask, 'True_Score']
            if s.empty:
                raise AssertionError(f"Expected pair ({a}, {b}) not found.")
            return float(s.iloc[0])

        # Case 2: Gene_Pair is the index (tuple Index or MultiIndex)
        try:
            return float(df.at[pair, 'True_Score'])
        except KeyError:
            try:
                return float(df.at[rev, 'True_Score'])
            except KeyError:
                # If it's a MultiIndex, .at may still KeyError; try .loc
                try:
                    return float(df.loc[pair, 'True_Score'])
                except KeyError:
                    try:
                        return float(df.loc[rev, 'True_Score'])
                    except KeyError:
                        raise AssertionError(f"Expected pair ({a}, {b}) not found.")

    def _in_df_pair(df, pair):
        """
        Order-agnostic membership check for a pair in df (column or index layout).
        """
        a, b = pair
        pair, rev = (a, b), (b, a)
        if 'Gene_Pair' in df.columns:
            return df['Gene_Pair'].isin([pair, rev]).any()
        return (pair in df.index) or (rev in df.index)


    N = num_proteins  # e.g., 1000
    good_pairs = [
        (f'Protein {N+1}', f'Protein {N+2}'),
        (f'Protein {N+3}', f'Protein {N+4}'),
        (f'Protein {N+5}', f'Protein {N+6}'),
    ]
    meh_pairs = [
        (f'Protein {N+7}',  f'Protein {N+8}'),
        (f'Protein {N+9}',  f'Protein {N+10}'),
    ]

    # presence + score retrieval (raises if any missing)
    good_scores = []
    for a, b in good_pairs:
        s = _score_or_fail(filtered_df, a, b)
        good_scores.append(s)
        print(f"Found good pair ({a}, {b}) with True_Score={s:.6f}")

    meh_scores = []
    for a, b in meh_pairs:
        s = _score_or_fail(summary_df, a, b)
        meh_scores.append(s)
        print(f"Found meh pair ({a}, {b}) with True_Score={s:.6f}")

    # ordering: each meh score must be strictly below ALL good scores
    min_good = min(good_scores)
    for (a, b), s in zip(meh_pairs, meh_scores):
        if not (s < min_good):
            raise AssertionError(
                f"Expected meh pair ({a}, {b}) to score below all good pairs "
                f"(min good={min_good:.6f}); got {s:.6f}"
            )
        else:
            print(f"Test passed: meh ({a}, {b}) [{s:.6f}] < min good [{min_good:.6f}]")
    
    # --- Assertions: dependent pairs present in summary_df, but only the strong one in filtered_df ---

    # Compute the expected IDs from num_proteins (N)
    N = num_proteins
    strong_pair = (f'Protein {N+11}', f'Protein {N+12}')  # e.g., (1011, 1012)
    weak_pair   = (f'Protein {N+11}', f'Protein {N+13}')  # e.g., (1011, 1013)

    # 1) Both dependent pairs must exist in the full summary_df
    if not summary_df['Gene_Pair'].isin([strong_pair, strong_pair[::-1]]).any():
        raise AssertionError(f"Expected dependent-strong pair {strong_pair} to be present in summary_df.")
    if not summary_df['Gene_Pair'].isin([weak_pair, weak_pair[::-1]]).any():
        raise AssertionError(f"Expected dependent-weak pair {weak_pair} to be present in summary_df.")
    else:
        print(f"Test passed: both dependent pairs {strong_pair} and {weak_pair} are present in summary_df.")

    # 2) In filtered_df: ONLY the strong pair should be present (disjoint selection should exclude the weak pair)
    if not _in_df_pair(filtered_df, strong_pair):
        raise AssertionError(f"Expected {strong_pair} to appear in filtered_df (stronger dependent pair), but it did not.")
    if _in_df_pair(filtered_df, weak_pair):
        raise AssertionError(f"Expected {weak_pair} to be excluded from filtered_df due to disjoint constraint, but it was included.")
    else:
        print(f"Test passed: {strong_pair} present and {weak_pair} excluded in filtered_df.")


if __name__ == "__main__":
    num_samples = 500
    num_proteins = 1000
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