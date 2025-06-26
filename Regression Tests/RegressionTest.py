import pandas as pd
import numpy as np
import sys
import os

current_dir = os.path.dirname(__file__)

parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from GenerateRules import GenerateRules
from EvaluateRules import EvaluateRules
from DataTableChecker import DataTableChecker

def test_pipeline():
    # Simulated Data
    quant_df = pd.DataFrame({
        'sample_id': ['Sample 1', 'Sample 2', 'Sample 3'],
        'Protein 1': [10, np.nan, 23],
        'Protein 2': [20, 23, 3],
        'Protein 3': [np.nan, 5, 1],
        'Protein 4': [2, 88, 42],
        'Protein 5': [32, 7, 24],
        'Protein 6': [45, 5, 123],
        'Protein 7': [2, 4, 44]
    })

    meta_df = pd.DataFrame({
        'sample_id': ['Sample 1', 'Sample 2', 'Sample 3'],
        'classification_label': ['H', 'D', 'H']
    })

    # Run Checks
    checker = DataTableChecker()
    checker.check_meta_file(meta_df)
    checker.check_samples(meta_df, quant_df)
    checker.check_quant_data(quant_df)
    checker.check_duplicate_proteins(quant_df)

    # Generate Rules
    rule_gen = GenerateRules()
    pairs = rule_gen.generate_rule_pairs(quant_df)

    # Score Rules
    evaluator = EvaluateRules()
    results = evaluator.evaluate_pairs(pairs, quant_df, meta_df)

    print(results)

    # Verify Against Expected
    expected_results = [(('Protein 1', 'Protein 2'), 0.5), (('Protein 1', 'Protein 3'), 1.0), (('Protein 1', 'Protein 4'), 0.5), (('Protein 1', 'Protein 5'), 0.0), (('Protein 1', 'Protein 6'), 0.0), (('Protein 1', 'Protein 7'), 0.5), (('Protein 2', 'Protein 3'), 0.0), (('Protein 2', 'Protein 4'), 0.5), (('Protein 2', 'Protein 5'), 1.0), (('Protein 2', 'Protein 6'), 1.0), (('Protein 2', 'Protein 7'), 0.5), (('Protein 3', 'Protein 4'), 0.0), (('Protein 3', 'Protein 5'), 0.0), (('Protein 3', 'Protein 6'), 0.0), (('Protein 3', 'Protein 7'), 1.0), (('Protein 4', 'Protein 5'), 0.5), (('Protein 4', 'Protein 6'), 1.0), (('Protein 4', 'Protein 7'), 1.0), (('Protein 5', 'Protein 6'), 1.0), (('Protein 5', 'Protein 7'), 0.5), (('Protein 6', 'Protein 7'), 0.0)]
    assert results == expected_results 

    print("Regression test passed")

    # Run Permutation Test
    #perm_results = evaluator.permutate(pairs, quant_df, meta_df, n_permutations=1000)
    #print("Permutation test results:")
    #print(perm_results)

    # TODO Add more regression tests for other dataframes 
    # TODO Add for imbalanced data/strange data/huge data (created in code so I know what to expect)
    # TODO Get creative with many tests


def test_pipeline_NA_quant():
    # All NA values
    quant_df = pd.DataFrame({
        'sample_id': ['Sample 1', 'Sample 2', 'Sample 3'],
        'Protein 1': [np.nan, np.nan, np.nan],
        'Protein 2': [np.nan, np.nan, np.nan],
        'Protein 3': [np.nan, np.nan, np.nan]
    })

    meta_df = pd.DataFrame({
            'sample_id': ['Sample 1', 'Sample 2', 'Sample 3'],
            'classification_label': ['H', 'D', 'H']
    })

    # Run Checks
    checker = DataTableChecker()
    checker.check_meta_file(meta_df)
    checker.check_samples(meta_df, quant_df)
    checker.check_quant_data(quant_df)
    checker.check_duplicate_proteins(quant_df)

    # Generate Rules
    rule_gen = GenerateRules()
    pairs = rule_gen.generate_rule_pairs(quant_df)

    # Score Rules
    evaluator = EvaluateRules()
    results = evaluator.evaluate_pairs(pairs, quant_df, meta_df)

def test_pipeline_NA_meta():
    quant_df = pd.DataFrame({
        'sample_id': ['Sample 1', 'Sample 2', 'Sample 3'],
        'Protein 1': [10, np.nan, 23],
        'Protein 2': [20, 23, 3],
        'Protein 3': [np.nan, 5, 1],
        'Protein 4': [2, 88, 42],
        'Protein 5': [32, 7, 24],
        'Protein 6': [45, 5, 123],
        'Protein 7': [2, 4, 44]
    })

    meta_df = pd.DataFrame({
        'sample_id': ['Sample 1', 'Sample 2', 'Sample 3'],
        'classification_label': [np.nan, np.nan, np.nan]
    })
    # Run Checks
    checker = DataTableChecker()
    checker.check_meta_file(meta_df)
    checker.check_samples(meta_df, quant_df)
    checker.check_quant_data(quant_df)
    checker.check_duplicate_proteins(quant_df)

    # Generate Rules
    rule_gen = GenerateRules()
    pairs = rule_gen.generate_rule_pairs(quant_df)

    # Score Rules
    evaluator = EvaluateRules()
    results = evaluator.evaluate_pairs(pairs, quant_df, meta_df)


def test_large_imbalanced():
    #Make df
    num_samples = 500
    num_proteins = 50

    sample_ids = [f'Sample {i+1}' for i in range(num_samples)]

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

    for i in range(1, num_proteins + 1):
        if i in [10, 20, 30, 40]:
            continue
        values = np.random.randint(0, 150, size=num_samples).astype(float)
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
    quant_data['Protein 20'] = prot_20
    quant_data['Protein 30'] = prot_30

    # Meaningless 10/40 pair for score of 0
    prot_10 = np.zeros(num_samples)
    prot_40 = np.zeros(num_samples)

    H_indices = np.where(labels == 'H')[0]
    D_indices = np.where(labels == 'D')[0]

    def assign_balanced_rule(indices):
        n = len(indices)
        half = n // 2
        for i, idx in enumerate(indices):
            base = np.random.randint(30, 120)
            delta = np.random.randint(5, 20)
            if i < half:
                # 10 < 40
                prot_10[idx] = base
                prot_40[idx] = base + delta
            else:
                # 10 > 40
                prot_10[idx] = base + delta
                prot_40[idx] = base

    assign_balanced_rule(H_indices)
    assign_balanced_rule(D_indices)

    quant_data['Protein 10'] = prot_10
    quant_data['Protein 40'] = prot_40


    quant_df = pd.DataFrame(quant_data)

    # checks
    checker = DataTableChecker()
    checker.check_meta_file(meta_df)
    checker.check_samples(meta_df, quant_df)
    checker.check_quant_data(quant_df)
    checker.check_duplicate_proteins(quant_df)

    rule_gen = GenerateRules()
    pairs = rule_gen.generate_rule_pairs(quant_df)

    evaluator = EvaluateRules()
    results = evaluator.evaluate_pairs(pairs, quant_df, meta_df)

    result_dict = dict(results)

    # check score for perfect separation pair
    score_20_30 = result_dict.get(('Protein 20', 'Protein 30'), None)
    if score_20_30 != 1.0:
        raise AssertionError(f"Expected score 1.0 for ('Protein 20', 'Protein 30'), got {score_20_30}")
    else:
        print("Test passed: ('Protein 20', 'Protein 30') has score 1.0.")

    # check score for meaningless pair
    score_10_40 = result_dict.get(('Protein 10', 'Protein 40'), None)
    if score_10_40 is None:
        raise AssertionError("Expected pair ('Protein 10', 'Protein 40') not found in results.")
    if score_10_40 > 0.05:
        raise AssertionError(f"Expected score ~0.0 for ('Protein 10', 'Protein 40'), got {score_10_40}")
    else:
        print("Test passed: ('Protein 10', 'Protein 40') has score near 0.0.")


if __name__ == "__main__":
    #test_pipeline()
    #test_pipeline_NA_quant()
    #test_pipeline_NA_meta()
    test_large_imbalanced()

