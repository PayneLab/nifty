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
    #perm_results = evaluator.permutation_test(pairs, quant_df, meta_df, n_permutations=1000)
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

    print(results)

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

    print(results)

def test_large_imbalanced():
    # Large imbalanced data
    num_samples = 500
    num_proteins = 50

    # Create sample_id list
    sample_ids = [f'Sample {i+1}' for i in range(num_samples)]

    # Start with sample_id column
    quant_data = {
        'sample_id': sample_ids
    }

    # Add each protein as a column
    for i in range(1, num_proteins + 1):
        # Random integers from 0 to 150, float type to support NaN
        values = np.random.randint(0, 150, size=num_samples).astype(float)
        # Add ~5% missing values
        nan_indices = np.random.choice(num_samples, size=int(0.05 * num_samples), replace=False)
        values[nan_indices] = np.nan
        quant_data[f'Protein {i}'] = values

    # Create DataFrame
    quant_df = pd.DataFrame(quant_data)


    sample_ids = [f'Sample {i+1}' for i in range(500)]

    # Randomly assign 'H' or 'D' labels
    labels_H = np.random.choice(['H'], size=470)
    labels_D = np.random.choice(['D'], size=30)
    labels = np.concatenate((labels_H, labels_D))
    np.random.shuffle(labels)

    # Create the meta_df
    meta_df = pd.DataFrame({
        'sample_id': sample_ids,
        'classification_label': labels
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

if __name__ == "__main__":
    test_pipeline()
    test_pipeline_NA_quant()
    test_pipeline_NA_meta()
    test_large_imbalanced()

#TODO Ensure output makes sense from the above tests