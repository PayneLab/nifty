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

    # Verify Against Expected
    expected_results = [(('Protein 1', 'Protein 2'), 0.5), (('Protein 1', 'Protein 3'), 1.0), (('Protein 1', 'Protein 4'), 0.5), (('Protein 1', 'Protein 5'), 0.0), (('Protein 1', 'Protein 6'), 0.0), (('Protein 1', 'Protein 7'), 0.5), (('Protein 2', 'Protein 3'), 0.0), (('Protein 2', 'Protein 4'), 0.5), (('Protein 2', 'Protein 5'), 1.0), (('Protein 2', 'Protein 6'), 1.0), (('Protein 2', 'Protein 7'), 0.5), (('Protein 3', 'Protein 4'), 0.0), (('Protein 3', 'Protein 5'), 0.0), (('Protein 3', 'Protein 6'), 0.0), (('Protein 3', 'Protein 7'), 1.0), (('Protein 4', 'Protein 5'), 0.5), (('Protein 4', 'Protein 6'), 1.0), (('Protein 4', 'Protein 7'), 1.0), (('Protein 5', 'Protein 6'), 1.0), (('Protein 5', 'Protein 7'), 0.5), (('Protein 6', 'Protein 7'), 0.0)]
    assert results == expected_results 

    print("Regression test passed")

if __name__ == "__main__":
    test_pipeline()
