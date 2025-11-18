import sys
import os
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from FeatureSelector import FeatureSelector


class TestFindFeatures(unittest.TestCase):
    """
    Tests for FeatureSelector.find_features
    """

    def setUp(self):
        # Instance of the class under test
        self.selector = FeatureSelector()

        # Minimal fake filtered_feature_quant_table
        # rows = samples, cols = proteins/features
        self.filtered_feature_quant_table = pd.DataFrame({
            'sample_id': ['s1', 's2', 's3'],
            'P1': [0.1, 0.2, 0.3],
            'P2': [0.4, 0.5, 0.6]
        }).set_index('sample_id')

        # Minimal meta table
        self.feature_meta_table = pd.DataFrame({
            'classification_label': [0, 1, 0]
        }, index=self.filtered_feature_quant_table.index)

        # Base configs dictionary
        self.base_configs = {
            'filtered_feature_quant_table': self.filtered_feature_quant_table.copy(),
            'feature_meta_table': self.feature_meta_table.copy(),
            'seed': 42
        }

    @patch('FeatureSelector.EvaluateRules')
    @patch('FeatureSelector.GenerateRules')
    def test_find_features_basic_flow(self, mock_generate_rules_cls, mock_evaluate_rules_cls):
        """
        Test that find_features:
        - Instantiates GenerateRules and EvaluateRules
        - Calls generate_rule_pairs with the correct quant table
        - Calls run_rule_evaluator with correct arguments
        - Returns rules, true_scores, all_evaluated_rules, top_k_rules from evaluator
        """
        configs = self.base_configs.copy()

        # ----- Set up GenerateRules mock -----
        gr_instance = MagicMock()
        mock_generate_rules_cls.return_value = gr_instance

        # Mock rules returned by generate_rule_pairs
        mock_rules = [('P1', 'P2'), ('P2', 'P3')]
        gr_instance.generate_rule_pairs.return_value = mock_rules

        # ----- Set up EvaluateRules mock -----
        er_instance = MagicMock()
        mock_evaluate_rules_cls.return_value = er_instance

        # Mock outputs from run_rule_evaluator
        mock_true_scores = {'(P1,P2)': 0.9, '(P2,P3)': 0.8}
        mock_all_evaluated_rules = pd.DataFrame({
            'pair': ['(P1,P2)', '(P2,P3)'],
            'score': [0.9, 0.8]
        })
        mock_top_k_rules = pd.DataFrame({
            'pair': ['(P1,P2)'],
            'score': [0.9]
        })

        er_instance.run_rule_evaluator.return_value = (
            mock_true_scores,
            mock_all_evaluated_rules,
            mock_top_k_rules
        )

        # ----- Call the function under test -----
        rules, true_scores, all_evaluated_rules, top_k_rules = self.selector.find_features(configs)

        # ----- Assertions on GenerateRules usage -----
        mock_generate_rules_cls.assert_called_once()
        gr_instance.generate_rule_pairs.assert_called_once()
        # generate_rule_pairs should be called with filtered_feature_quant_table
        gen_args, gen_kwargs = gr_instance.generate_rule_pairs.call_args
        # Either passed positionally or as keyword quant table
        self.assertTrue(
            (gen_args and gen_args[0] is configs['filtered_feature_quant_table']) or
            (gen_kwargs.get('quant_df') is configs['filtered_feature_quant_table'])
        )

        # ----- Assertions on EvaluateRules usage -----
        # EvaluateRules should be instantiated with seed
        mock_evaluate_rules_cls.assert_called_once_with(configs['seed'])

        er_instance.run_rule_evaluator.assert_called_once()
        eval_args, eval_kwargs = er_instance.run_rule_evaluator.call_args

        # Check that configs argument was forwarded correctly
        self.assertEqual(eval_kwargs.get('configs'), configs)

        # Check that pairs argument matches rules from GenerateRules
        self.assertEqual(eval_kwargs.get('pairs'), mock_rules)

        # quant_df and meta_df should match the values in configs
        self.assertIs(eval_kwargs.get('quant_df'), configs['filtered_feature_quant_table'])
        self.assertIs(eval_kwargs.get('meta_df'), configs['feature_meta_table'])

        # ----- Check returned values match mocks -----
        self.assertEqual(rules, mock_rules)
        self.assertIs(true_scores, mock_true_scores)
        self.assertIs(all_evaluated_rules, mock_all_evaluated_rules)
        self.assertIs(top_k_rules, mock_top_k_rules)

    @patch('FeatureSelector.EvaluateRules')
    @patch('FeatureSelector.GenerateRules')
    def test_find_features_with_empty_rules(self, mock_generate_rules_cls, mock_evaluate_rules_cls):
        """
        Edge case: generate_rule_pairs returns no rules.
        We still expect run_rule_evaluator to be called with an empty list.
        """
        configs = self.base_configs.copy()

        # GenerateRules returns empty rules list
        gr_instance = MagicMock()
        mock_generate_rules_cls.return_value = gr_instance
        gr_instance.generate_rule_pairs.return_value = []

        # EvaluateRules mock
        er_instance = MagicMock()
        mock_evaluate_rules_cls.return_value = er_instance

        mock_true_scores = {}
        mock_all_evaluated_rules = pd.DataFrame(columns=['pair', 'score'])
        mock_top_k_rules = pd.DataFrame(columns=['pair', 'score'])

        er_instance.run_rule_evaluator.return_value = (
            mock_true_scores,
            mock_all_evaluated_rules,
            mock_top_k_rules
        )

        rules, true_scores, all_evaluated_rules, top_k_rules = self.selector.find_features(configs)

        # Should still call run_rule_evaluator with an empty list
        er_instance.run_rule_evaluator.assert_called_once()
        eval_kwargs = er_instance.run_rule_evaluator.call_args.kwargs
        self.assertEqual(eval_kwargs.get('pairs'), [])

        self.assertEqual(rules, [])
        self.assertEqual(true_scores, {})
        self.assertTrue(all_evaluated_rules.empty)
        self.assertTrue(top_k_rules.empty)

    @patch('FeatureSelector.EvaluateRules')
    @patch('FeatureSelector.GenerateRules')
    def test_find_features_seed_none(self, mock_generate_rules_cls, mock_evaluate_rules_cls):
        """
        If seed is None, ensure EvaluateRules is called with None (or whatever the constructor expects).
        """
        configs = self.base_configs.copy()
        configs['seed'] = None  # explicit None

        gr_instance = MagicMock()
        mock_generate_rules_cls.return_value = gr_instance
        gr_instance.generate_rule_pairs.return_value = [('P1', 'P2')]

        er_instance = MagicMock()
        mock_evaluate_rules_cls.return_value = er_instance
        er_instance.run_rule_evaluator.return_value = ({}, pd.DataFrame(), pd.DataFrame())

        self.selector.find_features(configs)

        # Ensure EvaluateRules was instantiated with None
        mock_evaluate_rules_cls.assert_called_once_with(None)


if __name__ == "__main__":
    unittest.main()
