import unittest
import pandas as pd
from itertools import combinations
import sys
import os

current_dir = os.path.dirname(__file__)

parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from GenerateRules import GenerateRules

class TestGenerateRules(unittest.TestCase):
    def setUp(self):
        self.generator = GenerateRules()

    def test_get_protein_list_valid(self):
        df = pd.DataFrame(columns=['sample_id', 'P1', 'P2', 'P3'])
        expected = ['P1', 'P2', 'P3']
        result = self.generator.get_protein_list(df)
        self.assertEqual(result, expected)

    def test_generate_rule_pairs_typical(self):
        df = pd.DataFrame(columns=['sample_id', 'P1', 'P2', 'P3'])
        expected_rules = [('P1', 'P2'), ('P1', 'P3'), ('P2', 'P3')]
        result = self.generator.generate_rule_pairs(df)
        self.assertEqual(result, expected_rules)

    def test_generate_rule_pairs_minimum_proteins(self):
        df = pd.DataFrame(columns=['sample_id', 'P1'])
        expected_rules = []  # no pairs possible with only one protein
        result = self.generator.generate_rule_pairs(df)
        self.assertEqual(result, expected_rules)

    def test_generate_rule_pairs_empty_df(self):
        df = pd.DataFrame(columns=['sample_id'])  # no proteins
        expected_rules = []
        result = self.generator.generate_rule_pairs(df)
        self.assertEqual(result, expected_rules)

    # TODO make into one function

if __name__ == '__main__':
    unittest.main()