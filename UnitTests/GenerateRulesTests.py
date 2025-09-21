import unittest
import pandas as pd
import sys
import os

from GenerateRules import GenerateRules

current_dir = os.path.dirname(__file__)

parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

class TestGenerateRules(unittest.TestCase):
    def setUp(self):
        self.generator = GenerateRules()

    def test_generate_rules_return_all_proteins(self):
        # Get protein list.
        try:
            df = pd.DataFrame(columns=['sample_id', 'P1', 'P2', 'P3'])
            expected = ['P1', 'P2', 'P3']
            self.assertEqual(self.generator.get_protein_list(df), expected)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_generate_rules_3_pairs(self):
        # Generate rules with 3 proteins.
        try:
            df = pd.DataFrame(columns=['sample_id', 'P1', 'P2', 'P3'])
            expected_rules = [('P1', 'P2'), ('P1', 'P3'), ('P2', 'P3')]
            result = self.generator.generate_rule_pairs(df)
            self.assertEqual(result, expected_rules)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_generate_rules_1_protein(self):
        # Generate one rule with only one protein.
        try:
            df = pd.DataFrame(columns=['sample_id', 'P1'])
            expected_rules = []  # no pairs possible with only one protein
            result = self.generator.generate_rule_pairs(df)
            self.assertEqual(result, expected_rules)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")

    def test_generate_rules_no_proteins(self):
        # Generate no rules with no proteins.
        try:
            df = pd.DataFrame(columns=['sample_id'])  # no proteins
            expected_rules = []
            result = self.generator.generate_rule_pairs(df)
            self.assertEqual(result, expected_rules)
        except Exception as e:
            self.fail(f"Unexpected exception thrown: {e}")


if __name__ == '__main__':
    unittest.main()
