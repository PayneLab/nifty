import sys
import os

import unittest
from tempfile import TemporaryDirectory
import pandas as pd
import numpy as np

current_dir = os.path.dirname(__file__)

parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from DataSplitter import DataSplitter

# Test split_table()
class TestSplitTable(unittest.TestCase):

    def setUp(self):
        self.splitter = DataSplitter()

        reference_quant = {"sample_id": ['samp1', 'samp2', 'samp3', 'samp4', 'samp5', 'samp6', 'samp7', 'samp8', 'samp9', 'samp10'], 
                           "Protein1": [100, 0, np.nan, 50, 300, 1, np.nan, 1000, 77, 5], 
                           "Protein2": [50, 800, 25, 1000, 80, 75, 60, 400, 500, 900]}
        self.reference_quant = pd.DataFrame(reference_quant)

        reference_meta = {"sample_id": ['samp1', 'samp2', 'samp3', 'samp4', 'samp5', 'samp6', 'samp7', 'samp8', 'samp9', 'samp10'], 
                          "classification_label": ['H', 'D', 'H', 'H', 'H', 'D', 'D', 'H', 'D', 'D']}
        self.reference_meta = pd.DataFrame(reference_meta)

        self.reference_quant.set_index('sample_id', inplace=True)
        self.reference_meta.set_index('sample_id', inplace=True)

    def test_2_proportions_balanced_classes_07_03_split(self):
        quant1, meta1, quant2, meta2 = self.splitter.split_table(self.reference_quant, self.reference_meta, (0.7, 0.3), None)

        # quant1 and meta1 should have 7 rows
        self.assertEqual(len(quant1), 7)
        self.assertEqual(len(meta1), 7)

        # quant1 and meta1 should have the same 7 sample IDs in the same order
        self.assertEqual(quant1.index.tolist(), meta1.index.tolist())

        # quant2 and meta2 should have 3 rows
        self.assertEqual(len(quant2), 3)
        self.assertEqual(len(meta2), 3)

        # quant2 and meta2 should have the same 3 sample IDs in the same order
        self.assertEqual(quant2.index.tolist(), meta2.index.tolist())

        # quant/meta1 and quant/meta2 should have no sample overlap
        index1 = set(quant1.index.tolist())
        index2 = set(quant2.index.tolist())
        intersection = index1.intersection(index2)
        self.assertEqual(len(intersection), 0)

        # meta1 should have a near-even split of the classes
        num_rows_H = len(meta1[meta1['classification_label'] == 'H'])
        num_rows_D = len(meta1[meta1['classification_label'] == 'D'])
        self.assertAlmostEqual(num_rows_H, num_rows_D, delta=1)

        # meta2 should have a near-even split of the classes
        num_rows_H = len(meta2[meta2['classification_label'] == 'H'])
        num_rows_D = len(meta2[meta2['classification_label'] == 'D'])
        self.assertAlmostEqual(num_rows_H, num_rows_D, delta=1)

    def test_2_proportions_balanced_classes_06_04_split(self):
        quant1, meta1, quant2, meta2 = self.splitter.split_table(self.reference_quant, self.reference_meta, (0.6, 0.4), None)

        # quant1 and meta1 should have 7 rows
        self.assertEqual(len(quant1), 6)
        self.assertEqual(len(meta1), 6)

        # quant1 and meta1 should have the same 7 sample IDs in the same order
        self.assertEqual(quant1.index.tolist(), meta1.index.tolist())

        # quant2 and meta2 should have 3 rows
        self.assertEqual(len(quant2), 4)
        self.assertEqual(len(meta2), 4)

        # quant2 and meta2 should have the same 3 sample IDs in the same order
        self.assertEqual(quant2.index.tolist(), meta2.index.tolist())

        # quant/meta1 and quant/meta2 should have no sample overlap
        index1 = set(quant1.index.tolist())
        index2 = set(quant2.index.tolist())
        intersection = index1.intersection(index2)
        self.assertEqual(len(intersection), 0)

        # meta1 should have an even split of the classes
        num_rows_H = len(meta1[meta1['classification_label'] == 'H'])
        num_rows_D = len(meta1[meta1['classification_label'] == 'D'])
        self.assertAlmostEqual(num_rows_H, num_rows_D, delta=1)

        # meta2 should have an even split of the classes
        num_rows_H = len(meta2[meta2['classification_label'] == 'H'])
        num_rows_D = len(meta2[meta2['classification_label'] == 'D'])
        self.assertEqual(num_rows_H, num_rows_D)
        
    def test_3_proportions_balanced_classes(self):
        pass

    def test_1_proportion_balanced_classes(self):
        pass

    def test_4_proportions_balanced_classes(self):
        pass

    def test_2_proportions_imbalanced_classes(self):
        pass

    def test_3_proportions_imbalanced_classes(self):
        pass

    def test_1_proportion_imbalanced_classes(self):
        pass

    def test_4_proportions_imbalanced_classes(self):
        pass


if __name__ == "__main__":
    unittest.main()