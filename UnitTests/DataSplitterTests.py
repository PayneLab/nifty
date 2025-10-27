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
        self.reference_quant_balanced = pd.DataFrame(reference_quant)

        reference_meta = {"sample_id": ['samp1', 'samp2', 'samp3', 'samp4', 'samp5', 'samp6', 'samp7', 'samp8', 'samp9', 'samp10'], 
                          "classification_label": ['H', 'D', 'H', 'H', 'H', 'D', 'D', 'H', 'D', 'D']}
        self.reference_meta_balanced = pd.DataFrame(reference_meta)

        self.reference_quant_balanced.set_index('sample_id', inplace=True)
        self.reference_meta_balanced.set_index('sample_id', inplace=True)

    def test_2_proportions_balanced_classes_07_03_split(self):
        quant1, meta1, quant2, meta2 = self.splitter.split_table(self.reference_quant_balanced, self.reference_meta_balanced, (0.7, 0.3), None)

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
        quant1, meta1, quant2, meta2 = self.splitter.split_table(self.reference_quant_balanced, self.reference_meta_balanced, (0.6, 0.4), None)

        # quant1 and meta1 should have 6 rows
        self.assertEqual(len(quant1), 6)
        self.assertEqual(len(meta1), 6)

        # quant1 and meta1 should have the same 6 sample IDs in the same order
        self.assertEqual(quant1.index.tolist(), meta1.index.tolist())

        # quant2 and meta2 should have 4 rows
        self.assertEqual(len(quant2), 4)
        self.assertEqual(len(meta2), 4)

        # quant2 and meta2 should have the same 4 sample IDs in the same order
        self.assertEqual(quant2.index.tolist(), meta2.index.tolist())

        # quant/meta1 and quant/meta2 should have no sample overlap
        index1 = set(quant1.index.tolist())
        index2 = set(quant2.index.tolist())
        intersection = index1.intersection(index2)
        self.assertEqual(len(intersection), 0)

        # meta1 should have an even split of the classes
        num_rows_H = len(meta1[meta1['classification_label'] == 'H'])
        num_rows_D = len(meta1[meta1['classification_label'] == 'D'])
        self.assertEqual(num_rows_H, num_rows_D)

        # meta2 should have an even split of the classes
        num_rows_H = len(meta2[meta2['classification_label'] == 'H'])
        num_rows_D = len(meta2[meta2['classification_label'] == 'D'])
        self.assertEqual(num_rows_H, num_rows_D)
        
    def test_3_proportions_balanced_classes_02_06_02_split(self):
        quant1, meta1, quant2, meta2, quant3, meta3 = self.splitter.split_table(self.reference_quant_balanced, self.reference_meta_balanced, (0.2, 0.6, 0.2), None)

        # quant1 and meta1 should have 2 rows
        self.assertEqual(len(quant1), 2)
        self.assertEqual(len(meta1), 2)

        # quant1 and meta1 should have the same 2 sample IDs in the same order
        self.assertEqual(quant1.index.tolist(), meta1.index.tolist())

        # quant2 and meta2 should have 6 rows
        self.assertEqual(len(quant2), 6)
        self.assertEqual(len(meta2), 6)

        # quant2 and meta2 should have the same 6 sample IDs in the same order
        self.assertEqual(quant2.index.tolist(), meta2.index.tolist())

        # quant3 and meta3 should have 2 rows
        self.assertEqual(len(quant3), 2)
        self.assertEqual(len(meta3), 2)

        # quant3 and meta3 should have the same 2 sample IDs in the same order
        self.assertEqual(quant3.index.tolist(), meta3.index.tolist())

        # quant/meta1, quant/meta2, quant/meta3 should have no sample overlap
        index1 = set(quant1.index.tolist())
        index2 = set(quant2.index.tolist())
        index3 = set(quant3.index.tolist())
        intersection = (index1 & index2 & index3)
        self.assertEqual(len(intersection), 0)

        # meta1 should have an even split of the classes
        num_rows_H = len(meta1[meta1['classification_label'] == 'H'])
        num_rows_D = len(meta1[meta1['classification_label'] == 'D'])
        self.assertEqual(num_rows_H, num_rows_D)

        # meta2 should have an even split of the classes
        num_rows_H = len(meta2[meta2['classification_label'] == 'H'])
        num_rows_D = len(meta2[meta2['classification_label'] == 'D'])
        self.assertEqual(num_rows_H, num_rows_D)

        # meta3 should have an even split of the classes
        num_rows_H = len(meta3[meta3['classification_label'] == 'H'])
        num_rows_D = len(meta3[meta3['classification_label'] == 'D'])
        self.assertEqual(num_rows_H, num_rows_D)

    def test_3_proportions_balanced_classes_03_03_04_split(self):
        quant1, meta1, quant2, meta2, quant3, meta3 = self.splitter.split_table(self.reference_quant_balanced, self.reference_meta_balanced, (0.3, 0.3, 0.4), None)

        # quant1 and meta1 should have 3 rows
        self.assertEqual(len(quant1), 3)
        self.assertEqual(len(meta1), 3)

        # quant1 and meta1 should have the same 3 sample IDs in the same order
        self.assertEqual(quant1.index.tolist(), meta1.index.tolist())

        # quant2 and meta2 should have 2 rows
        self.assertEqual(len(quant2), 2)
        self.assertEqual(len(meta2), 2)

        # quant2 and meta2 should have the same 2 sample IDs in the same order
        self.assertEqual(quant2.index.tolist(), meta2.index.tolist())

        # quant3 and meta3 should have 5 rows
        self.assertEqual(len(quant3), 5)
        self.assertEqual(len(meta3), 5)

        # quant3 and meta3 should have the same 5 sample IDs in the same order
        self.assertEqual(quant3.index.tolist(), meta3.index.tolist())

        # quant/meta1, quant/meta2, quant/meta3 should have no sample overlap
        index1 = set(quant1.index.tolist())
        index2 = set(quant2.index.tolist())
        index3 = set(quant3.index.tolist())
        intersection = (index1 & index2 & index3)
        self.assertEqual(len(intersection), 0)

        # meta1 should have a near-even split of the classes
        num_rows_H = len(meta1[meta1['classification_label'] == 'H'])
        num_rows_D = len(meta1[meta1['classification_label'] == 'D'])
        self.assertAlmostEqual(num_rows_H, num_rows_D, delta = 1)

        # meta2 should have an even split of the classes
        num_rows_H = len(meta2[meta2['classification_label'] == 'H'])
        num_rows_D = len(meta2[meta2['classification_label'] == 'D'])
        self.assertEqual(num_rows_H, num_rows_D)

        # meta3 should have a near-even split of the classes
        num_rows_H = len(meta3[meta3['classification_label'] == 'H'])
        num_rows_D = len(meta3[meta3['classification_label'] == 'D'])
        self.assertAlmostEqual(num_rows_H, num_rows_D, delta = 1)

    # TODO: add tests for imbalanced classes?

    def test_1_proportion(self):
        with self.assertRaises(SystemExit) as e:
            self.splitter.split_table(self.reference_quant_balanced, self.reference_meta_balanced, (1.0,), None)

        self.assertEqual(e.exception.code, 1)

    def test_4_proportions(self):
        with self.assertRaises(SystemExit) as e:
            self.splitter.split_table(self.reference_quant_balanced, self.reference_meta_balanced, (0.25, 0.25, 0.25, 0.25), None)

        self.assertEqual(e.exception.code, 1)


if __name__ == "__main__":
    unittest.main()