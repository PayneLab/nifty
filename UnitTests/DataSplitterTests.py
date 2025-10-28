import sys
import os

import unittest
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
        
        # balanced classes - H:5, D:5
        self.reference_quant_balanced = pd.DataFrame(reference_quant)

        reference_meta_balanced = {"sample_id": ['samp1', 'samp2', 'samp3', 'samp4', 'samp5', 'samp6', 'samp7', 'samp8', 'samp9', 'samp10'], 
                          "classification_label": ['H', 'D', 'H', 'H', 'H', 'D', 'D', 'H', 'D', 'D']}
        self.reference_meta_balanced = pd.DataFrame(reference_meta_balanced)

        self.reference_quant_balanced.set_index('sample_id', inplace=True)
        self.reference_meta_balanced.set_index('sample_id', inplace=True)

        # imbalanced classes, H:2, D:8
        self.reference_quant_imbalanced = pd.DataFrame(reference_quant)

        reference_meta_imbalanced = {"sample_id": ['samp1', 'samp2', 'samp3', 'samp4', 'samp5', 'samp6', 'samp7', 'samp8', 'samp9', 'samp10'], 
                          "classification_label": ['D', 'D', 'H', 'D', 'H', 'D', 'D', 'D', 'D', 'D']}
        self.reference_meta_imbalanced = pd.DataFrame(reference_meta_imbalanced)

        self.reference_quant_imbalanced.set_index('sample_id', inplace=True)
        self.reference_meta_imbalanced.set_index('sample_id', inplace=True)

    # test balanced classes
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

        # quant2 and meta2 should have 3 rows
        self.assertAlmostEqual(len(quant2), 3, delta = 1)
        self.assertAlmostEqual(len(meta2), 3, delta = 1)

        # quant2 and meta2 should have the same 3 sample IDs in the same order
        self.assertEqual(quant2.index.tolist(), meta2.index.tolist())

        # quant3 and meta3 should have 4 rows
        self.assertAlmostEqual(len(quant3), 4, delta = 1)
        self.assertAlmostEqual(len(meta3), 4, delta = 1)

        # quant3 and meta3 should have the same 4 sample IDs in the same order
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

        # meta2 should have a near-even split of the classes
        num_rows_H = len(meta2[meta2['classification_label'] == 'H'])
        num_rows_D = len(meta2[meta2['classification_label'] == 'D'])
        self.assertAlmostEqual(num_rows_H, num_rows_D, delta = 1)

        # meta3 should have a near-even split of the classes
        num_rows_H = len(meta3[meta3['classification_label'] == 'H'])
        num_rows_D = len(meta3[meta3['classification_label'] == 'D'])
        self.assertAlmostEqual(num_rows_H, num_rows_D, delta = 1)

    # TODO: add more tests for imbalanced classes?
    def test_2_proportions_imbalanced_classes_05_05_split(self):
        quant1, meta1, quant2, meta2 = self.splitter.split_table(self.reference_quant_imbalanced, self.reference_meta_imbalanced, (0.5, 0.5), None)

        # quant1 and meta1 should have 5 rows
        self.assertEqual(len(quant1), 5)
        self.assertEqual(len(meta1), 5)

        # quant1 and meta1 should have the same 5 sample IDs in the same order
        self.assertEqual(quant1.index.tolist(), meta1.index.tolist())

        # quant2 and meta2 should have 5 rows
        self.assertEqual(len(quant2), 5)
        self.assertEqual(len(meta2), 5)

        # quant2 and meta2 should have the same 5 sample IDs in the same order
        self.assertEqual(quant2.index.tolist(), meta2.index.tolist())

        # quant/meta1 and quant/meta2 should have no sample overlap
        index1 = set(quant1.index.tolist())
        index2 = set(quant2.index.tolist())
        intersection = index1.intersection(index2)
        self.assertEqual(len(intersection), 0)

        # meta1 should have a 1:4 ratio between H:D
        num_rows_H = len(meta1[meta1['classification_label'] == 'H'])
        num_rows_D = len(meta1[meta1['classification_label'] == 'D'])
        self.assertAlmostEqual((num_rows_H / num_rows_D), (2 / 8), places=2)

        # meta2 should have a 1:4 ratio between H:D
        num_rows_H = len(meta2[meta2['classification_label'] == 'H'])
        num_rows_D = len(meta2[meta2['classification_label'] == 'D'])
        self.assertAlmostEqual((num_rows_H / num_rows_D), (2 / 8), places=2)

    # test bad input values
    def test_1_proportion(self):
        with self.assertRaises(SystemExit) as e:
            self.splitter.split_table(self.reference_quant_balanced, self.reference_meta_balanced, (1.0,), None)

        self.assertEqual(e.exception.code, 1)

    def test_4_proportions(self):
        with self.assertRaises(SystemExit) as e:
            self.splitter.split_table(self.reference_quant_balanced, self.reference_meta_balanced, (0.25, 0.25, 0.25, 0.25), None)

        self.assertEqual(e.exception.code, 1)

    # TODO: test fixed seed
    def test_random_seed(self):
        quant1, meta1, quant2, meta2 = self.splitter.split_table(self.reference_quant_balanced, self.reference_meta_balanced, (0.3, 0.7), None)
        quant3, meta3, quant4, meta4 = self.splitter.split_table(self.reference_quant_balanced, self.reference_meta_balanced, (0.3, 0.7), None)
        
        indexes1 = quant1.index.tolist()
        indexes2 = quant2.index.tolist()
        indexes3 = quant3.index.tolist()
        indexes4 = quant4.index.tolist()

        self.assertNotEqual(indexes1, indexes3)
        self.assertNotEqual(indexes2, indexes4)

    def test_seed_42(self):
        # 42, 42 should be exactly the same
        quant1, meta1, quant2, meta2 = self.splitter.split_table(self.reference_quant_balanced, self.reference_meta_balanced, (0.8, 0.2), 42)
        quant3, meta3, quant4, meta4 = self.splitter.split_table(self.reference_quant_balanced, self.reference_meta_balanced, (0.8, 0.2), 42)
        
        indexes1 = quant1.index.tolist()
        indexes2 = quant2.index.tolist()
        indexes3 = quant3.index.tolist()
        indexes4 = quant4.index.tolist()

        self.assertEqual(indexes1, indexes3)
        self.assertEqual(indexes2, indexes4)

    def test_seed_100(self):
        # 100, 100 should be exactly the same
        quant1, meta1, quant2, meta2 = self.splitter.split_table(self.reference_quant_balanced, self.reference_meta_balanced, (0.4, 0.6), 100)
        quant3, meta3, quant4, meta4 = self.splitter.split_table(self.reference_quant_balanced, self.reference_meta_balanced, (0.4, 0.6), 100)
        
        indexes1 = quant1.index.tolist()
        indexes2 = quant2.index.tolist()
        indexes3 = quant3.index.tolist()
        indexes4 = quant4.index.tolist()

        self.assertEqual(indexes1, indexes3)
        self.assertEqual(indexes2, indexes4)

    def test_fixed_seeds(self):
        # 50, 50 should be exactly the same
        # 75, 75 should be exactly the same
        # 50, 75 should be different
        quant1, meta1, quant2, meta2 = self.splitter.split_table(self.reference_quant_balanced, self.reference_meta_balanced, (0.2, 0.8), 50)
        quant3, meta3, quant4, meta4 = self.splitter.split_table(self.reference_quant_balanced, self.reference_meta_balanced, (0.2, 0.8), 50)
        
        indexes1 = quant1.index.tolist()
        indexes2 = quant2.index.tolist()
        indexes3 = quant3.index.tolist()
        indexes4 = quant4.index.tolist()

        self.assertEqual(indexes1, indexes3)
        self.assertEqual(indexes2, indexes4)


        quant5, meta5, quant6, meta6 = self.splitter.split_table(self.reference_quant_balanced, self.reference_meta_balanced, (0.2, 0.8), 75)
        quant7, meta7, quant8, meta8 = self.splitter.split_table(self.reference_quant_balanced, self.reference_meta_balanced, (0.2, 0.8), 75)
        
        indexes5 = quant5.index.tolist()
        indexes6 = quant6.index.tolist()
        indexes7 = quant7.index.tolist()
        indexes8 = quant8.index.tolist()

        self.assertEqual(indexes5, indexes7)
        self.assertEqual(indexes6, indexes8)

        self.assertNotEqual(indexes1, indexes5)
        self.assertNotEqual(indexes2, indexes6)
        self.assertNotEqual(indexes3, indexes7)
        self.assertNotEqual(indexes4, indexes8)


if __name__ == "__main__":
    unittest.main()