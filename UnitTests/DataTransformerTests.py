




def test_vectorize_pair_no_na(self):
        df = pd.DataFrame({
            'P1': [1, 4, 6, 3, 1, 7, 1, 7],
            'P2': [2, 3, 6, 2, 6, 1, 2, 9]
        })
        expected = np.array([False, True, False, True, False, True, False, False])
        result = self.evaluator.vectorize_pair(['P1', 'P2'], df)
        np.testing.assert_array_equal(result, expected)

    def test_vectorize_pair_na_protein_1(self):
        df = pd.DataFrame({
            'P1': [np.nan, 4, 6, np.nan],
            'P2': [2, 3, 6, 1]
        })
        expected = np.array([False, True, False, False])
        result = self.evaluator.vectorize_pair(['P1', 'P2'], df)
        np.testing.assert_array_equal(result, expected)

    def test_vectorize_pair_na_protein_2(self):
        df = pd.DataFrame({
            'P1': [1, 4, 6],
            'P2': [2, np.nan, 6]
        })
        expected = np.array([False, True, False])
        result = self.evaluator.vectorize_pair(['P1', 'P2'], df)
        np.testing.assert_array_equal(result, expected)

    def test_vectorize_pair_na_in_both(self):
        df = pd.DataFrame({
            'P1': [0, 4, 6, 3, np.nan, 7, np.nan, 7],
            'P2': [2, 3, 6, np.nan, 6, np.nan, np.nan, 9]
        })
        expected = np.array([False, True, False, True, False, True, False, False])
        result = self.evaluator.vectorize_pair(['P1', 'P2'], df)
        np.testing.assert_array_equal(result, expected)

    def test_vectorize_pair_same_proteins(self):
        df = pd.DataFrame({'P1': [1, 2, 3, 4]})
        expected = np.array([False, False, False, False])
        result = self.evaluator.vectorize_pair(['P1', 'P1'], df)
        np.testing.assert_array_equal(result, expected)