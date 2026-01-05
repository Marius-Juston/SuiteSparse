import unittest

from suitesparse_amd.amd import amd

class TestList(unittest.TestCase):
    def test_empty(self):
        matrix = None

        with self.assertRaises(TypeError):
            val = amd(matrix)

    def test_one_d_empty(self):
        matrix = []

        p, v = amd(matrix)

        self.assertEqual(len(p), 0)
        self.assertEqual(v[0], 0)
        self.assertEqual(v[1], 0)

    def test_one_d_full(self):
        matrix = [0]

        with self.assertRaises(TypeError):
            p, v = amd(matrix)

    def test_two_d_empty(self):
        matrix = [[]]

        with self.assertRaises(TypeError):
            p, v = amd(matrix)

    def test_two_d_non_list(self):
        matrix = [0, [0]]

        with self.assertRaises(TypeError):
            amd(matrix)

    def test_two_d_non_square(self):
        matrix = [[0], [0, 1]]

        with self.assertRaises(TypeError):
            amd(matrix)

    def test_two_d_square_identity_dense(self):
        matrix = [[1, 0], [0, 1]]

        p, info = amd(matrix, dense_permutation=True)

        self.assertEqual(matrix, p)

    def test_two_d_square_identity(self):
        matrix = [[1, 0], [0, 1]]

        p, info = amd(matrix, dense_permutation=False)

        self.assertEqual(p, [0, 1])


if __name__ == '__main__':
    unittest.main()
