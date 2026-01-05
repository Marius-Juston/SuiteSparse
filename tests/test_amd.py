import unittest
from pathlib import Path
from unittest import skipIf

from suitesparse_amd.amd import amd

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class TestList(unittest.TestCase):
    def test_empty(self):
        matrix = None

        with self.assertRaises(TypeError):
            amd(matrix)

    def test_one_d_empty(self):
        matrix = []

        p, v = amd(matrix)

        self.assertEqual(len(p), 0)
        self.assertEqual(v[0], 0)
        self.assertEqual(v[1], 0)

    def test_one_d_full(self):
        matrix = [0]

        with self.assertRaises(TypeError):
            amd(matrix)

    def test_two_d_empty(self):
        matrix = [[]]

        with self.assertRaises(TypeError):
            amd(matrix)

    def test_two_d_non_list(self):
        matrix = [0, [0]]

        with self.assertRaises(TypeError):
            amd(matrix)

    def test_two_d_non_square(self):
        matrix = [[0], [0, 1]]

        with self.assertRaises(TypeError):
            amd(matrix)

    def test_two_d_square_not_number(self):
        matrix = [[0, 0], [0, 's']]

        p, info = amd(matrix, dense_permutation=False)

        self.assertEqual(p, [0, 1])

    def test_two_d_square_simple_string(self):
        matrix = [['a', 'bc', 'd', 'e'],
                  ['ff', 'gg', '', ''],
                  ['qq', '', 'q', ''],
                  ['f', '', '', 'laugh']]

        with self.assertWarns(RuntimeWarning):
            p, info = amd(matrix, dense_permutation=False)

        self.assertEqual(p, [3, 2, 1, 0])

    def test_two_d_square_identity_dense(self):
        matrix = [[1, 0], [0, 1]]

        p, info = amd(matrix, dense_permutation=True)

        self.assertEqual(matrix, p)

    def test_two_d_square_identity(self):
        matrix = [[1, 0], [0, 1]]

        p, info = amd(matrix, dense_permutation=False)

        self.assertEqual(p, [0, 1])

    def test_two_d_square_simple(self):
        matrix = [[1, 1, 1, 1],
                  [1, 1, 0, 0],
                  [1, 0, 1, 0],
                  [1, 0, 0, 1]]

        p, info = amd(matrix, dense_permutation=False)

        self.assertEqual(p, [3, 2, 1, 0])

    def test_two_d_square_simple_2(self):
        matrix = [[1, 0, 0, 1],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [1, 0, 0, 1]]

        p, info = amd(matrix, dense_permutation=False)

        self.assertEqual(p, [1, 2, 0, 3])

    def test_two_d_square_simple_3(self):
        matrix = [[1, 0, 0, 1],
                  [0, 1, 1, 0],
                  [0, 1, 1, 0],
                  [1, 0, 0, 1]]

        p, info = amd(matrix, dense_permutation=False)

        self.assertEqual(p, [1, 2, 0, 3])

    def test_two_d_square_simple_4(self):
        matrix = [[1, 1, 1, 0],
                  [1, 1, 1, 0],
                  [1, 1, 1, 1],
                  [0, 0, 1, 1]]

        p, info = amd(matrix, dense_permutation=False)

        self.assertEqual(p, [3, 0, 1, 2])

    def test_two_d_square_complex(self):
        """
        Checks if we get the expected output from the matlab example https://www.mathworks.com/help/matlab/ref/amd.html
        :return:
        """

        base_dir = Path(__file__).parent

        with open(base_dir / "P.csv") as f:
            expected_p = list(map(int, f.readline().split(',')))

        n = len(expected_p)

        matrix = [[0 for _ in range(n)] for _ in range(n)]

        with open(base_dir / "A.csv") as f:
            for line in f.readlines():
                data = list(map(int, line.split(',')))

                matrix[data[0]][data[1]] = data[2]

        p, info = amd(matrix, dense_permutation=False)

        self.assertEqual(p, expected_p)


@skipIf(not HAS_NUMPY, "Numpy is not installed")
class TestNumpy(unittest.TestCase):
    def test_one_d_empty(self):
        matrix = np.array([])

        with self.assertRaises(ValueError):
            amd(matrix)

    def test_one_d_full(self):
        matrix = np.array([0])

        with self.assertRaises(ValueError):
            amd(matrix)

    def test_two_d_empty(self):
        matrix = np.array([[]])

        with self.assertRaises(ValueError):
            amd(matrix)

    def test_two_d_square_not_number(self):
        matrix = np.array([[0, 0], [0, 's']])

        p, info = amd(matrix, dense_permutation=False)

        self.assertEqual(p, [0, 1])

    def test_two_d_square_identity_dense(self):
        matrix = np.eye(10)

        p, info = amd(matrix, dense_permutation=True)

        self.assertTrue(np.allclose(matrix, p))

    def test_two_d_square_datatypes(self):
        datatypes = [
            np.bool,
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.intp,
            np.uintp,
            np.int64,
            np.uint64,
            np.float16,
            np.float32,
            np.float64,
            np.float128,
            np.complex64,
            np.complex128,
            np.complex256
        ]

        for dtype in datatypes:
            matrix = np.eye(10, dtype=dtype)

            p, info = amd(matrix, dense_permutation=True)

            self.assertTrue(np.allclose(matrix, p), f"Failed to handle numpy array of type: {dtype}")

    def test_two_d_square_identity(self):
        matrix = np.array([[1, 0], [0, 1]])

        p, info = amd(matrix, dense_permutation=False)

        self.assertEqual(p, [0, 1])

    def test_two_d_square_simple(self):
        matrix = np.array([[1, 1, 1, 1],
                           [1, 1, 0, 0],
                           [1, 0, 1, 0],
                           [1, 0, 0, 1]])

        p, info = amd(matrix, dense_permutation=False)

        self.assertEqual(p, [3, 2, 1, 0])

    def test_two_d_square_simple_2(self):
        matrix = np.array([[1, 0, 0, 1],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [1, 0, 0, 1]])

        p, info = amd(matrix, dense_permutation=False)

        self.assertEqual(p, [1, 2, 0, 3])

    def test_two_d_square_simple_3(self):
        matrix = np.array([[1, 0, 0, 1],
                           [0, 1, 1, 0],
                           [0, 1, 1, 0],
                           [1, 0, 0, 1]])

        p, info = amd(matrix, dense_permutation=False)

        self.assertEqual(p, [1, 2, 0, 3])

    def test_two_d_square_simple_4(self):
        matrix = np.array([[1, 1, 1, 0],
                           [1, 1, 1, 0],
                           [1, 1, 1, 1],
                           [0, 0, 1, 1]])

        p, info = amd(matrix, dense_permutation=False)

        self.assertEqual(p, [3, 0, 1, 2])

    def test_two_d_square_complex(self):
        """
        Checks if we get the expected output from the matlab example https://www.mathworks.com/help/matlab/ref/amd.html
        :return:
        """

        base_dir = Path(__file__).parent

        with open(base_dir / "P.csv") as f:
            expected_p = list(map(int, f.readline().split(',')))

        n = len(expected_p)

        matrix = [[0 for _ in range(n)] for _ in range(n)]

        with open(base_dir / "A.csv") as f:
            for line in f.readlines():
                data = list(map(int, line.split(',')))

                matrix[data[0]][data[1]] = data[2]

        matrix = np.array(matrix)

        p, info = amd(matrix, dense_permutation=False)

        self.assertEqual(p, expected_p)


if __name__ == '__main__':
    unittest.main()
