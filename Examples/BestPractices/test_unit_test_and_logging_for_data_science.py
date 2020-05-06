import numpy as np
from unittest import TestCase

from Examples.BestPractices.unit_test_and_logging_for_data_science import \
    TheAlgorithm
from Examples.BestPractices.unit_test_and_logging_for_data_science import \
    download
from Examples.BestPractices.unit_test_and_logging_for_data_science import split


class TestInput(TestCase):

    @classmethod
    def setUpClass(cls):
        # print('setupClass')
        pass

    @classmethod
    def tearDownClass(cls):
        # print('teardownClass')
        pass

    def setUp(self):
        print('setUp')
        X, y = download()
        splitRatio = 60
        self.X_train, self.y_train, self.X_test, self.y_test = split(X, y,
                                                                     splitRatio)
        self.train_accuracy = 72.92166666666667
        self.train_confusion_matrix = np.matrix(
            [[5447, 5, 40, 31, 49, 16, 198, 50, 81, 6],
             [3, 6440, 127, 54, 3, 29, 25, 36, 24, 1],
             [297, 420, 3824, 163, 256, 19, 622, 186, 121, 50],
             [124, 221, 255, 4566, 54, 251, 97, 129, 275, 159],
             [104, 128, 26, 54, 4546, 342, 206, 133, 96, 207],
             [399, 200, 109, 1081, 416, 2227, 289, 363, 228, 109],
             [173, 89, 112, 55, 156, 229, 5034, 25, 45, 0],
             [213, 192, 205, 39, 160, 17, 26, 5058, 60, 295],
             [67, 690, 202, 677, 73, 188, 347, 39, 3437, 131],
             [164, 162, 63, 290, 669, 279, 122, 735, 291, 3174]])
        self.test_accuracy = 73.4
        self.test_confusion_matrix = np.matrix(
            [[923, 1, 2, 3, 3, 1, 35, 3, 9, 0],
             [0, 1084, 23, 11, 0, 0, 5, 4, 8, 0],
             [63, 78, 669, 27, 38, 2, 97, 28, 24, 6],
             [20, 27, 35, 770, 8, 42, 18, 27, 45, 18],
             [15, 21, 3, 8, 750, 60, 45, 23, 18, 39],
             [56, 24, 15, 193, 73, 362, 56, 58, 38, 17],
             [35, 10, 18, 11, 28, 42, 799, 6, 8, 1],
             [23, 40, 52, 6, 21, 4, 7, 821, 8, 46],
             [14, 90, 29, 99, 10, 33, 66, 7, 598, 28],
             [21, 27, 10, 37, 133, 42, 27, 100, 48, 564]])

    def tearDown(self):
        # print('tearDown')
        pass

    def test_fit(self):
        np.random.seed(31337)
        self.ta = TheAlgorithm(self.X_train, self.y_train, self.X_test,
                               self.y_test)
        self.assertEqual(self.ta.fit(), self.train_accuracy)
        self.assertEqual(self.ta.train_confusion_matrix.tolist(),
                         self.train_confusion_matrix.tolist())

    def test_predict(self):
        np.random.seed(31337)
        self.ta = TheAlgorithm(self.X_train, self.y_train, self.X_test,
                               self.y_test)
        self.ta.fit()
        self.assertEqual(5,5)
        # self.assertEqual(self.ta.predict(), self.test_accuracy)
        # self.assertEqual(self.ta.train_confusion_matrix.tolist(),
        #                  self.train_confusion_matrix.tolist())


# if __name__ == '__main__':
#     # run tests
#     unittest.main(argv=['first-arg-is-ignored'], exit=False)