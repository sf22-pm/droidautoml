from unittest import TestCase, main

from numpy import array, nan, isnan, ndarray
from pandas import DataFrame

from droidautoml.colors import ConsoleColors
from droidautoml.preprocessors import DataPreprocessor


class PreprocessorsTestCase(TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    print(f'{ConsoleColors.BOLD}{ConsoleColors.OKCYAN}Estimators Test Suite{ConsoleColors.END_COLORS}')

  def setUp(self) -> None:
    self.raw_matrix = array([[1, 2, 3], [4, nan, 6], [7, 8, 9], [7, 8, 9]])
    self.data_preprocessor = DataPreprocessor()
    self.processed_matrix = self.data_preprocessor.run(self.raw_matrix)

  def test_remove_null_values(self):
    print('Should return a matrix without duplicates')
    self.assertFalse(isnan(self.processed_matrix).any())

  def test_convert_to_nparray(self):
    print('Should return a matrix with ndarray data type')
    self.assertTrue(isinstance(self.processed_matrix, ndarray))

  def test_remove_duplicates(self):
    print('Should remove duplicated rows')
    df_matrix = DataFrame(self.processed_matrix)
    self.assertFalse(all(df_matrix.duplicated()))


if __name__ == '__main__':
    main()
