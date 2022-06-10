import unittest
import utils
import os

class TestFiresMethods(unittest.TestCase):
    # Utils class test
    def test_reading_dataset_csv(self):
        utils.reading_dataset("Data/covtype.csv")

    def test_reading_dataset_arff(self):
        utils.reading_dataset("Data/spambase.arff")

    def test_get_chart(self):
        utils.get_chart("")

    def test_shuffle_data_without_output(self):
        with self.assertRaises(TypeError):
            utils.shuffle_data("Data/spambase.arff")

    def test_shuffle_data(self):
        utils.shuffle_data("../Data/covtype.csv", "covtype_shuffle.csv")
        os.remove("covtype_shuffle.csv")

if __name__ == '__main__':
    unittest.main(exit=False)