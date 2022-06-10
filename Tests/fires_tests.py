import unittest
from Fires import FIRES

class TestFiresMethods(unittest.TestCase):
    # Fires class test
    def test_apply_fires_without_data(self):
        with self.assertRaises(ValueError):
            FIRES.apply_fires(classifier_name="Naive Bayes", classifier_parameters={},
                              data="", target_index=0, batch_size=100)

    def test_apply_fires(self):
        with self.assertRaises(FileNotFoundError):
            FIRES.apply_fires(classifier_name="Naive Bayes", classifier_parameters={},
                              data="Data/Fires/covtype_5_vs_all.csv", target_index=0, batch_size=100)

if __name__ == '__main__':
    unittest.main(exit=False)