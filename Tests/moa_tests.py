import unittest
from ABFS.ABFS import ABFS

class TestMoaMethods(unittest.TestCase):
    # Moa class test
    def test_ABFS(self):
        abfs = ABFS()
        abfs.run_abfs(classifier_name="Naive Bayes", classifier_parameters={},
                            data="Data/spambase.arff", target_index=0, data_shuffle=False,
                            batch_size=50)

    def test_ABFS_without_data(self):
        abfs = ABFS()
        res = abfs.run_abfs(classifier_name="Naive Bayes", classifier_parameters={},
                            data="", target_index=0, data_shuffle=True,
                            batch_size=50)
        self.assertEqual({}, res)

if __name__ == '__main__':
    unittest.main(exit=False)