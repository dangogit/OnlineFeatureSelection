import os
import pandas as pd
from utils import data_shuffle, run_moa_subprocess


class ABFS():
    # Globals:
    def __init__(self):
        """
        ABFS class constructor
        """
        self.result = {}

    def run_abfs(self, classifier_name, classifier_parameters, data, batch_size, target_index='-1', shuffle=False):
        """
        This method generates ABFS paths and runs moa with selected parameters
        :param classifier_name: name of the classifier ["Naive Bayes", "Hoeffding Tree", "KNN", "Perceptron Mask (ANN)"]
        :param classifier_parameters: specific parameters for the classifier
        :param data: the data path 
        :param batch_size: size of the batch
        :param target_index: the index of the class column
        :param shuffle: True/False if data shuffle is needed
        :return: (dict) {'avg_acc': "", 'avg_stab': "", 'num_of_features': ""}
        """
        if not os.path.exists(os.path.join(self.this_dir, 'Results')):
            os.makedirs(os.path.join(self.this_dir, 'Results'))

        dataset_name = os.path.basename(data).split('.')[0].strip()
        output_path = os.path.join(self.this_dir, 'Results', f"{dataset_name}.csv")

        try:
            if shuffle:
                if not os.path.exists(os.path.join(self.this_dir, 'Data', 'Shuffle')):
                    os.makedirs(os.path.join(self.this_dir, 'Data', 'Shuffle'))
                input_path = os.path.join(self.this_dir, 'Data', 'Shuffle', '{}.arff'.format(dataset_name))
                data_shuffle(input_path=data, output_path=input_path)
            else:
                input_path = data

            run_moa_subprocess(classifier=classifier_name, input_path=input_path, output_path=output_path, target_index=target_index, batch_size=batch_size, classifier_parameters=classifier_parameters)
            if os.path.exists(output_path):
                df = pd.read_csv(output_path)
                self.result['avg_acc'] = df['classifications correct (percent)'].mean()
                self.result['evaluation_time'] = df['evaluation time (cpu seconds)'].sum()
                self.result['num_of_features'] = df['# of features selected'].max()
            else:
                print(f"Failed to run {classifier_name}_{dataset_name}_{batch_size}")

        except Exception as e:
            print(e)

        finally:
            if os.path.isfile(output_path):
                os.remove(output_path)
            if os.path.isfile(input_path):
                os.remove(input_path)

        return self.result
