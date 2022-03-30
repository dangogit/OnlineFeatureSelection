import subprocess
import os
import pandas as pd
import numpy as np
import random

from main import this_dir


class ABFS():
    # Globals:
    def __init__(self):
        pass

    this_dir = os.path.dirname(os.path.abspath(__file__))
    moa_jar = os.path.join(this_dir, "moa-pom.jar")
    sizeofag_jar = os.path.join(this_dir, "sizeofag-1.0.4.jar")

    classifiers_db = {
        "Naive Bayes": "EvaluatePrequential -l (meta.featureselection.FeatureSelectionClassifier -l bayes.NaiveBayes -s (newfeatureselection.BoostingSelector2 -g 100 -t 0.05 -D) -g 0.7 -m) -s (ArffFileStream -f input_file) -d output_file",
        "Hoeffding Tree": "EvaluatePrequential -l (meta.featureselection.FeatureSelectionClassifier -s (newfeatureselection.BoostingSelector2 -g 100 -t 0.05 -D) -g 0.7 -m) -s (ArffFileStream -f input_file) -d output_file",
        "KNN": "EvaluatePrequential -l (meta.featureselection.FeatureSelectionClassifier -l (lazy.kNN -k 500 -w 50000) -s (newfeatureselection.BoostingSelector2 -g 100 -t 0.05 -D) -g 0.7 -m) -s (ArffFileStream -f input_file) -d output_file",
        "ABFS-HAT": "EvaluatePrequential -l (meta.featureselection.FeatureSelectionClassifier -l trees.HoeffdingAdaptiveTree -s (newfeatureselection.BoostingSelector2 -g 100 -t 0.05 -D)) -s (ArffFileStream -f input_file) -d output_file",
        "Perceptron Mask (ANN)": "EvaluatePrequential -l (meta.featureselection.FeatureSelectionClassifier -l rules.functions.Perceptron -s (newfeatureselection.BoostingSelector2 -g 100 -t 0.05 -D)) -s (ArffFileStream -f input_file) -d output_file"
    }

    result = {}

    def run(self, moa_jar_path, size_of_jar_path, command, input_path, output_path):
        args = command.replace("input_file", input_path).replace("output_file", output_path)

        cmd = f'java -cp {moa_jar_path} -javaagent:{size_of_jar_path}  moa.DoTask {args}'

        print(f"Running the command: {cmd}")
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        process.wait()
        print(process.stdout.read())

    def parameters(self, classifier_name, classifier_parameters, data, target_index):
        if not os.path.exists(os.path.join(self.this_dir, 'Results')):
            os.makedirs(os.path.join(self.this_dir, 'Results'))

        if not os.path.exists(os.path.join(self.this_dir, 'Data', 'Shuffle')):
            os.makedirs(os.path.join(self.this_dir, 'Data', 'Shuffle'))

        dataset_name = os.path.basename(data).split('.')[0].strip()

        output_path = os.path.join(self.this_dir, 'Results', f"{dataset_name}.csv")

        # Shuffle data:
        shuffle_output_path = os.path.join(self.this_dir, 'Data', 'Shuffle', '{}.arff'.format(dataset_name))

        self.data_shuffle(input_path=data, output_path=shuffle_output_path)

        self.run(self.moa_jar, self.sizeofag_jar, self.classifiers_db[classifier_name], shuffle_output_path, output_path)
        df = pd.read_csv(output_path)
        self.result['avg_acc'] = df['classifications correct (percent)'].mean()
        self.result['evaluation_time'] = df['evaluation time(cpu seconds)'].sum()
        self.result['avg_stab'] = ""

        os.remove(output_path)
        os.remove(shuffle_output_path)
        return self.result

        # with open(os.path.join(self.this_dir, 'result.csv'), 'w') as f:
        #     for key in self.result.keys():
        #         f.write("%s, %s\n" % (key, self.result[key]))

    def data_shuffle(self, input_path, output_path):
        fid = open(os.path.join(this_dir, 'Data', input_path), "r")
        li = fid.readlines()
        fid.close()

        data_index = li.index('@data\n')

        meta_data = li[:data_index + 1]

        data = li[data_index + 1:]
        random.shuffle(data)

        fid = open(output_path, "w")
        fid.writelines(meta_data + data)
        fid.close()
