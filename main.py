import subprocess
import os
import pandas as pd
import numpy as np
import random

# Globals:
this_dir = os.path.dirname(os.path.abspath(__file__))
moa_jar = os.path.join(this_dir, "moa-pom.jar")
sizeofag_jar = os.path.join(this_dir, "sizeofag-1.0.4.jar")

classifiers_db = {
   "ABFS-NB": "EvaluatePrequential -l (meta.featureselection.FeatureSelectionClassifier -l bayes.NaiveBayes -s (newfeatureselection.BoostingSelector2 -g 100 -t 0.05 -D) -g 0.7 -m) -s (ArffFileStream -f input_file) -d output_file",
   "ABFS-HT": "EvaluatePrequential -l (meta.featureselection.FeatureSelectionClassifier -s (newfeatureselection.BoostingSelector2 -g 100 -t 0.05 -D) -g 0.7 -m) -s (ArffFileStream -f input_file) -d output_file",
   "ABFS-KNN": "EvaluatePrequential -l (meta.featureselection.FeatureSelectionClassifier -l (lazy.kNN -k 500) -s (newfeatureselection.BoostingSelector2 -g 100 -t 0.05 -D) -g 0.7 -m) -s (ArffFileStream -f input_file) -d output_file",
   "ABFS-HAT": "EvaluatePrequential -l (meta.featureselection.FeatureSelectionClassifier -l trees.HoeffdingAdaptiveTree -s (newfeatureselection.BoostingSelector2 -g 100 -t 0.05 -D)) -s (ArffFileStream -f input_file) -d output_file",
   "NB": "EvaluatePrequential -l bayes.NaiveBayes -s (ArffFileStream -f input_file) -d output_file",
   "KNN": "EvaluatePrequential -l (lazy.kNN -k 500) -s (ArffFileStream -f input_file) -d output_file",
   "HT": "EvaluatePrequential -l (trees.HoeffdingTree -g 100) -s (ArffFileStream -f input_file) -d output_file",
   "HAT": "EvaluatePrequential -l trees.HoeffdingAdaptiveTree -s (ArffFileStream -f input_file) -d output_file",
}

data_db = {
    "covtype": os.path.join(this_dir, 'Data', "covtype.arff"),
    "IADS": os.path.join(this_dir, 'Data', "ad-dataset.arff"),
    "Nomao": os.path.join(this_dir, 'Data', "Nomao.arff"),
    "PAMAP2": "",
    "SPAM": os.path.join(this_dir, 'Data', "spambase.arff")
}

result = {}


def run(moa_jar_path, size_of_jar_path, command, input_path, output_path):
    args = command.replace("input_file", input_path).replace("output_file", output_path)

    cmd = f'java -cp {moa_jar_path} -javaagent:{size_of_jar_path}  moa.DoTask {args}'

    print(f"Running the command: {cmd}")
    print(subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read())


def run_simulations(num_of_tests):
    total_correct=0
    for i in range(1, num_of_tests+1):
        for classifier in classifiers_db.keys():
            for data in data_db.keys():
                if data_db[data] == "":
                    continue
                output_path = os.path.join(this_dir, 'Results', f"{classifier}_{data}_{i}.csv")
                run(moa_jar, sizeofag_jar, classifiers_db[classifier], data_db[data], output_path)
                # res_dict = pd.read_csv(output_path)
                # avg_correct = np.mean(res_dict['classifications correct (percent)'])
                # total_correct+=avg_correct


def run_simulations1(num_of_tests):
    if not os.path.exists(os.path.join(this_dir, 'Results')):
        os.makedirs(os.path.join(this_dir, 'Results'))

    if not os.path.exists(os.path.join(this_dir, 'Data', 'Shuffle')):
        os.makedirs(os.path.join(this_dir, 'Data', 'Shuffle'))

    for classifier in classifiers_db.keys():
        for dataset_name in data_db.keys():
            if data_db[dataset_name] == "":
                continue
            else:
                total = 0
                for i in range(1, num_of_tests+1):
                    output_path = os.path.join(this_dir, 'Results', f"{classifier}_{dataset_name}_{i}.csv")

                    # Shuffle data:
                    shuffle_output_path = os.path.join(this_dir, 'Data', 'Shuffle', '{}_{}_{}.arff'.format(dataset_name, classifier, i))
                    data_shuffle(input_path=data_db[dataset_name], output_path=shuffle_output_path)

                    run(moa_jar, sizeofag_jar, classifiers_db[classifier], shuffle_output_path, output_path)
                    df = pd.read_csv(output_path)
                    total += df['classifications correct (percent)'].mean()
            result[classifier + '-' + dataset_name] = total/num_of_tests
            print(result)

    with open(os.path.join(this_dir, 'result.csv'), 'w') as f:
        for key in result.keys():
            f.write("%s, %s\n" % (key, result[key]))


def suffle_dataset(file_name="Data/covtype.csv"):
    df = pd.read_csv(file_name)
    df = df.iloc[np.random.permutation(len(df))]
    output_path = os.path.join(this_dir, 'Data', "covtype1.csv")
    df.to_csv(output_path)


def data_shuffle(input_path, output_path):
    fid = open(input_path, "r")
    li = fid.readlines()
    fid.close()

    data_index = li.index('@data\n')

    meta_data = li[:data_index+7]

    data = li[data_index+7:]
    random.shuffle(data)

    fid = open(output_path, "w")
    fid.writelines(meta_data+data)
    fid.close()


if __name__ == "__main__":
    run_simulations1(30)
    # suffle_dataset()


