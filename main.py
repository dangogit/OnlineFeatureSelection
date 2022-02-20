import subprocess
import os
import pandas as pd
import numpy as np

this_dir = os.path.dirname(os.path.abspath(__file__))

moa_jar = os.path.join(this_dir, "moa-pom.jar")

sizeofag_jar = os.path.join(this_dir, "sizeofag-1.0.4.jar")

classifiers_db = {
   "ABFS-NB": "EvaluatePrequential -l (meta.featureselection.FeatureSelectionClassifier -l bayes.NaiveBayes -s (newfeatureselection.BoostingSelector2 -g 100 -t 0.05 -D) -g 0.7 -m) -s (ArffFileStream -f input_file) -d output_file",
   # "ABFS-HT": "EvaluatePrequential -l (meta.featureselection.FeatureSelectionClassifier -s (newfeatureselection.BoostingSelector2 -g 100 -t 0.05 -D) -g 0.7 -m) -s (ArffFileStream -f input_file) -d output_file",
   # "ABFS-KNN": "EvaluatePrequential -l (meta.featureselection.FeatureSelectionClassifier -l (lazy.kNN -k 500) -s (newfeatureselection.BoostingSelector2 -g 100 -t 0.05 -D) -g 0.7 -m) -s (ArffFileStream -f input_file) -d output_file",
   # "NB": "EvaluatePrequential -l bayes.NaiveBayes -s (ArffFileStream -f input_file) -d output_file",
   # "KNN": "EvaluatePrequential -l (lazy.kNN -k 500) -s (ArffFileStream -f input_file) -d output_file",
   # "HT": "EvaluatePrequential -l (trees.HoeffdingTree -g 100) -s (ArffFileStream -f input_file) -d output_file"
    }

data_db = {
   "COVTYPE": os.path.join(this_dir, 'Data', "covtype.arff"),
   "IADS": "",
   "NOMAO": "",
   "PAMAP2": "",
   "SPAM": ""}

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
                res_dict = pd.read_csv(output_path)
                avg_correct = np.mean(res_dict['classifications correct (percent)'])
                total_correct+=avg_correct



def run_simulations1(num_of_tests):
    for classifier in classifiers_db.keys(): #ABFS-NB
        for data in data_db.keys(): #COVTYPE
            if data_db[data] == "":
                continue
            else:
                total = 0
                for i in range(1, num_of_tests+1):
                    output_path = os.path.join(this_dir, 'Results', f"{classifier}_{data}_{i}.csv")
                    run(moa_jar, sizeofag_jar, classifiers_db[classifier], data_db[data], output_path)
                    df = pd.read_csv(output_path)
                    total += df['classifications correct (percent)'].mean()
            result[classifier + '-' + data] = total/num_of_tests
    print(result)

run_simulations1(1)

