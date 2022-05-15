import subprocess
import os
import time

import pandas as pd
import numpy as np
import random

# Globals:
this_dir = os.path.dirname(os.path.abspath(__file__))
moa_jar = "moa-pom.jar"
sizeofag_jar = "sizeofag-1.0.4.jar"

classifiers_db = {
    # "NB": "EvaluatePrequential -l (meta.featureselection.FeatureSelectionClassifier -l bayes.NaiveBayes -s (newfeatureselection.BoostingSelector2 -g 100 -t 0.05 -D)) -s (ArffFileStream -f (input_file) -c target_index) -f batch_size -d output_file",
    # "HT": "EvaluatePrequential -l (meta.featureselection.FeatureSelectionClassifier -s (newfeatureselection.BoostingSelector2 -g 100 -t 0.05 -D)) -s (ArffFileStream -f (input_file) -c target_index) -f batch_size -d output_file",
    # "KNN": "EvaluatePrequential -l (meta.featureselection.FeatureSelectionClassifier -l (lazy.kNN -k 500) -s (newfeatureselection.BoostingSelector2 -g 100 -t 0.05 -D)) -s (ArffFileStream -f (input_file) -c target_index) -f batch_size -d output_file",
    "ANN": "EvaluatePrequential -l (meta.featureselection.FeatureSelectionClassifier -l functions.Perceptron -s (newfeatureselection.BoostingSelector2 -g 100 -t 0.05 -D)) -s (ArffFileStream -f (input_file) -c target_index) -f batch_size -d output_file"

}

data_db = {
    # "covtype": [os.path.join(this_dir, 'Data', 'expirement', "covtype_2_vs_all.arff"), -1],
    # "Electricity": [os.path.join(this_dir, 'Data', 'expirement', "electricity_data.arff"), -1],
    # "df_madelon": [os.path.join(this_dir, 'Data', 'expirement', "df_madelon.arff"), 0],
    "SPAM": [os.path.join(this_dir, 'Data', 'expirement', "df_spambase_sample.arff"), -1],
    # "Poker_Hand": [os.path.join(this_dir, 'Data', 'expirement', 'poker_9_vs_all.arff'), -1],
    "HAR": [os.path.join(this_dir, 'Data', 'expirement', 'df_har.arff'), 0],
    # "Dota": [os.path.join(this_dir, 'Data', 'expirement', 'df_dota.arff'), 0],
    # "Gisette": [os.path.join(this_dir, 'Data', 'expirement', 'df_gisette.arff'), 0],
    # "KDD": [os.path.join(this_dir, 'Data', 'expirement', 'df_kdd.arff'), -1],
    # "RTG": [os.path.join(this_dir, 'Data', 'expirement', 'df_rtg.arff'), 0],
    # "MNIST": [os.path.join(this_dir, 'Data', 'expirement', 'df_mnist.arff'), 0],

    # "10_KDD99":[os.path.join(this_dir, 'Data', '10kdd.arff'), 118],
    # "Spam_Assassin_SCorpus":[os.path.join(this_dir, 'Data', 'corpus_data.arff'), 39916]
    # "df_kdd": os.path.join(this_dir, 'Data', "df_kdd.arff") # 41
    # "Forest_CoverType":[os.path.join(this_dir, 'Data', 'covtype_2_vs_all.arff'), 54],
    # "Electricity":[os.path.join(this_dir, 'Data', 'electricity_data.arff'),14],
    # "Airlines":[os.path.join(this_dir, 'Data', 'airline_data.arff'),607],
    # "Madelon":[os.path.join(this_dir, 'Data', 'df_madelon.arff'),0],
    # "RBF":[os.path.join(this_dir, 'Data', 'df_rbf.arff'), 0],
    # "IADS": os.path.join(this_dir, 'Data', "ad-dataset.arff"),
    # "Nomao": os.path.join(this_dir, 'Data', "Nomao.arff"),
    # "PAMAP2": "",
    # "Spambase":[os.path.join(this_dir, 'Data', 'df_spambase_sample.arff'), 57],

}
result = {}


def run(moa_jar_path, size_of_jar_path, command, input_path, output_path, target_index, batch_size):
    if target_index == -1:
        args = command.replace("input_file", input_path).replace("output_file", output_path).replace(" -c target_index",
                                                                                                     "").replace(
            "batch_size", str(batch_size))
    else:
        args = command.replace("input_file", input_path).replace("output_file", output_path).replace("target_index",
                                                                                                     str(target_index)).replace(
            "batch_size", str(batch_size))

    cmd = f'java -cp {moa_jar_path} -javaagent:{size_of_jar_path}  moa.DoTask {args}'

    print(f"Running the command: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    #process.wait(120)
    print(process.stdout.read())
    return


def run_simulations1(num_of_tests):
    if not os.path.exists(os.path.join(this_dir, 'Results')):
        os.makedirs(os.path.join(this_dir, 'Results'))
    for classifier in classifiers_db.keys():
        for dataset_name in data_db.keys():
            if data_db[dataset_name] == "":
                continue
            else:
                for batch_size in [25, 50, 75, 100]:
                    total_acc = 0
                    total_time = 0
                    output_path = os.path.join(this_dir, 'Results', f"{classifier}_{dataset_name}_{batch_size}.csv")
                    timeout_start = time.time()
                    timeout = 600
                    try:
                        run(moa_jar, sizeofag_jar, classifiers_db[classifier], data_db[dataset_name][0], output_path, data_db[dataset_name][1], batch_size)
                        while not os.path.exists(output_path) and time.time() < timeout_start + timeout:
                            continue
                        if os.path.exists(output_path):
                            df = pd.read_csv(output_path)
                            res_acc = df['classifications correct (percent)'].mean()
                            res_time = df['evaluation time (cpu seconds)'].sum()
                            total_time += res_time
                            total_acc += res_acc
                        else:
                            print(f"timeout for {classifier}_{dataset_name}_{batch_size}")

                        result[f"{classifier}_{dataset_name}_acc_{batch_size}"] = res_acc
                        result[f"{classifier}_{dataset_name}_time_{batch_size}"] = res_time
                    except Exception as e:
                        print(e)
                    try:
                        result[f"{classifier}_{dataset_name}_acc_avg"] = total_acc / 4
                        result[f"{classifier}_{dataset_name}_time_avg"] = total_time / 4
                        result[f"{classifier}_{dataset_name}_num_of_features"] = df['# of features selected'].max()

                    except Exception as e:
                        print(e)

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

    meta_data = li[:data_index + 1]

    data = li[data_index + 1:]
    random.shuffle(data)

    fid = open(output_path, "w")
    fid.writelines(meta_data + data)
    fid.close()


def run_main():
    run_simulations1(1)
    return result


if __name__ == "__main__":
    run_simulations1(1)
    # suffle_dataset()
