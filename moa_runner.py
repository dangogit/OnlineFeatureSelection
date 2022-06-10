import os
import pandas as pd

# Globals:
from utils import run_moa_subprocess, moa_classifiers_db

this_dir = os.path.dirname(os.path.abspath(__file__))

data_db = {
    "covtype": [os.path.join(this_dir, 'Data', 'expirement', "covtype_2_vs_all.arff"), -1],
    "Electricity": [os.path.join(this_dir, 'Data', 'expirement', "electricity_data.arff"), -1],
    "df_madelon": [os.path.join(this_dir, 'Data', 'expirement', "df_madelon.arff"), 0],
    "SPAM": [os.path.join(this_dir, 'Data', 'expirement', "df_spambase_sample.arff"), -1],
    "Poker_Hand": [os.path.join(this_dir, 'Data', 'expirement', 'poker_9_vs_all.arff'), -1],
    "HAR": [os.path.join(this_dir, 'Data', 'expirement', 'df_har.arff'), 0],
    "Dota": [os.path.join(this_dir, 'Data', 'expirement', 'df_dota.arff'), 0],
    "Gisette": [os.path.join(this_dir, 'Data', 'expirement', 'df_gisette.arff'), 0],
    "KDD": [os.path.join(this_dir, 'Data', 'expirement', 'df_kdd.arff'), -1],
    "RTG": [os.path.join(this_dir, 'Data', 'expirement', 'df_rtg.arff'), 0],
    "MNIST": [os.path.join(this_dir, 'Data', 'expirement', 'df_mnist.arff'), 0],

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


def run_simulation(batches):
    """
    This function will run the simulations for the requested batch sizes and collect the results
    :param batches: batches list
    """
    if not os.path.exists(os.path.join(this_dir, 'Results')):
        os.makedirs(os.path.join(this_dir, 'Results'))
    for classifier in moa_classifiers_db.keys():
        for dataset_name in data_db.keys():
            if data_db[dataset_name] == "":
                continue
            else:
                for batch_size in batches:
                    total_acc = 0
                    total_time = 0
                    output_path = os.path.join(this_dir, 'Results', f"{classifier}_{dataset_name}_{batch_size}.csv")
                    try:
                        run_moa_subprocess(classifier, data_db[dataset_name][0], output_path, data_db[dataset_name][1], batch_size)

                        if os.path.exists(output_path):
                            df = pd.read_csv(output_path)
                            res_acc = df['classifications correct (percent)'].mean()
                            res_time = df['evaluation time (cpu seconds)'].sum()
                            total_time += res_time
                            total_acc += res_acc
                            result[f"{classifier}_{dataset_name}_acc_{batch_size}"] = res_acc
                            result[f"{classifier}_{dataset_name}_time_{batch_size}"] = res_time
                        else:
                            print(f"failed to run {classifier}_{dataset_name}_{batch_size}")
                    except Exception as e:
                        print(e)

                # save mean of results
                try:
                    result[f"{classifier}_{dataset_name}_acc_avg"] = total_acc / len(batches)
                    result[f"{classifier}_{dataset_name}_time_avg"] = total_time / len(batches)
                    result[f"{classifier}_{dataset_name}_num_of_features"] = df['# of features selected'].max()

                except Exception as e:
                    print(e)

                print(result)

    with open(os.path.join(this_dir, 'result.csv'), 'w') as f:
        for key in result.keys():
            f.write("%s, %s\n" % (key, result[key]))


if __name__ == "__main__":
    run_simulation([25, 50, 75, 100])
