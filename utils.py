import os
import altair as alt
import random
import subprocess

this_dir = os.path.dirname(os.path.abspath(__file__))
moa_jar = os.path.join(this_dir, 'resources', "moa-pom.jar")
sizeofag_jar = os.path.join(this_dir, 'resources', "sizeofag-1.0.4.jar")

moa_classifiers_db = {
    "Naive Bayes": "EvaluatePrequential -l (meta.featureselection.FeatureSelectionClassifier -l bayes.NaiveBayes -s (newfeatureselection.BoostingSelector2 -g 100 -t 0.05 -D) -g 0.7 -m) -s (ArffFileStream -f input_file -c target_index) -f batch_size -d output_file",
    "Hoeffding Tree": "EvaluatePrequential -l (meta.featureselection.FeatureSelectionClassifier -s (newfeatureselection.BoostingSelector2 -g 100 -t 0.05 -D) -g 0.7 -m) -s (ArffFileStream -f input_file -c target_index) -f batch_size -d output_file",
    "KNN": "EvaluatePrequential -l (meta.featureselection.FeatureSelectionClassifier -l (lazy.kNN -k n_neighbors -w leaf_size) -s (newfeatureselection.BoostingSelector2 -g 100 -t 0.05 -D) -g 0.7 -m) -s (ArffFileStream -f input_file -c target_index) -f batch_size -d output_file",
    "Perceptron Mask (ANN)": "EvaluatePrequential -l (meta.featureselection.FeatureSelectionClassifier -l (rules.functions.Perceptron -r alpha) -s (newfeatureselection.BoostingSelector2 -g 100 -t 0.05 -D)) -s (ArffFileStream -f input_file -c target_index) -f batch_size -d output_file"
}


def run_moa_subprocess(classifier, input_path, output_path, target_index, batch_size, classifier_parameters=None):
    """
    This function replace the default java moa commands with given parameters and runs the command in a subprocess
    :param moa_jar_path: path to moa jar file
    :param size_of_jar_path: path to size of jar file
    :param command: the command from the classifiers db
    :param input_path: input file path for moa
    :param output_path: output file path for results
    :param target_index: dataset target index
    :param batch_size: batch size for the online learning simulation
    """
    command = moa_classifiers_db[classifier]
    if int(target_index) == -1:
        args = command.replace("input_file", input_path).replace("output_file", output_path).replace(" -c target_index",
                                                                                                     "").replace(
            "batch_size", str(batch_size))
    else:
        args = command.replace("input_file", input_path).replace("output_file", output_path).replace("target_index",
                                                                                                     str(target_index)).replace(
            "batch_size", str(batch_size))

    if classifier_parameters:
        for param in classifier_parameters:
            args.replace(param, classifier_parameters[param])

    cmd = f'java -cp {moa_jar} -javaagent:{sizeofag_jar}  moa.DoTask {args}'

    print(f"Running the command: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    print(process.stdout.read())
    return


def data_shuffle(input_path, output_path):
    """
    shuffle dataset
    :param input_path: path to input file
    :param output_path: path to output file
    """
    fid = open(input_path, "r")
    li = fid.readlines()
    fid.close()

    if input_path.endswith('arff'):
        data_index = li.index('@data\n')
    else:
        data_index = 1

    meta_data = li[:data_index + 1]

    data = li[data_index + 1:]
    random.shuffle(data)

    fid = open(output_path, "w")
    fid.writelines(meta_data + data)
    fid.close()


def get_chart(data):
    """
    create chart for the data
    :param data: the dataset
    :return: interactive chart
    """
    hover = alt.selection_single(
        fields=["date"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    lines = (
        alt.Chart(data, title="Evolution of stock prices")
        .mark_line()
        .encode(
            x="date",
            y="price",
            color="symbol",
            strokeDash="symbol",
        )
    )

    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)

    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="date",
            y="price",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("date", title="Date"),
                alt.Tooltip("price", title="Price (USD)"),
            ],
        )
        .add_selection(hover)
    )

    return (lines + points + tooltips).interactive()