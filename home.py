import os
from ABFS.ABFS import ABFS
from Fires import FIRES
import pandas as pd
import streamlit as st
import altair as alt
import utils

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Final Project", page_icon="🔍", layout="wide")
hide_streamlit_style = """
            <style>
            footer {
	        visibility: hidden;
	            }
            footer:after {
	            content:'developed by Tom Dugma,Samuel Benichou and Daniel Goldman'; 
	            visibility: visible;
	            display: block;
	            position: relative;
	            #background-color: red;
	            padding: 5px;
	            top: 2px;
                    }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# body
c1, c2, c3 = st.columns(3)
with c2:
    st.image(os.path.join('resources', 'bgu-large.png'), width=450, )
st.title("Final Project in Online Feature Selection, By:")
st.header("Daniel Goldman, Samuel Benichou and Tom Dugma")
st.sidebar.header("--------------MENU---------------")
with st.sidebar.subheader('Upload your file'):
    uploaded_file = st.sidebar.file_uploader("Please upload a file of type: arff, csv", type=["csv", "arff"])
    if uploaded_file:
        file_path = utils.save_uploaded_file(uploaded_file)
    else:
        st.write("Please upload a file")

    data_shuffle = st.sidebar.checkbox("Data Shuffle")

    OFS_Algorithm = st.sidebar.selectbox("Choose OFS Algorithm", ["ABFS", "FIRES"])
    OL_Algorithm = st.sidebar.selectbox("Choose OL Algorithm",
                                        ["KNN", "Perceptron Mask (ANN)", "Hoeffding Tree", "Naive Bayes"])
    classifier_parameters = {}
    if OL_Algorithm == "KNN":
        n_neighbors = st.sidebar.slider("select size of K", 0, 500)
        leaf_size = st.sidebar.slider("select size of leaf size", 0, 50000)
        classifier_parameters['KNN'] = {'n_neighbors': n_neighbors, 'leaf_size': leaf_size}
    elif OL_Algorithm == "Perceptron Mask (ANN)":
        alpha = st.sidebar.slider("select size of alpha", 0.01, 0.99)
        max_iter = st.sidebar.slider("select max num of iterations", 1, 1000)
        random_state = st.sidebar.slider("select random state", 0, 100)
        classifier_parameters['Perceptron Mask (ANN)'] = {'alpha': alpha, 'max_iter': max_iter,
                                                          'random_state': random_state}

    batch_size = st.sidebar.number_input("Enter batch size between 1 - 1000", min_value=1, max_value=1000)
    target_index = st.sidebar.slider("Enter target index", -1, 100)

st.subheader("After modifying all parameters and uploading data, click 'RUN' to start evaluation")
st.write('')
c1, c2, c3 = st.columns(3)

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(240,248,255);
    padding: 15px 100px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 20px;
    margin: 4px 2px;
    cursor: pointer;
}
</style>""", unsafe_allow_html=True)

with c2:
    run = st.button("RUN")
if uploaded_file is not None and run:
    st.write(
        'Preparing models...'
    )
    res = None
    if OFS_Algorithm == 'ABFS':
        if uploaded_file.name.endswith('arff'):
            st.write(
                'Running ABFS'
            )
            abfs = ABFS()
            res = abfs.run_abfs(classifier_name=OL_Algorithm, classifier_parameters=classifier_parameters,
                                data=file_path, target_index=target_index, data_shuffle=data_shuffle,
                                batch_size=batch_size)
        else:
            st.write("ABFS support only arff file formats.")
    elif OFS_Algorithm == 'FIRES':
        if uploaded_file.name.endswith('csv'):
            st.write(
                'Running FIRES'
            )
            res = FIRES.apply_fires(classifier_name=OL_Algorithm, classifier_parameters=classifier_parameters,
                                    data=file_path, target_index=target_index, batch_size=batch_size)
        else:
            st.write("Fires support only csv file formats.")
    if res:
        st.write("Success!")
        st.write("Results:")
        if 'evaluation_time' in res:
            st.write(f"Evaluation time:  {res['evaluation_time']} seconds")
        if 'avg_acc' in res:
            st.write(f"Average accuracy:  {res['avg_acc']}%")
        if 'avg_stab' in res:
            st.write(f"Average Stability:  {res['avg_stab']}%")
        if 'num_of_features' in res:
            st.write(f"Num Of Features: {res['num_of_features']}")
        for key in res.keys():
            if not res[key]:
                res[key] = 0

        st.subheader('Display Results')
        try:
            df = pd.DataFrame.from_dict(res, orient='index')
            df = df.rename({0: 'Value'}, axis='columns')
            df.reset_index(inplace=True)
            df = df.rename(columns={'index': 'Type'})
            st.write(df)
            ### 4. Display Bar Chart using Altair
            st.subheader('Display Bar chart')
            p = alt.Chart(df).mark_bar().encode(
                x='Type',
                y='Value'
            )
            p = p.properties(
                width=alt.Step(150),
                height=alt.Step(150)
            )
            st.write(p)
        except Exception as e:
            print(e)
    else:
        st.write("Error! please check your parameters and data.")
