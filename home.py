import time

import pandas as pd
import streamlit as st
import hydralit_components as hc

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


@st.cache(allow_output_mutation=True, persist=True)
def reading_dataset():
    global dataset
    try:
        dataset = pd.read_excel(uploaded_file)
    except ValueError:
        dataset = pd.read_csv(uploaded_file)
    return dataset


c1, c2, c3 = st.beta_columns(3)
with c2:
    st.image('bgu-large.png', width=450, )
st.title("Final Project in Online Feature Selection, By:")
st.header("Daniel Goldman, Samuel Benichou and Tom Dugma")
st.sidebar.header("--------------MENU---------------")
with st.sidebar.subheader('Upload your file'):
    uploaded_file = st.sidebar.file_uploader("Please upload a file of type: xlsx, csv", type=["xlsx", "csv","arff"])

    OL_Algorithm = st.sidebar.selectbox("Choose OL Algorithm",
                                        ["K-NN", "Perceptron Mask (ANN)", "Random Forest", "Naive Bayes"])
    if OL_Algorithm == "K-NN":
        n_neighbors = st.sidebar.slider("select size of K", 0, 10)
        windows_size = st.sidebar.slider("select max windows size", 0, 1000)
        leaf_size = st.sidebar.slider("select size of leaf size", 0, 100)
    elif OL_Algorithm == "Perceptron Mask (ANN)":
        alpha = st.sidebar.slider("select size of alpha", 0.001, 0.99)
        max_iter = st.sidebar.slider("select max num of iterations", 1, 1000)
        random_state = st.sidebar.slider("select random state", 0, 100)
    elif OL_Algorithm == "Random Forest":
        n_estimators = st.sidebar.slider("select num of estimators", 1, 10)
        lambda_value = st.sidebar.slider("select lambda value", 1, 10)
        split_confidence = st.sidebar.slider("select split conf", 0.001, 0.1)
        tie_threshold = st.sidebar.slider("select tie threshold", 0.001, 0.1)

    OFS_Algorithm = st.sidebar.selectbox("Choose OFS Algorithm", ["ABFS", "FIRES"])
    feature_precent = st.sidebar.number_input("Enter feature precentage between 0.01 - 100", min_value=0.01,
                                              max_value=100.0)
    batch_size = st.sidebar.number_input("Enter batch size between 1 - 1000", min_value=1, max_value=1000)
    target_index = st.sidebar.slider("Enter target index", 0, 100)

st.subheader("after choosing wanted options and you uploaded the file, click 'RUN' to get results")
st.write('')
c1, c2, c3 = st.beta_columns(3)

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
        'beginning calculations... please hold on'
    )
    if OFS_Algorithm == 'ABFS':
        st.write(
            'Running ABFS'
        )

    elif OFS_Algorithm == 'FIRES':
        st.write(
            'Running FIRES'
        )
    with hc.HyLoader('', hc.Loaders.pulse_bars):
        time.sleep(10)
        st.write("YEY!")