import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
with open("logistic_regression_model_health.pkl", 'rb') as file:
    model = pickle.load(file)
    st.title("Medical Test Result Classification")
    st.write("This app classifies. ")

import pickle

# Make sure the path is correct and file exists
try:
    with open('path_to_your_model_file.pkl', 'rb') as file:
        model = pickle.load(file)
except EOFError:
    print("Error: The file is empty or corrupted.")
