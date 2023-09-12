# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle

import streamlit as st
import pickle 
import pandas as pd


#reading the encoder, model and scaler object files
encoder = pickle.load(open("C:/Users/Dell/Documents/Coding Samurai Internship/encoder.pkl", 'rb'))
model = pickle.load(open("C:/Users/Dell/Documents/Coding Samurai Internship/SVC_model.pkl", 'rb'))
scaler = pickle.load(open("C:/Users/Dell/Documents/Coding Samurai Internship/scaler.pkl", 'rb'))


#setting the title 
st.title("Iris Flower Species Prediction")


#taking the input from user
input_SL = st.number_input("Enter sepal Length (cm):", min_value=0.0)
input_SW = st.number_input("Enter sepal Width (cm):", min_value=0.0)
input_PL = st.number_input("Enter petal Length (cm):", min_value=0.0)
input_PW = st.number_input("Enter petal Width (cm):", min_value=0.0)

#button to trigger the classification
if st.button("Classify"):
    input_values = pd.DataFrame([[input_SL, input_SW, input_PL, input_PW]])
    input_values = scaler.transform(input_values)
    prediction = model.predict(input_values)
    result = encoder.inverse_transform(prediction)
    st.markdown(f"Prediction result: **{result[0]}**")