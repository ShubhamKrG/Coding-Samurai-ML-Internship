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
newSL = st.number_input("Enter sepal Length (cm):", min_value=0.0)
newSW = st.number_input("Enter sepal Width (cm):", min_value=0.0)
newPL = st.number_input("Enter petal Length (cm):", min_value=0.0)
newPW = st.number_input("Enter petal Width (cm):", min_value=0.0)

#button to trigger the classification
if st.button("Classify"):
    newValue = pd.DataFrame([[newSL, newSW, newPL, newPW]])
    newValue = scaler.transform(newValue)
    prediction = model.predict(newValue)
    finalAns = encoder.inverse_transform(prediction)
    st.markdown(f"Prediction result: **{finalAns[0]}**")