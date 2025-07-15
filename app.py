import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
st.title("Customer Churn Prediction")
model=load_model('model.h5')
with open('One_Hot_Encoder_geo.pkl','rb') as file:
    OHE_geo=pickle.load(file)
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)
with open('StandScaler.pkl','rb') as file:
    scaler=pickle.load(file)
creditscore=st.number_input("Credit score")
Geography=st.selectbox("Geography",OHE_geo.categories_[0])
Gender=st.selectbox("Gender", label_encoder_gender.classes_ )
Age=st.slider("Age",18,100)
Tenure=st.slider("Tenure",0,10)
Balance=st.number_input("Balance")
products=st.slider("Number of products",0,4)
CrCard=st.selectbox("Has Credit Card",[0,1])
ActMem=st.selectbox("Is Active Member",[0,1])
Sal=st.number_input("Salary")
input_data = {
    'CreditScore': creditscore,
    'Geography': Geography,
    'Gender': Gender,
    'Age': Age,
    'Tenure': Tenure,
    'Balance': Balance,
    'NumOfProducts': products,
    'HasCrCard': CrCard,
    'IsActiveMember': ActMem,
    'EstimatedSalary': Sal
}
input_data=pd.DataFrame([input_data])
input_data['Gender']=label_encoder_gender.transform(input_data['Gender'])
input_data_geo=OHE_geo.transform(input_data[['Geography']])
input_data_geo=pd.DataFrame(input_data_geo,columns= OHE_geo.get_feature_names_out(['Geography']))
input_data=pd.concat([input_data.drop('Geography',axis=1),input_data_geo],axis=1)
input_data=scaler.transform(input_data)
prediction=model.predict(input_data)
prob=prediction[0][0]
but=st.button("Predict whether customer will churn")

if but:
    st.write(f'Churn Probability: {prob:.2f}')
    if prob<0.5:
        st.write("The customer is not likely to churn")
    else:
        st.write("The customer is likely to churn")