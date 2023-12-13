import streamlit as st
from PIL import Image
import diagnosis
import joblib

def load_model():
    return joblib.load('C:/Users/Campin Waladsae A/Downloads/cp05-main/Covid19-dataset')
st.title('Immune Deficiency Classifier')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "gif"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, width=300)
    st.write("Diagnosis:", diagnosis.diagnosis(uploaded_file))
else:
    st.write("Please upload an image.")