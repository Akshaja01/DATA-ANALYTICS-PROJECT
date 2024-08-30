pip install joblib
import streamlit as st
import joblib

st.set_page_config(page_title="Real Estate Price Prediction", page_icon="🏠", layout="wide")
st.title("🏠 Real Estate Price Prediction")

model_ridge = joblib.load('project/ridge_real_estate.pkl')
model_lasso = joblib.load('project/lasso_real_estate.pkl')
model_elasticnet = joblib.load('project/elasticnet_real_estate.pkl')

st.header("Prediction")


n1 = float(st.number_input("Enter value for Transaction Date (e.g., 2013.250): "))
n2 = float(st.number_input("Enter value for House Age (years): "))
n3 = float(st.number_input("Enter distance to the nearest MRT station (meters): "))
n4 = int(st.number_input("Enter number of convenience stores nearby: "))
n5 = float(st.number_input("Enter latitude of the property: "))
n6 = float(st.number_input("Enter longitude of the property: "))


sample1 = [[n1, n2, n3, n4, n5, n6]]

if st.button("Predict the price"):
    
    t1 = model_ridge.predict(sample1)
    t2 = model_lasso.predict(sample1)
    t3 = model_elasticnet.predict(sample1)
    
 
    if t1 is not None:
        st.write("Predicted price of the property is:")
        c1, c2, c3 = st.columns(3)
        c1.subheader("Ridge Regression")
        c2.subheader("LASSO Regression")
        c3.subheader("ElasticNet Regression")
        c1.write(t1[0])
        c2.write(t2[0])
        c3.write(t3[0])
    else:
        st.write("Price cannot be determined")
