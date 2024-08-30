import streamlit as st
import pickle


model1 = pickle.load(open('kmeans_wholesale.pkl', 'rb'))

st.header("Wholesale Customer Segment Prediction")


c1, c2 = st.columns(2)
n1 = float(c1.number_input("Enter Fresh product spend"))
n2 = float(c1.number_input("Enter Milk product spend"))
n3 = float(c2.number_input("Enter Grocery spend"))
n4 = float(c2.number_input("Enter Frozen product spend"))
n5 = float(c1.number_input("Enter Detergents_Paper spend"))
n6 = float(c2.number_input("Enter Delicassen spend"))

st.write("Enter the annual spending on the various product categories in monetary units.")


sample = [[n1, n2, n3, n4, n5, n6]]

if st.button("Predict Customer Segment"):
    t = model1.predict(sample)
    if t == 0:
        st.write("🛒 Segment 1: Retail Channel")
    elif t == 1:
        st.write("🛒 Segment 2: Horeca (Hotel/Restaurant/Café)")
    else:
        st.write("Customer segment not listed")
