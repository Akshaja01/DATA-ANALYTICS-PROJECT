import streamlit as st

pg=st.navigation([

st.Page("retail.py",title="Retail Sales Dashboard"),
st.Page("realnewmain.py",title="Real Estate Price  Data Analysis"),
st.Page("realnewpred.py",title="Real Estate Price Prediction"),
st.Page("whosl.py",title="Wholesale Customers Data Analysis"),
st.Page("mushmain.py",title="Mushroom Data Analysis"),
st.Page("mushpred.py",title="Mushroom Data  Prediction")





])

pg.run()