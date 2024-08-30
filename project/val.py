import streamlit as st

pg=st.navigation([

st.Page("realnewmain.py",title="Real estate  data analysis"),
st.Page("realnewpred.py",title="Real estate data prediction"),
st.Page("whosl.py",title="wholesale data analysis"),
st.Page("mushmain.py",title="Mushroom data analysis"),
st.Page("mushpred.py",title="Mushroom data  prediction")





])

pg.run()