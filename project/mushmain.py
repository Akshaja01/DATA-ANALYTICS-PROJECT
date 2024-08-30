import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Mushroom Classification", page_icon="🍄", layout="wide")
st.title("🍄 Mushroom Data Analysis")


mdf = pd.read_csv('mushrooms.csv')
st.header("Mushroom Dataset")
st.dataframe(mdf.head())


st.subheader("Converting Data to Numerical Labels")
le = LabelEncoder()
for column in mdf.columns:
    mdf[column] = le.fit_transform(mdf[column])
st.dataframe(mdf.head())

st.title("Mushroom Data Analysis")
st.subheader("Mushroom Dataset")
st.dataframe(mdf)

st.subheader("Summary statistics")
st.write(mdf.describe())

st.subheader("Pairplot")
pairplot = sns.pairplot(mdf, hue='class')
st.pyplot(pairplot)

plt.figure(figsize=(10, 6))
heatmap = sns.heatmap(mdf.corr(), annot=True, cmap='coolwarm')
st.pyplot(heatmap.figure)
