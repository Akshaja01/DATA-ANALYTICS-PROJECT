import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn import metrics as mat
import plotly.express as px
from sklearn.cluster import KMeans
import numpy as np

st.set_page_config(page_title="Wholesale Customers Data Analysis", page_icon="📊", layout="wide")
st.title("📊 Wholesale Customers Data Analysis 📊")


df = pd.read_csv('Wholesale_customers_data.csv')
st.header('📊 WHOLESALE CUSTOMERS DATA SET 📊')
st.table(df.head())


df.dropna(inplace=True)

cl1, cl2 = st.columns(2)

cl1.header("Count of unique Channel types")
cl1.table(df['Channel'].value_counts())

cl1.header("Count of unique Region types")
cl1.table(df['Region'].value_counts())

x = df[['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']]
y = df[['Channel']]

wcss = []
k = []
for i in range(1, 11):
    k.append(i)
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

st.write("Values of k and WCSS")
fig, ax = plt.subplots(figsize=(2, 2))
ax.plot(k, wcss, c='g', marker='o', mfc='r')
st.pyplot(fig)

km_final = KMeans(n_clusters=2, init='k-means++', max_iter=300, random_state=0)  # Assuming 2 channels
df['new_label'] = km_final.fit_predict(x)

st.header('📊 WHOLESALE CUSTOMERS DATA SET WITH NEW LABELS 📊')
st.table(df)

st.header("Visualizing the new labels and clusters")

fig2 = px.scatter(df, x='Fresh', y='Milk', size='Grocery', color='new_label')
st.plotly_chart(fig2, use_container_width=True)

fig3 = px.scatter(df, x='Fresh', y='Milk', size='Grocery', color='Channel')
st.plotly_chart(fig3, use_container_width=True)

dbs = mat.davies_bouldin_score(x, km_final.labels_)
sil = mat.silhouette_score(x, km_final.labels_)
cal = mat.calinski_harabasz_score(x, km_final.labels_)

ars = mat.adjusted_rand_score(df['Channel'], km_final.labels_)
mu = mat.mutual_info_score(df['Channel'], km_final.labels_)

st.header("Evaluation Scores")

c1, c2, c3 = st.columns(3)
c4, c5 = st.columns(2)

c1.subheader('Davies-Bouldin Score')
c1.subheader(dbs)
c2.subheader('Silhouette Score')
c2.subheader(sil)
c3.subheader('Calinski-Harabasz Score')
c3.subheader(cal)
c4.subheader('Adjusted Rand Score')
c4.subheader(ars)
c5.subheader('Mutual Information Score')
c5.subheader(mu)


pickle.dump(km_final, open('kmeans_wholesale.pkl', 'wb'))
