import streamlit as st
import pandas as pd
import plotly.express as px
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Retail Sales Dashboard", page_icon=":bar_chart:", layout="wide")

st.title(":bar_chart: Retail Sales Dashboard")

df = pd.read_csv("retail_sales_dataset.csv")  

st.subheader("First 5 Rows of Retail Data")
st.dataframe(df.head())

df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

st.sidebar.header("Choose Your Filters")
year_filter = st.sidebar.multiselect("Select Year(s)", options=df['Year'].unique(), default=df['Year'].unique())
gender_filter = st.sidebar.multiselect("Select Gender(s)", options=df['Gender'].unique(), default=df['Gender'].unique())


filtered_df = df[
    (df['Year'].isin(year_filter)) &
    (df['Gender'].isin(gender_filter))
]

if st.button('Show Summary'):

    col1, col2 = st.columns(2)
    
    col1.subheader("Total Transactions")
    col1.write(f"{filtered_df.shape[0]:,}")

    col2.subheader("Unique Customers")
    col2.write(f"{filtered_df['Customer ID'].nunique():,}")


st.subheader("Transactions Trend")
transactions_trend = filtered_df.groupby('Month').size().reset_index(name='TotalTransactions')
fig1 = px.line(transactions_trend, x='Month', y='TotalTransactions', title='Monthly Transactions Trend')
st.plotly_chart(fig1, use_container_width=True)


st.subheader("Transactions by Gender")
gender_distribution = filtered_df.groupby('Gender').size().reset_index(name='TotalTransactions')
fig2 = px.pie(gender_distribution, values='TotalTransactions', names='Gender', hole=0.2, title='Transactions by Gender')
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Top 10 Customers by Transactions")
top_customers = filtered_df.groupby('Customer ID').size().reset_index(name='TotalTransactions')
top_customers = top_customers.sort_values(by='TotalTransactions', ascending=False).head(10)
fig3 = px.bar(top_customers, x='Customer ID', y='TotalTransactions', title='Top 10 Customers by Transactions')
st.plotly_chart(fig3, use_container_width=True)

if 'Region' in df.columns:
    st.subheader("Sales by Region")
    sales_by_region = filtered_df.groupby('Region').size().reset_index(name='TotalSales')
    fig4 = px.bar(sales_by_region, x='Region', y='TotalSales', title='Sales by Region')
    st.plotly_chart(fig4, use_container_width=True)