import streamlit as st
import pandas as pd
import plotly.express as px
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Retail Sales Dashboard", page_icon=":bar_chart:", layout="wide")


st.title(":bar_chart: Retail Sales Dashboard")


df = pd.read_csv("project/retail_sales_dataset.csv")


st.subheader("First 5 Rows of Retail Data")
st.dataframe(df.head())

df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.strftime('%B')


st.sidebar.header("Choose Your Filters")
year_filter = st.sidebar.multiselect("Select Year(s)", options=df['Year'].unique(), default=df['Year'].unique())
gender_filter = st.sidebar.multiselect("Select Gender(s)", options=df['Gender'].unique(), default=df['Gender'].unique())
category_filter = st.sidebar.multiselect("Select Product Category", options=df['Product Category'].unique(), default=df['Product Category'].unique())

filtered_df = df[
    (df['Year'].isin(year_filter)) &
    (df['Gender'].isin(gender_filter)) &
    (df['Product Category'].isin(category_filter))
]


if st.button('Show Summary'):
    col1, col2 = st.columns(2)
    
    col1.subheader("Total Transactions")
    col1.write(f"{filtered_df.shape[0]:,}")

    col2.subheader("Unique Customers")
    col2.write(f"{filtered_df['Customer ID'].nunique():,}")


st.subheader("Transactions Trend")
transactions_trend = filtered_df.groupby('Month').size().reset_index(name='TotalTransactions')
transactions_trend['Month'] = pd.Categorical(transactions_trend['Month'], categories=[datetime.strptime(m, "%B").strftime("%B") for m in transactions_trend['Month']], ordered=True)
transactions_trend = transactions_trend.sort_values('Month')

fig1 = px.line(transactions_trend, x='Month', y='TotalTransactions', title='Monthly Transactions Trend',
               markers=True, line_shape='spline', color_discrete_sequence=px.colors.qualitative.Plotly)
fig1.update_layout(xaxis_title='Month', yaxis_title='Total Transactions', plot_bgcolor='rgba(0,0,0,0)')
fig1.update_traces(line=dict(width=4))

st.plotly_chart(fig1, use_container_width=True)


st.subheader("Transactions by Gender")
gender_distribution = filtered_df.groupby('Gender').size().reset_index(name='TotalTransactions')

fig2 = px.pie(gender_distribution, values='TotalTransactions', names='Gender', title='Transactions by Gender',
              color_discrete_sequence=px.colors.qualitative.Set2, hole=0.3)
fig2.update_traces(textposition='inside', textinfo='percent+label')
fig2.update_layout(showlegend=True)

st.plotly_chart(fig2, use_container_width=True)


st.subheader("Top 10 Customers by Transactions")
top_customers = filtered_df.groupby('Customer ID').size().reset_index(name='TotalTransactions')
top_customers = top_customers.sort_values(by='TotalTransactions', ascending=False).head(10)

fig3 = px.bar(top_customers, x='Customer ID', y='TotalTransactions', title='Top 10 Customers by Transactions',
              text='TotalTransactions', color='TotalTransactions', color_continuous_scale=px.colors.sequential.Inferno)
fig3.update_layout(xaxis_title='Customer ID', yaxis_title='Total Transactions', plot_bgcolor='rgba(0,0,0,0)')
fig3.update_traces(texttemplate='%{text:.2s}', textposition='outside')

st.plotly_chart(fig3, use_container_width=True)


st.subheader("Transactions by Product Category")
category_distribution = filtered_df.groupby('Product Category').size().reset_index(name='TotalTransactions')

fig4 = px.bar(category_distribution, x='Product Category', y='TotalTransactions', title='Transactions by Product Category',
              text='TotalTransactions', color='TotalTransactions', color_continuous_scale=px.colors.sequential.Magma)
fig4.update_layout(xaxis_title='Product Category', yaxis_title='Total Transactions', plot_bgcolor='rgba(0,0,0,0)')
fig4.update_traces(texttemplate='%{text:.2s}', textposition='outside')

st.plotly_chart(fig4, use_container_width=True)
