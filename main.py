from narwhals import col
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import os

st.set_page_config(page_title='Data Visualization', page_icon='📊', layout='wide')
st.markdown('<style>div.block-container{padding-top: 1rem;}</style>', unsafe_allow_html=True)
st.title('Visualization of Corruption, GDP and Happiness')

# File uploader
fl = st.file_uploader(':file_folder: Upload your dataset', type=(['csv', 'txt', 'xlsx', 'xls']))
if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv(f'datasets/{filename}', encoding='ISO-8859-1')



st.sidebar.header("Choose filters: ")
continent = st.sidebar.multiselect('Continent', df['Continent'].unique())
if not continent:
    df2 = df.copy()
else:
    df2 = df[df['Continent'].isin(continent)]

country = st.sidebar.multiselect('Country', df2['Country'].unique())
if not country:
    df3 = df2.copy()
else:
    df3 = df2[df2['Country'].isin(country)]

if not continent and not country:
    filtered_df = df
elif not country:
    filtered_df = df[df['Continent'].isin(continent)]
elif not continent:
    filtered_df = df[df['Country'].isin(country)]
else:
    filtered_df = df3[df3['Country'].isin(country) & df3['Continent'].isin(continent)]

st.sidebar.header("Choose a metric for average visualization:")
metric = st.sidebar.selectbox(
    "Select a metric",
    options=['Happiness_Score', 'CPI_Score', 'GDP', 'Freedom_Index'],
    index=0
)

st.subheader("Interactive Global Map")

# Format the GDP column as currency
filtered_df['Formatted_GDP'] = filtered_df['GDP'].apply(lambda x: f"${x:,.0f}")

# Ensure columns are named correctly for Plotly
hover_data = {
    "CPI_Score": True,
    "Formatted_GDP": True,
    "Happiness_Score": True
}

fig = px.choropleth(
    filtered_df,
    locations="Country",  # Column for country names
    locationmode="country names",  # Match with plotly's country names
    color="Continent",  # Color by Happiness Score for a better gradient
    hover_name="Country",  # Display country names on hover
    hover_data=hover_data,  # Show CPI, GDP, and Happiness Score
    title="Global Map of CPI, GDP, and Happiness Score",
    color_continuous_scale="Viridis",  # Smooth gradient
)

# Customize layout for a cleaner look
fig.update_layout(
    geo=dict(
        showframe=False,
        showcoastlines=True,
        coastlinecolor="LightGrey",
        projection_type="natural earth",
        bgcolor="#0E1117"
    ),
    title={
        'text': "Global Map of CPI, GDP, and Happiness Score",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    margin={"r": 0, "t": 50, "l": 0, "b": 0},
    paper_bgcolor="#0E1117",
    font=dict(color="white")
)

# Add annotations and interactivity
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
avg_df = filtered_df.groupby('Continent')[[metric]].mean().reset_index()

with col1:
    st.subheader(f"Average {metric.replace('_', ' ')} by Continent")
    fig = px.bar(
        avg_df,
        x='Continent',
        y=metric,
        color='Continent',
        title=f"Average {metric.replace('_', ' ')} by Continent",
        labels={metric: metric.replace('_', ' ')}
    )
    st.plotly_chart(fig, use_container_width=True)


