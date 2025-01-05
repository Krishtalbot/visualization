from narwhals import col
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans

st.set_page_config(page_title='Data Visualization', page_icon='📊', layout='wide')
st.markdown('<style>div.block-container{padding-top: 1rem;}</style>', unsafe_allow_html=True)
st.title('Visualization of Corruption, GDP and Happiness')

df = pd.read_csv(f'datasets/merged.csv', encoding='ISO-8859-1')

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


st.sidebar.header("Metric for average visualization:")
metric = st.sidebar.selectbox(
    "Select a metric",
    options=['Happiness_Score', 'CPI_Score', 'GDP', 'Freedom_Index'],
    index=0
)

st.markdown("""
    <h2 style="color:#E09145; font-size:24px; text-align:center; font-weight: bold; font-size: 2rem;">
        Global Map
    </h2>
    """,
    unsafe_allow_html=True)
            
filtered_df['Formatted_GDP'] = filtered_df['GDP'].apply(lambda x: f"${x:,.0f}")

hover_data = {
    "CPI_Score": True,
    "Formatted_GDP": True,
    "Happiness_Score": True
}

fig = px.choropleth(
    filtered_df,
    locations="Country",  
    locationmode="country names",  
    color="Continent",  
    hover_name="Country",  
    hover_data=hover_data,  

    color_continuous_scale="Viridis",  
)

fig.update_layout(
    geo=dict(
        showframe=False,
        showcoastlines=True,
        coastlinecolor="LightGrey",
        projection_type="natural earth",
        bgcolor="#0E1117"
    ),
    margin={"r": 0, "t": 50, "l": 0, "b": 0},
    paper_bgcolor="#0E1117",
    font=dict(color="white")
)

st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
avg_df = filtered_df.groupby('Continent')[[metric]].mean().reset_index()

with col1:
    st.markdown('''<h2 style="color:#FAFAFA; font-size:2rem; margin-bottom:-2rem; margin-top:2rem; ">Average <span style="font-weight: bold;"> {}</span> by Continent</h2>'''.format(metric.replace('_', ' ')), 
                unsafe_allow_html=True)
    fig = px.bar(
        avg_df,
        x='Continent',
        y=metric,
        color='Continent',
        labels={metric: metric.replace('_', ' ')}
    )
    fig.update_layout(
    margin=dict(t=20, b=10),)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('''<h2 style="color:#FAFAFA; font-size:2rem; margin-bottom:-2rem; margin-top:2rem;">Top 10 countries by <span style="font-weight: bold;"> {}</span></h2>'''.format(metric.replace('_', ' ')), 
                unsafe_allow_html=True)
    top10_df = filtered_df.nlargest(10, metric)
    fig = px.bar(
        top10_df,
        x='Country',
        y=metric,
        color='Country',
        labels={metric: metric.replace('_', ' ')},
    )
    fig.update_layout(
    margin=dict(t=20, b=10),)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Scatter Plot")

col3, col4 = st.columns(2)
with col3:  
    x_axis = st.selectbox("Select X-axis", ['CPI_Score', 'Freedom_Index', 'Happiness_Score', 'Unemployment_Rate', 'GDP'], index=0)

with col4:
    y_axis = st.selectbox("Select Y-axis", ['CPI_Score', 'Freedom_Index', 'Happiness_Score', 'Unemployment_Rate', 'GDP'], index=1)

scatter_fig = px.scatter(
    filtered_df,
    x=x_axis,
    y=y_axis,
    color='Continent',
    hover_name='Country',
    # size='GDP' if 'GDP' in filtered_df.columns else None, 
    title=f"{x_axis.replace('_', ' ')} vs {y_axis.replace('_', ' ')}",
    labels={x_axis: x_axis.replace('_', ' '), y_axis: y_axis.replace('_', ' ')},
)

scatter_fig.update_layout(
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
    font=dict(color="white"),
    margin={"r": 0, "t": 50, "l": 0, "b": 0},
)

st.plotly_chart(scatter_fig, use_container_width=True)

st.subheader("Heatmap: Correlation Matrix")


# Clustering Analysis
st.subheader("Cluster Analysis")
cluster_features = ['CPI_Score', 'GDP', 'Happiness_Score']
if all(feature in filtered_df.columns for feature in cluster_features):
    kmeans = KMeans(n_clusters=3, random_state=42)
    filtered_df['Cluster'] = kmeans.fit_predict(filtered_df[cluster_features].dropna())

    cluster_fig = px.scatter(
        filtered_df,
        x='CPI_Score',
        y='Happiness_Score',
        color='Cluster',
        size='GDP',
        hover_name='Country',
        title="Clusters of Countries Based on CPI, GDP, and Happiness Scores",
        labels={'Cluster': 'Cluster Group'}
    )
    st.plotly_chart(cluster_fig, use_container_width=True)

# Standard Error Visualization
st.subheader("Standard Errors for CPI and Happiness Scores")
se_df = filtered_df.groupby('Continent')[['CPI_Score', 'Happiness_Score']].agg(['mean', 'sem']).reset_index()
se_df.columns = ['Continent', 'CPI_Mean', 'CPI_SE', 'Happiness_Mean', 'Happiness_SE']

fig_cpi = px.bar(
    se_df,
    x='Continent',
    y='CPI_Mean',
    error_y='CPI_SE',
    title="CPI Scores with Standard Errors by Continent",
    labels={'CPI_Mean': 'Mean CPI Score', 'Continent': 'Continent'},
    color='Continent'
)
st.plotly_chart(fig_cpi, use_container_width=True)

fig_happiness = px.bar(
    se_df,
    x='Continent',
    y='Happiness_Mean',
    error_y='Happiness_SE',
    title="Happiness Scores with Standard Errors by Continent",
    labels={'Happiness_Mean': 'Mean Happiness Score', 'Continent': 'Continent'},
    color='Continent'
)
st.plotly_chart(fig_happiness, use_container_width=True)

