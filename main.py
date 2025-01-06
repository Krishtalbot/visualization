import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

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
    options=['Happiness_Score', 'CPI_Score', 'GDP', 'Freedom_Index', 'Unemployment_Rate'],
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
    "CPI_Rank": True,
    "CPI_Score": True,
    "Formatted_GDP": True,
    "Happiness_Score": True,
    "Continent": False,
    "Country": False,
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

metrics = ['Happiness_Score', 'CPI_Score', 'Freedom_Index', 'Unemployment_Rate'] 
global_avg = filtered_df[metrics].mean()

continent_avg = filtered_df.groupby('Continent')[metrics].mean()

deviation_df = continent_avg - global_avg

deviation_df.reset_index(inplace=True)

heatmap_melted = deviation_df.melt(
    id_vars='Continent',
    var_name='Metric',
    value_name='Deviation'
)

heatmap_pivot = heatmap_melted.pivot(index='Metric', columns='Continent', values='Deviation')

import plotly.graph_objects as go

heatmap_fig = go.Figure(
    data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale="RdBu",
        colorbar=dict(title="Deviation"),
        text=heatmap_pivot.round(2).values, 
        texttemplate="%{text}", 
        textfont=dict(size=10), 
    )
)

heatmap_fig.update_layout(
    title="Deviation of Metrics from Global Averages by Continent",
    xaxis=dict(title="Continent", tickangle=45, side="top"),
    yaxis=dict(title="Metric"),
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
    font=dict(color="white"),
    margin={"r": 0, "t": 50, "l": 0, "b": 50}
)

st.plotly_chart(heatmap_fig, use_container_width=True)

st.subheader("Cluster Analysis")
num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)

col5, col6 = st.columns(2)
with col5:
    x_axis = st.selectbox("Select X-axis for Clustering", ['CPI_Score', 'Happiness_Score', 'GDP'], index=0)
with col6:
    y_axis = st.selectbox("Select Y-axis for Clustering", ['CPI_Score', 'Happiness_Score', 'GDP'], index=1)

cluster_features = ['CPI_Score', 'GDP', 'Happiness_Score']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(filtered_df[cluster_features])

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
filtered_df['Cluster'] = kmeans.fit_predict(scaled_data)

centroids = kmeans.cluster_centers_
centroids_df = pd.DataFrame(scaler.inverse_transform(centroids), columns=cluster_features)

cluster_fig = px.scatter(
    filtered_df,
    x=x_axis,
    y=y_axis,
    color='Cluster',
    hover_name='Country',
    title=f"Cluster Analysis: {x_axis.replace('_', ' ')} vs {y_axis.replace('_', ' ')}",
    labels={'Cluster': 'Cluster', x_axis: x_axis.replace('_', ' '), y_axis: y_axis.replace('_', ' ')},
    color_discrete_sequence=px.colors.qualitative.Plotly,
)

# Add centroids to the plot
cluster_fig.add_scatter(
    x=centroids_df[x_axis],
    y=centroids_df[y_axis],
    mode='markers+text',
    marker=dict(symbol='x', size=10, color='black'),
    text=['Centroid'] * num_clusters,
    textposition='top center',
    name='Centroids',
)

cluster_fig.update_layout(
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
    font=dict(color="white"),
    margin={"r": 0, "t": 50, "l": 0, "b": 0}
)

st.plotly_chart(cluster_fig, use_container_width=True)

st.subheader("Box and Whiskers Plot for Standard Errors")

metric_selection = st.selectbox(
    "Select a Metric to Display",
    options=['CPI_Score', 'Happiness_Score'],
    index=0
)

box_fig = px.box(
    filtered_df,
    x='Continent',
    y=metric_selection,
    points='all',
    title=f"{metric_selection.replace('_', ' ')} Scores by Continent",
    labels={metric_selection: f"{metric_selection.replace('_', ' ')} Score", 'Continent': 'Continent'},
    color='Continent',
    color_discrete_sequence=px.colors.qualitative.Plotly
)

box_fig.update_layout(
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
    font=dict(color="white"),
    margin={"r": 0, "t": 50, "l": 0, "b": 0}
)

st.plotly_chart(box_fig, use_container_width=True)

st.subheader("Anomaly Detection")

anomaly_metric = st.selectbox(
    "Select a metric for anomaly detection:",
    options=['GDP', 'CPI_Score', 'Happiness_Score'],
    index=0
)

# Z-Score Anomaly Detection
z_scores = (filtered_df[anomaly_metric] - filtered_df[anomaly_metric].mean()) / filtered_df[anomaly_metric].std()
filtered_df['Z_Score'] = z_scores
filtered_df['Z_Anomaly'] = z_scores.abs() > 2.5

iso_forest = IsolationForest(contamination=0.025, random_state=42)
filtered_df['Isolation_Anomaly'] = iso_forest.fit_predict(filtered_df[[anomaly_metric]])
filtered_df['Isolation_Anomaly'] = filtered_df['Isolation_Anomaly'].apply(lambda x: x == -1)  # Convert to boolean

st.markdown(f"<h3 style='color:#FAFAFA;'>Scatter Plot for {anomaly_metric.replace('_', ' ')}</h3>", unsafe_allow_html=True)
anomaly_fig = px.scatter(
    filtered_df,
    x='Country',
    y=anomaly_metric,
    color='Z_Anomaly',
    symbol='Isolation_Anomaly',
    title=f"Anomaly Detection for {anomaly_metric.replace('_', ' ')}",
    hover_name='Country',
    labels={
        anomaly_metric: anomaly_metric.replace('_', ' '),
        'Z_Anomaly': 'Z-Score Anomaly',
        'Isolation_Anomaly': 'Isolation Forest Anomaly'
    },
    color_discrete_map={True: 'red', False: 'blue'},
    symbol_map={True: 'x', False: 'circle'}
)
anomaly_fig.update_layout(
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
    font=dict(color="white"),
    margin={"r": 0, "t": 50, "l": 0, "b": 0}
)
st.plotly_chart(anomaly_fig, use_container_width=True)


st.markdown(f"<h3 style='color:#FAFAFA;'>Detected Anomalies for {anomaly_metric.replace('_', ' ')}</h3>", unsafe_allow_html=True)
z_anomalies = filtered_df[filtered_df['Z_Anomaly']]
isolation_anomalies = filtered_df[filtered_df['Isolation_Anomaly']]
col1, col2 = st.columns(2)
with col1:
    st.write("**Z-Score Anomalies:**")
    st.dataframe(z_anomalies[['Country', anomaly_metric, 'Z_Score']])
with col2:
    st.write("**Isolation Forest Anomalies:**")
    st.dataframe(isolation_anomalies[['Country', anomaly_metric]])

