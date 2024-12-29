import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import os

st.set_page_config(page_title='Data Visualization', page_icon='📊', layout='wide')
st.markdown('<style>div.block-container{padding-top: 1rem;}</style>', unsafe_allow_html=True)
st.title('Visualization of Corruption, GDP and Happiness')

fl = st.file_uploader(':file_folder: Upload your dataset', type=(['csv','txt','xlsx','xls']))
if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv(f'datasets/{filename}', encoding='ISO-8859-1')
else:
    os.chdir(r"C:\Users\krish\OneDrive\Desktop\Dev\projects\Corruption_vs_gdp_vs_happiness")
    df = pd.read_csv('datasets/merged.csv', encoding='ISO-8859-1')


