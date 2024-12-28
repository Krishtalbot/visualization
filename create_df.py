import pandas as pd
import functools as ft


corruption_df = pd.read_csv('datasets/corruption.csv')
gdp_df = pd.read_csv('datasets/gdp_all.csv')
happiness_df = pd.read_csv('datasets/happiness_index.csv')

corruption_df.set_index('Country', inplace=True)
gdp_df.set_index('Country', inplace=True)
happiness_df.set_index('Country', inplace=True)

df = corruption_df.merge(gdp_df,on='Country').merge(happiness_df,on='Country')

df.to_csv('datasets/merged.csv')