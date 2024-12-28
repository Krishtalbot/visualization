import numpy as np
import pandas as pd


corruption_df = pd.read_csv('datasets/corruption.csv')
gdp_df = pd.read_csv('datasets/gdp_all.csv')
happiness_df = pd.read_csv('datasets/happiness_index.csv')

corruption_df.set_index('Country', inplace=True)
gdp_df.set_index('Country', inplace=True)
happiness_df.set_index('Country', inplace=True)

df = corruption_df.join(gdp_df, how='outer')

print(df.head())