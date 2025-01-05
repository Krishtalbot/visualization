import pandas as pd

corruption_df = pd.read_csv('datasets/corruption.csv')
gdp_df = pd.read_csv('datasets/gdp_all.csv')
happiness_df = pd.read_csv('datasets/happiness_index.csv')
continent_df = pd.read_csv('datasets/continent.csv')

corruption_df.set_index('Country', inplace=True)
gdp_df.set_index('Country', inplace=True)
happiness_df.set_index('Country', inplace=True)
continent_df.set_index('Country', inplace=True)

df = corruption_df.merge(gdp_df,on='Country').merge(happiness_df,on='Country').merge(continent_df,on='Country')

df['GDP'] = df['GDP'].replace({r'\$':'', ',':''}, regex=True).astype(float)
df['Unemployment rate'] = df['Unemployment rate'].str.replace('%', '').astype(float)

df['Unemployment rate'] = df['Unemployment rate'].fillna(0)

numeric_columns = ['CPI score', 'Rank CPI', 'Standard error of CPI', 
                   'Happiness score', 'Standard error of happiness score', 
                   'Freedom to make life choices']

df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

df.rename(columns={
    'CPI score': 'CPI_Score',
    'Rank CPI': 'CPI_Rank',
    'Standard error of CPI': 'CPI_Standard_Error',
    'Unemployment rate': 'Unemployment_Rate',
    'Happiness score': 'Happiness_Score',
    'Standard error of happiness score': 'Happiness_Standard_Error',
    'Freedom to make life choices': 'Freedom_Index'
}, inplace=True)

df.to_csv('datasets/merged.csv')