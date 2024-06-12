#don't run this
import pandas as pd

df = pd.read_csv('all_haiku.csv')

df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
df.to_csv('all_haiku.csv', index=False)

print(df.head())
