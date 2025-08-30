#don't run this
import pandas as pd

df = pd.read_csv('all_haiku.csv')

'''df = df.drop(columns=['source', 'hash'])
df['0'] = df['0']+"*"+df['1']+"*"+df['2']+"**"
df = df.drop(columns=['1','2'])'''
#df.drop(columns=' ', inplace=True)
df.to_csv('all_haiku.csv', index=False)
print(df.head())
