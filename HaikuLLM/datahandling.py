import pandas as pd # type: ignore

df = pd.read_csv('hi.csv')
df = df.drop(columns=['label', 'source', 'RDizzl3_seven'])
df['text'] = df['prompt_name'] + ": " + df['text']
with open('output.txt', 'w') as f:
    for index, row in df.iterrows():
        f.write(f"{row['text']}\n")

