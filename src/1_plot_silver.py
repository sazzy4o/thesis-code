#%%
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt

def load_jsonl_df(path):
    with open(path) as f:
        rows = [json.loads(l) for l in f.readlines()]
    return pd.DataFrame(rows)
    
#%%
train_df = load_jsonl_df(Path('../data/squad_silver/train-0.8.jsonl'))
train_df['question_type'] = train_df['question_type'].apply(lambda x: x if x != 'Unknown' else 'Discarded')
# %%
df_counts = train_df.question_type.value_counts().to_frame()
df_counts.sort_index(inplace=True)
df_counts_main = df_counts.iloc[[i for i in range(len(df_counts)) if i !=3], :]
df_counts_discard = df_counts.iloc[3:4]
df_counts = pd.concat([df_counts_main, df_counts_discard])
df_counts['color'] = '#0343df'
df_counts.loc['Discarded', 'color'] = '#e50000'
df_counts.plot(kind='bar', color=df_counts['color'].values)
plt.bar(df_counts.index.values, df_counts.question_type.values, color=df_counts['color'].values)
plt.xlabel('Question Type')
plt.ylabel('Number of Training Examples')
plt.legend().remove()
plt.savefig('squad_silver_train_counts.svg', bbox_inches='tight')
plt.show()

# %%
