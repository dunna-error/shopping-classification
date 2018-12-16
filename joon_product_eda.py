import pandas as pd
dataset_dir = '/workspace/dataset/'

df = pd.read_pickle(dataset_dir + 'df_word_count.pkl')

print(df.head(50))