import pandas as pd
import pickle

df_term_vector = pd.concat([
    pd.read_pickle('./data/df_product_train_dataset.pkl'),
    pd.read_pickle('./data/df_product_dev_dataset.pkl'),
    pd.read_pickle('./data/df_product_test_dataset.pkl')],
    axis=0)

term_vector_dict = pd.Series(df_term_vector.pid.values, index=df_term_vector.term_vector).to_dict()

with open('./data/' + 'term_vector_dict.pickle', 'wb') as f:
    pickle.dump(term_vector_dict, f, pickle.HIGHEST_PROTOCOL)