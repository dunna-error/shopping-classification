import pandas as pd
from misc import get_logger, Option
import re
import pickle
from collections import Counter
import seaborn as sns

opt = Option('./config.json')
dataset_dir = '/workspace/dataset/'

sr_product = pd.read_pickle(dataset_dir+'sr_product.pkl')
# print(sr_product.sample(100))
# #
corpus = " ".join(sr_product.tolist())
noise_pattern = r'\[|\]|/|\(|\)|-|\+|_|=|:|&|\d%'
en_stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
                "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
                "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
                "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
                "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
                "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
                "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
                "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

# del sr_product

# char 단위 pre-processing
corpus = re.sub(noise_pattern, repl=" ", string=corpus)
corpus = re.sub(r'( )+', repl=" ", string=corpus)
# #
# with open(dataset_dir + 'product_corpus.pkl', 'wb') as _file:
#     pickle.dump(corpus, _file)

# with open(dataset_dir + 'product_corpus.pkl', 'rb') as _file:
#     corpus = pickle.load(_file)
#
# print(corpus[:10])
# token 단위 pre-processing
corpus = corpus.split(" ")
corpus = [token for token in corpus if (token not in en_stopwords) and (not token.isdigit())]

# with open(dataset_dir + 'product_corpus_list.pkl', 'wb') as _file:
#     pickle.dump(corpus, _file)
#
# print(corpus[:10])

'''
with open(dataset_dir + 'product_corpus_list.pkl', 'rb') as _file:
    corpus = pickle.load(_file)

df_corpus = pd.DataFrame(corpus, columns=['token'])
df_corpus.to_pickle(dataset_dir+'df_corpus.pkl')
print(df_corpus.head(10))
del corpus

df_corpus = df_corpus.groupby('token').size().to_frame()
df_corpus.columns = ['cnt']
df_corpus = df_corpus.reset_index()
df_corpus = df_corpus.sort_values(by='cnt', ascending=False)
df_corpus.to_pickle(dataset_dir+'df_count.pkl')
print(df_corpus.head(10))
# print(corpus[:10])

'''

df = pd.read_pickle(dataset_dir+'df_count.pkl')

print(df.head(50))
print(df.tail(50))

'''
>> train + inference
step-1) 정규표현식을 이용한 1차 noise 제거 (char 단위)
step-2) es nori를 이용한 형태소 분석
step-3) stop_words를 이용한 2차 noise 제거 (token 단위)

>> train EDA
step-4) 
step-5) stop words 보강
step-6) 사전 보강

'''