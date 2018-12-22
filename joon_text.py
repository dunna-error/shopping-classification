import pandas as pd
from misc import get_logger, Option
import re
import pickle
from collections import Counter
import seaborn as sns
import json
import dask.dataframe as dd
from elasticsearch5 import Elasticsearch
from elasticsearch5.helpers import reindex
from itertools import chain

opt = Option('./config.json')
conf = Option('./server_config.json')

dataset_dir = '/workspace/dataset/'
noise_pattern = r'\[|\]|\/|\(|\)|-|\+|_|=|:|&|\d%|!|\?|\*|\^|\$|\@|\{|\}|\"|\'|\\'
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

ko_stopwords = []

# print(''' product 불러오기 ''')
# df = pd.read_pickle(dataset_dir+'df_product.pkl')
# print(df.head(10))
#
#
# print(''' char 단위 pre-processing ''')
# df['product'] = df['product'].apply(lambda x: re.sub(noise_pattern, repl=" ", string=x))
# df['product'] = df['product'].apply(lambda x: re.sub(r'( )+', repl=" ", string=x))
# df.to_pickle(dataset_dir+'df_product_dataset.pkl')
#
#
# # df = pd.read_pickle(dataset_dir+'df_product_dataset.pkl')
# print(df.head())
# print(len(df))
# print('''connect ES''')
# # es = Elasticsearch(conf.es_host)

es = Elasticsearch()

# print(''' upload data to es ''')
# def gen_bulk(row):
#     _head = {"update": {"_id": row.pid, "_type": "_doc", "_index": conf.es_origin_index, "retry_on_conflict": 3}}
#     _body = dict()
#     _body["doc_as_upsert"] = True
#     _body["doc"] = {"pid": row['pid'], "product": row['product']}
#     return [json.dumps(_head), json.dumps(_body)]
#
# body = []
#
# for idx, row in df.iterrows():
#     if (idx+1) % 100000 == 0:
#         print("bulk : {}".format(idx+1))
#     temp = gen_bulk(row)
#     body.extend(temp)
#     if (idx+1) % 10000 == 0:
#         body = "\n".join(body)
#         es.bulk(body)
#         body = []
# body = "\n".join(body)
# es.bulk(body)
# #
#
# print(''' parsing data ''')
# reindex(es, conf.es_origin_index, conf.es_adjv_index)
#
#
print(''' get_parsed_token ''')
def get_mtermvectors(ids):
    body = dict()
    body['ids'] = ids
    body['parameters'] = {"fields": ["product"]}
    res = es.mtermvectors(index=conf.es_adjv_index, doc_type='_doc', body=body)['docs']
    return res


def sort_term_vectors(term_vector):
    if not term_vector:
        return None
    term_dict = {}
    for term, val in term_vector[0].items():
        for pos_info in val['tokens']:
            term_dict[pos_info['position']] = term
    sorted_terms = sorted(term_dict.items())
    sorted_terms = [tup[1] for tup in sorted_terms]
    return sorted_terms


df = pd.read_pickle(dataset_dir+'df_product_dataset.pkl')
df.pid = df.pid.str.strip()
df['term_vectors'] = None

# df = dd.from_pandas(df, npartitions=50)
count_list = list(range(0, len(df), 10000)) + [len(df)]
sorted_term_vectors = list()
sorted_term_vectors.append(es.search(index='nori_with_adjv', size=10000, scroll='1m',
                                     filter_path=['hits.hits._source.sorted_term', 'hits.hits._source.pid']))
scroll_id = sorted_term_vectors[0]['_scroll_id']

# def gen_bulk_2(pid, sorted_term):
#     _head = {"update": {"_id": pid, "_type": "_doc", "_index": conf.es_adjv_index, "retry_on_conflict": 3}}
#     _body = dict()
#     _body["doc_as_upsert"] = True
#     _body["doc"] = {"sorted_term": sorted_term}
#     return [json.dumps(_head), json.dumps(_body)]
#
# for idx in range(len(count_list)-1):
#     print("terms vector : {}".format((idx+1)*10000))
#
#     ids = df.loc[count_list[idx]:count_list[idx+1]].pid.tolist()
#     term_list = get_mtermvectors(ids)
#     ids = []
#     temp = []
#     for x in term_list:
#         ids.append(x['_id'])
#         if 'product' in x['term_vectors'].keys():
#             temp.append([x['term_vectors']['product']['terms']])
#         else:
#             temp.append(None)
#
#     temp = [sort_term_vectors(term_vector) for term_vector in temp]
#     body = [gen_bulk_2(pid, term_vector) for pid, term_vector in zip(ids, temp)]
#     body = [x for x in chain(*body)]
#     body = "\n".join(body)
#     es.bulk(body)

for idx in range(len(count_list)-1):
    print("terms vector : {}".format((idx+1)*10000))

    ids = df.loc[count_list[idx]:count_list[idx+1]].pid.tolist()
    term_list = get_mtermvectors(ids)

    ids = []
    temp = []
    for x in term_list:
        ids.append(x['_id'])
        if 'product' in x['term_vectors'].keys():
            temp.append([x['term_vectors']['product']['terms']])
        else:
            temp.append(None)

    temp = [sort_term_vectors(term_vector) for term_vector in temp]
    body = [gen_bulk_2(pid, term_vector) for pid, term_vector in zip(ids, temp)]
    body = [x for x in chain(*body)]
    body = "\n".join(body)
    es.bulk(body)


#
#
#     # df = df.loc[df.pid.isin(ids), 'term_vectors'].assign(temp)
#     # df['term_vectors'] = df['term_vectors'].mask(df['pid'].isin(ids), temp)
#
#
# print(''' sort_term_vectors ''')
#
#
# df = df.compute()
# df['term_vectors'] = df.term_vectors.apply(lambda term_vector: sort_term_vectors(term_vector))
# df.to_pickle(dataset_dir+'df_product_dataset_with_sorted_term_vector.pkl')

# print(''' token 단위 pre-processing ''')
# def token_preprocessing(term_vector):
#     return [token for token in term_vector if (token not in en_stopwords) and (not token.isdigit())]
#
#
# df['term_vectors'] = df.term_vectors.apply(token_preprocessing)
# df.to_pickle(dataset_dir+'df_product_dataset.pkl')
# df.drop(['product'], axis=1)
#
#
# print(''' word_count ''')
# df_corpus = pd.DataFrame(list(chain(*(df.term_vectors.tolist()))), columns=['token'])
# df_corpus.to_pickle(dataset_dir+'df_corpus.pkl')
#
# print(df_corpus.head(10))
# del df
#
# df_corpus = df_corpus.groupby('token').size().to_frame()
# df_corpus.columns = ['cnt']
# df_corpus = df_corpus.reset_index()
# df_corpus = df_corpus.sort_values(by='cnt', ascending=False)
# df_corpus.to_pickle(dataset_dir+'df_count.pkl')
# print(df_corpus.head(10))
# # print(corpus[:10])
#
# df = pd.read_pickle(dataset_dir+'df_count.pkl')
#
# print(df.head(50))
# print(df.tail(50))


'''
>> [phase A] train + inference
step-1) 정규표현식을 이용한 1차 noise 제거 (char 단위)
step-2) es nori를 이용한 형태소 분석
step-3) stop_words를 이용한 2차 noise 제거 (token 단위)

>> [phase B] A:step-3 train + inference
step-1) word_count 기준으로 dictionary 제작
step-2) integer로 바꾼 후 dataset으로 사용

>> [phase C] A:step-3 이후 EDA
step-1) stop words 보강
step-2) 사전 보강

>> [phase D] modelling
step-1) word2vec
step-2) 
'''