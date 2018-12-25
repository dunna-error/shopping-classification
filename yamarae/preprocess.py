# -*- coding: utf-8 -*-
# Copyright 2018 error모르겠다
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Author : yoonkt200@gmail.com, joonable2@gmail.com
"""

import pickle
import datetime
import time

import fire
import pandas as pd
import numpy as np
import h5py

import gensim

from misc import get_logger, Option
opt = Option('./config.json')


class Preprocessor:
    def __init__(self):
        self.logger = get_logger('preprocessor')
        self.train_df_file = "train_df.csv"
        self.dev_df_file = "dev_df.csv"
        self.test_df_file = "test_df.csv"
        self.train_df_columns = ['bcateid', 'mcateid', 'scateid', 'dcateid', 'brand', 'maker', 'model',
                                 'product', 'price', 'updttm', 'pid']
        self.dev_df_columns = ['brand', 'maker', 'model',
                               'product', 'price', 'updttm', 'pid']
        self.test_df_columns = ['brand', 'maker', 'model',
                                'product', 'price', 'updttm', 'pid']
        self.data_path_list = opt.train_data_list

    def _make_df(self, hdf5_data, data_name):
        idx = len(hdf5_data[data_name]['pid'])
        if data_name == "train":
            df = pd.DataFrame(
                {'bcateid': hdf5_data['train']['bcateid'][0:idx],
                 'mcateid': hdf5_data['train']['mcateid'][0:idx],
                 'scateid': hdf5_data['train']['scateid'][0:idx],
                 'dcateid': hdf5_data['train']['dcateid'][0:idx],
                 'brand': np.array([x.decode('utf-8') for x in hdf5_data['train']['brand'][0:idx]]),
                 'maker': np.array([x.decode('utf-8') for x in hdf5_data['train']['maker'][0:idx]]),
                 'product': np.array([x.decode('utf-8') for x in hdf5_data['train']['product'][0:idx]]),
                 'price': hdf5_data['train']['price'][0:idx],
                 'updttm': hdf5_data['train']['updttm'][0:idx],
                 'pid': hdf5_data['train']['pid'][0:idx]
                 })
            return df
        else:
            df = pd.DataFrame(
                {'brand': np.array([x.decode('utf-8') for x in hdf5_data[data_name]['brand'][0:idx]]),
                 'maker': np.array([x.decode('utf-8') for x in hdf5_data[data_name]['maker'][0:idx]]),
                 'product': np.array([x.decode('utf-8') for x in hdf5_data[data_name]['product'][0:idx]]),
                 'price': hdf5_data[data_name]['price'][0:idx],
                 'updttm': hdf5_data[data_name]['updttm'][0:idx],
                 'pid': hdf5_data[data_name]['pid'][0:idx]
                 })
            return df

    def _get_df(self, data_name):
        path = opt.dataset_path
        if data_name == "train":
            path = path + self.train_df_file
        elif data_name == "dev":
            path = path + self.dev_df_file
        else:
            path = path + self.test_df_file
        df = pd.read_csv(path)
        return df

    def _get_unix_time_aging(self, stand_unix_time, time_str):
        date_str = time_str[2:10]
        unix_time = time.mktime(datetime.datetime.strptime(date_str, "%Y%m%d").timetuple())
        return (stand_unix_time - unix_time)

    def _get_price_quantile_dict(self, df):
        quantile_1 = df[df['price'] != -1]['price'].quantile(0.3)
        quantile_2 = df[df['price'] != -1]['price'].quantile(0.7)
        price_quantile_dict = {"quantile_1": quantile_1, "quantile_2": quantile_2}
        return price_quantile_dict

    def _get_time_aging_dict(self, train_df, dev_df, test_df):
        time_aging_dict = {'train': {'min': 0,
                                     'max': 0,
                                     'stand_unix_time': 0},
                           'dev': {'min': 0,
                                   'max': 0,
                                   'stand_unix_time': 0},
                           'test': {'min': 0,
                                    'max': 0,
                                    'stand_unix_time': 0}}

        # getting standard date each dataset
        train_df['year_mon_day'] = train_df['updttm'].apply(lambda x: int(str(x[2:10])))
        train_recent_updttm_day = train_df['year_mon_day'].max()
        unix_time = time.mktime(datetime.datetime.strptime(str(train_recent_updttm_day), "%Y%m%d").timetuple())
        time_aging_dict['train']['stand_unix_time'] = unix_time + 86400
        dev_df['year_mon_day'] = dev_df['updttm'].apply(lambda x: int(str(x[2:10])))
        dev_recent_updttm_day = dev_df['year_mon_day'].max()
        unix_time = time.mktime(datetime.datetime.strptime(str(dev_recent_updttm_day), "%Y%m%d").timetuple())
        time_aging_dict['dev']['stand_unix_time'] = unix_time + 86400
        test_df['year_mon_day'] = test_df['updttm'].apply(lambda x: int(str(x[2:10])))
        test_recent_updttm_day = test_df['year_mon_day'].max()
        unix_time = time.mktime(datetime.datetime.strptime(str(test_recent_updttm_day), "%Y%m%d").timetuple())
        time_aging_dict['test']['stand_unix_time'] = unix_time + 86400

        # getting time_aging each dataset
        div_time_aging = time_aging_dict['train']['stand_unix_time']
        train_df['unix_time_aging'] = train_df['updttm'].apply(lambda x: self._get_unix_time_aging(div_time_aging, x))
        div_time_aging = time_aging_dict['dev']['stand_unix_time']
        dev_df['unix_time_aging'] = dev_df['updttm'].apply(lambda x: self._get_unix_time_aging(div_time_aging, x))
        div_time_aging = time_aging_dict['test']['stand_unix_time']
        test_df['unix_time_aging'] = test_df['updttm'].apply(lambda x: self._get_unix_time_aging(div_time_aging, x))

        # define min, max to dict
        time_aging_dict['train']['min'] = train_df['unix_time_aging'].min()
        time_aging_dict['train']['max'] = train_df['unix_time_aging'].max()
        time_aging_dict['dev']['min'] = dev_df['unix_time_aging'].min()
        time_aging_dict['dev']['max'] = dev_df['unix_time_aging'].max()
        time_aging_dict['test']['min'] = test_df['unix_time_aging'].min()
        time_aging_dict['test']['max'] = test_df['unix_time_aging'].max()

        return time_aging_dict

    def _check_valid(self, keyword):
        if keyword is np.nan:
            return False
        else:
            if any(exception in keyword for exception in opt.exception_word_list):
                return False
            elif keyword in opt.exception_tag_list:
                return False
            else:
                return True

    def _get_trimed_raw_tag(self, brand, maker):
        brand_valid = self._check_valid(brand)
        maker_valid = self._check_valid(maker)
        if brand_valid:
            return brand
        else:
            if maker_valid:
                return maker
        return "-1"

    def _get_encoded_tag(self, key):
        if key in self.b2v_dict:
            return self.b2v_dict[key]
        else:
            return self.b2v_dict["-1"]

    def _get_trimed_tag(self, brand, maker):
        brand_valid = self._check_valid(brand)
        maker_valid = self._check_valid(maker)
        if brand_valid:
            return self._get_encoded_tag(brand)
        else:
            if maker_valid:
                return self._get_encoded_tag(brand)
        return self.b2v_dict["-1"]

    def _get_b2v_dict(self, df):
        df['tag'] = df.apply(lambda x: self._get_trimed_raw_tag(x['brand'], x['maker']), axis=1)
        tag_list = df['tag'].value_counts().index
        b2v_dict = {}
        idx = 1
        for tag in tag_list:
            b2v_dict[tag] = idx
            idx = idx + 1
        return b2v_dict

    def _train_b2v(self, df):
        # make training dataset for word2vect model
        self.logger.info('make trainset for b2v...')
        df['tag'] = df.apply(lambda x: self._get_trimed_tag(x['brand'], x['maker']), axis=1)
        df = df[['bcateid', 'mcateid', 'scateid', 'dcateid', 'tag']]
        df['tag'] = df['tag'].apply(lambda x: str(x))

        b_list = df.groupby('bcateid')['tag'].apply(lambda x: list(x.unique())).values.tolist()
        train_list = [x for x in b_list if len(x) > 1]
        m_list = df.groupby('mcateid')['tag'].apply(lambda x: list(x.unique())).values.tolist()
        m_list = [x for x in m_list if len(x) > 1]
        s_list = df.groupby('scateid')['tag'].apply(lambda x: list(x.unique())).values.tolist()
        s_list = [x for x in s_list if len(x) > 1]
        d_list = df.groupby('dcateid')['tag'].apply(lambda x: list(x.unique())).values.tolist()
        d_list = [x for x in d_list if len(x) > 1]
        train_list.extend(m_list)
        train_list.extend(s_list)
        train_list.extend(d_list)

        self.logger.info('start train b2v model.')
        # train b2v model
        import logging
        logging.basicConfig(
            format='%(asctime)s : %(levelname)s : %(message)s',
            level=logging.INFO)
        model = gensim.models.Word2Vec(train_list, size=200, window=3, min_count=2, workers=4, iter=15, sg=1)
        return model

    def make_df(self, data_name):
        self.logger.info('make df from hdf5 ' + data_name + ' files.')
        df_columns = []
        data_list = []
        path = ""
        if data_name == "train":
            df_columns = self.train_df_columns
            data_list = opt.train_data_list
            path = path + self.train_df_file
        elif data_name == "dev":
            df_columns = self.dev_df_columns
            data_list = opt.dev_data_list
            path = path + self.dev_df_file
        else:
            df_columns = self.test_df_columns
            data_list = opt.test_data_list
            path = path + self.test_df_file

        df = pd.DataFrame(columns=df_columns)
        for data_path in data_list:
            hdf5_data = h5py.File(data_path, 'r')
            chunk_df = self._make_df(hdf5_data, data_name)
            self.logger.info('appending chunk file, sz:' + str(chunk_df.shape[0]))
            df = df.append(chunk_df, sort=False)
        file_name = opt.dataset_path + path
        df.to_csv(file_name, index=False, header=True)
        self.logger.info('file save complete. ' + 'sz:' + str(df.shape[0]) + ", path:" + file_name)

    def make_dict(self):
        self.logger.info('start make dictionary from dataframes')
        self.logger.info('load dataframes...')
        train_df = self._get_df("train")
        dev_df = self._get_df("dev")
        test_df = self._get_df("test")

        # make price quantile dictionary
        self.logger.info('make price quantile dict...')
        price_quantile_dict = self._get_price_quantile_dict(train_df)

        # make time aging dictionary
        self.logger.info('make time aging dict...')
        time_aging_dict = self._get_time_aging_dict(train_df, dev_df, test_df)

        # make b2v dictionary
        self.logger.info('make b2v dict...')
        b2v_dict = self._get_b2v_dict(train_df)

        # save dicts
        with open('./data/price_quantile_dict.pickle', 'wb') as f:
            pickle.dump(price_quantile_dict, f, pickle.HIGHEST_PROTOCOL)
        with open('./data/time_aging_dict.pickle', 'wb') as f:
            pickle.dump(time_aging_dict, f, pickle.HIGHEST_PROTOCOL)
        with open('./data/b2v_dict.pickle', 'wb') as f:
            pickle.dump(b2v_dict, f, pickle.HIGHEST_PROTOCOL)
        self.logger.info('save dict files complete')

    def make_b2v_model(self):
        self.logger.info('start make brand2vect model.')
        self.logger.info('load train dataframe.')
        train_df = self._get_df("train")
        self.b2v_dict = pickle.load(open('./data/b2v_dict.pickle', 'rb'))
        b2v_model = self._train_b2v(train_df)
        b2v_model.save("./data/b2v.model")

    def make_d2v_model(self):
        pass


if __name__ == '__main__':
    preprocessor = Preprocessor()
    fire.Fire({'make_df': preprocessor.make_df,
               'make_dict': preprocessor.make_dict,
               'make_b2v_model': preprocessor.make_b2v_model
               # 'make_d2v_model': preprocessor.make_d2v_model
               })
