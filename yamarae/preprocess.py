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

import fire
import pandas as pd
import numpy as np
import h5py

from misc import get_logger, Option
opt = Option('./config.json')


class Preprocessor:
    def __init__(self):
        self.logger = get_logger('preprocessor')
        self.df_columns = ['bcateid', 'mcateid', 'scateid', 'dcateid', 'brand', 'maker', 'model',
                           'product', 'price', 'updttm', 'pid']
        self.data_path_list = opt.train_data_list

    def _make_df(self, hdf5_data):
        idx = len(hdf5_data['train']['pid'])
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

    def make_df(self):
        self.logger.info('make df from hdf5 train files.')
        df = pd.DataFrame(columns=self.df_columns)
        for train_data_path in opt.train_data_list:
            hdf5_data = h5py.File(train_data_path, 'r')
            chunk_df = self._make_df(hdf5_data)
            self.logger.info('appending chunk file, sz:' + str(chunk_df.shape[0]))
            df = df.append(chunk_df, sort=False)
        file_name = opt.dataset_path + "train_df.csv"
        df.to_csv(file_name, index=False, header=True)
        self.logger.info('file save complete. ' + 'sz:' + str(df.shape[0]))

    def make_dict(self):
        pass

    def make_b2v_model(self):
        pass

    def make_d2v_model(self):
        pass


if __name__ == '__main__':
    preprocessor = Preprocessor()
    fire.Fire({'make_df': preprocessor.make_df
               # 'make_dict': preprocessor.make_dict,
               # 'make_b2v_model': preprocessor.make_b2v_model,
               # 'make_d2v_model': preprocessor.make_d2v_model
               })
