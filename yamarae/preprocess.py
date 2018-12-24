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

opt = Option('./config.json')


class Preprocessor:
    def __init__(self):
        self.df_columns = ['bcateid', 'mcateid', 'scateid', 'dcateid', 'brand', 'maker', 'model',
                           'product', 'price', 'updttm', 'pid']
        self.data_path_list = opt.train_data_list

    def make_df(self):
        df = pd.DataFrame(columns=columns)
        for number in range(1, 10):
            hdf5_file = "train.chunk." + "0" + str(number)
            hdf5_data = h5py.File(data_path + hdf5_file, 'r')
            chunk_df = make_df(hdf5_data, 1000000)
            df = df.append(chunk_df)
        file_name = data_path + "train_sample.csv"
        df.to_csv(file_name, index=False, header=True)

    def make_dict(self):
        pass

    def make_b2v_model(self):
        pass

    def make_d2v_model(self):
        pass


if __name__ == '__main__':
    preprocessor = Preprocessor()
    fire.Fire({'make_df': preprocessor.make_df,
               'make_dict': preprocessor.make_dict,
               'make_b2v_model': preprocessor.make_b2v_model,
               'make_d2v_model': preprocessor.make_d2v_model})
