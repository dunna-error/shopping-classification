# -*- coding: utf-8 -*-
# Copyright 2017 Kakao, Recommendation Team
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
Edit by 'error모르겠다' Team
Author : yoonkt200@gmail.com, joonable2@gmail.com
"""

import tensorflow as tf

import keras
from keras.models import Model
from keras.layers import Dense, Input, Dropout

from misc import get_logger, Option
opt = Option('./config.json')


def top1_acc(x, y):
    return keras.metrics.top_k_categorical_accuracy(x, y, k=1)


class ShopNet:
    def __init__(self):
        self.logger = get_logger('shopnet')
        self.b_input_size = 2352
        self.m_input_size = 2352
        self.s_input_size = 2352
        self.d_input_size = 2352

    def get_b_model(self, num_classes):
        with tf.device('/gpu:0'):
            optm = keras.optimizers.Nadam(opt.lr)
            i_layer = Input(shape=(self.b_input_size,))
            h_layer = Dense(2048, activation='relu')(i_layer)
            h_layer = Dense(1024, activation='relu')(h_layer)
            h_layer = Dense(512, activation='relu')(h_layer)
            h_layer = Dense(256, activation='relu')(h_layer)
            o_layer = Dense(num_classes, activation='softmax')(h_layer)
            model = Model(inputs=i_layer, outputs=o_layer)
            model.compile(loss='categorical_crossentropy',
                          optimizer=optm,
                          metrics=[top1_acc])
        return model

    def get_m_model(self, num_classes):
        with tf.device('/gpu:0'):
            optm = keras.optimizers.Nadam(opt.lr)
            i_layer = Input(shape=(self.m_input_size,))
            h_layer = Dense(2048, activation='relu')(i_layer)
            h_layer = Dropout(0.15)(h_layer)
            h_layer = Dense(1024, activation='relu')(h_layer)
            h_layer = Dense(512, activation='relu')(h_layer)
            o_layer = Dense(num_classes, activation='softmax')(h_layer)
            model = Model(inputs=i_layer, outputs=o_layer)
            model.compile(loss='categorical_crossentropy',
                          optimizer=optm,
                          metrics=[top1_acc])
        return model

    def get_s_model(self, num_classes):
        with tf.device('/gpu:0'):
            optm = keras.optimizers.Nadam(opt.lr)
            i_layer = Input(shape=(self.s_input_size,))
            h_layer = Dense(2048, activation='relu')(i_layer)
            h_layer = Dense(1024, activation='relu')(h_layer)
            h_layer = Dense(512, activation='relu')(h_layer)
            o_layer = Dense(num_classes, activation='softmax')(h_layer)
            model = Model(inputs=i_layer, outputs=o_layer)
            model.compile(loss='categorical_crossentropy',
                          optimizer=optm,
                          metrics=[top1_acc])
        return model

    def get_d_model(self, num_classes):
        with tf.device('/gpu:0'):
            optm = keras.optimizers.Nadam(opt.lr)
            i_layer = Input(shape=(self.d_input_size,))
            h_layer = Dense(2048, activation='relu')(i_layer)
            h_layer = Dropout(0.15)(h_layer)
            h_layer = Dense(1024, activation='relu')(h_layer)
            h_layer = Dense(512, activation='relu')(h_layer)
            o_layer = Dense(num_classes, activation='softmax')(h_layer)
            model = Model(inputs=i_layer, outputs=o_layer)
            model.compile(loss='categorical_crossentropy',
                          optimizer=optm,
                          metrics=[top1_acc])
        return model