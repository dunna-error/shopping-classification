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

import os
import json
import threading
import pickle

import fire
import h5py
import tqdm
import numpy as np
import six

from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from six.moves import zip, cPickle

from misc import get_logger, Option
from multilayer_network import ShopNet, top1_acc

opt = Option('./config.json')
if six.PY2:
    cate1 = json.loads(open('./data/cate1.json').read())
else:
    cate1 = json.loads(open('./data/cate1.json', 'rb').read().decode('utf-8'))


class Classifier():
    def __init__(self):
        self.logger = get_logger('Classifier')
        self.num_classes = 0
        self.encoded_dict = {
            "price_lev": 3
        }
        self.cate_index_dict = pickle.load(open('./data/cate_index_dict.pickle', 'rb'))
        self.predict_encoder = pickle.load(open('./data/predict_encoder.pickle', 'rb'))
        self.cate_split_index = {"b": 0, "m": 1, "s": 2, "d": 3}
        self.prev_cate_list = {"m": "b", "s": "m", "d": "s"}
        self.b_model = None
        self.m_model = None
        self.s_model = None
        self.d_model = None

    def get_predict_sample_generator(self, ds, batch_size, raise_stop_event=False):
        left, limit = 0, ds['aging'].shape[0]
        while True:
            right = min(left + batch_size, limit)

            # 1-length numerical feature
            X = ds['aging'][left:right]
            X = np.reshape(X, (X.shape[0], 1))

            # list value feature
            for t in ['b2v', 'img_feat', 'd2v']:
                x = ds[t][left:right, :]
                X = np.hstack((X, x))

            # 1-length categorical feature
            for t in ['price_lev']:
                x = ds[t][left:right]
                encoded = np.zeros((x.shape[0], self.encoded_dict[t]))
                for i in range(left - left, right - left):
                    encoded[i][x[i] - 1] = 1
                X = np.hstack((X, encoded))

            # result return shape : (batch_size, 2352+len(prev_cate))
            yield X
            left = right
            if right == limit:
                left = 0
                if raise_stop_event:
                    raise StopIteration

    def get_sample_generator(self, ds, batch_size, raise_stop_event=False, target_cate="b"):
        left, limit = 0, ds['aging'].shape[0]
        while True:
            right = min(left + batch_size, limit)
            filter_idx_list = []
            prev_filter_idx_list = []

            # 1-length numerical feature
            X = ds['aging'][left:right]
            X = np.reshape(X, (X.shape[0], 1))

            # list value feature
            for t in ['b2v', 'img_feat', 'd2v']:
                x = ds[t][left:right, :]
                X = np.hstack((X, x))

            # 1-length categorical feature
            for t in ['price_lev']:
                x = ds[t][left:right]
                encoded = np.zeros((x.shape[0], self.encoded_dict[t]))
                for i in range(left-left, right-left):
                    encoded[i][x[i] - 1] = 1
                X = np.hstack((X, encoded))

            # y label
            if target_cate == "b" or target_cate == "m":
                Y = ds['y'][left:right]
                encoded_Y = np.zeros((Y.shape[0], opt.y_vocab_len[target_cate]))
                for i in range(left - left, right - left):
                    idx = self.cate_index_dict[target_cate][Y[i]]
                    encoded_Y[i][idx] = 1
            else:
                Y = ds['y'][left:right]
                encoded_Y = np.zeros((Y.shape[0], opt.y_vocab_len[target_cate]))
                for i in range(left - left, right - left):
                    idx = self.cate_index_dict[target_cate][Y[i]]
                    if idx == 0:
                        filter_idx_list.append(i)
                    encoded_Y[i][idx] = 1

            rm_list = np.unique(filter_idx_list + prev_filter_idx_list)
            X = np.delete(X, rm_list, 0)
            encoded_Y = np.delete(encoded_Y, rm_list, 0)

            # result return shape : (batch_size, 2352) // (batch_size, len(cate))
            yield X, encoded_Y
            left = right
            if right == limit:
                left = 0
                if raise_stop_event:
                    raise StopIteration

    def write_prediction_result(self, data, pred_b, pred_m, pred_s, pred_d, out_path, readable):
        # # 개발 테스트용 pid order
        # pid_order = []
        # pid_order.extend(data['pid'][::])
        # dev 제출용 pid order
        pid_order = []
        for data_path in opt.dev_data_list:
            h = h5py.File(data_path, 'r')['dev']
            pid_order.extend(h['pid'][::])
        # # 최종 제출용 pid order
        # pid_order = []
        # for data_path in opt.test_data_list:
        #     h = h5py.File(data_path, 'r')['test']
        #     pid_order.extend(h['pid'][::])

        # file write
        rets = {}
        for pid, b, m, s, d in zip(data['pid'], pred_b, pred_m, pred_s, pred_d):
            if six.PY3:
                pid = pid.decode('utf-8')
            tpl = '{pid}\t{b}\t{m}\t{s}\t{d}'
            rets[pid] = tpl.format(pid=pid, b=b, m=m, s=s, d=d)
        no_answer = '{pid}\t-1\t-1\t-1\t-1'
        with open(out_path, 'w') as fout:
            for pid in pid_order:
                if six.PY3:
                    pid = pid.decode('utf-8')
                ans = rets.get(pid, no_answer.format(pid=pid))
                fout.write(ans)
                fout.write('\n')

    def sequential_predict(self, X): # todo: comment and hide list -> one hot vector VS origin vector TEST
        # append b
        b_cate_ohv = self.b_model.predict(X)
        idx_list = np.argmax(b_cate_ohv, axis=1).tolist()
        b_y_list = [self.predict_encoder['b'][i] for i in idx_list]

        # append m
        m_cate_ohv = self.m_model.predict(X)
        idx_list = np.argmax(m_cate_ohv, axis=1).tolist()
        m_y_list = [self.predict_encoder['m'][i] for i in idx_list]

        # append s
        s_cate_ohv = self.s_model.predict(X)
        idx_list = np.argmax(s_cate_ohv, axis=1).tolist()
        s_y_list = [self.predict_encoder['s'][i] for i in idx_list]

        # append d
        d_cate_ohv = self.d_model.predict(X)
        idx_list = np.argmax(d_cate_ohv, axis=1).tolist()
        d_y_list = [self.predict_encoder['d'][i] for i in idx_list]
        return b_y_list, m_y_list, s_y_list, d_y_list

    def predict(self, data_root, model_root, test_root, test_div, out_path, readable=False):
        # load y_sequential model
        b_model_fname = model_root + '/b_model.h5'
        m_model_fname = model_root + '/m_model.h5'
        s_model_fname = model_root + '/s_model.h5'
        d_model_fname = model_root + '/d_model.h5'
        self.b_model = load_model(b_model_fname, custom_objects={'top1_acc': top1_acc})
        self.m_model = load_model(m_model_fname, custom_objects={'top1_acc': top1_acc})
        self.s_model = load_model(s_model_fname, custom_objects={'top1_acc': top1_acc})
        self.d_model = load_model(d_model_fname, custom_objects={'top1_acc': top1_acc})

        test_path = os.path.join(test_root, 'data.h5py')
        test_data = h5py.File(test_path, 'r')
        test = test_data[test_div]
        batch_size = opt.batch_size
        pred_b = []
        pred_m = []
        pred_s = []
        pred_d = []
        test_gen = ThreadsafeIter(self.get_predict_sample_generator(test, batch_size, raise_stop_event=True))
        total_test_samples = test['y'].shape[0]

        with tqdm.tqdm(total=total_test_samples) as pbar:
            for chunk in test_gen:
                total_test_samples = test['y'].shape[0]
                X = chunk
                b,m,s,d = self.sequential_predict(X)
                pred_b.extend(b)
                pred_m.extend(m)
                pred_s.extend(s)
                pred_d.extend(d)
                pbar.update(X[0].shape[0])
        self.write_prediction_result(test, pred_b, pred_m, pred_s, pred_d, out_path, readable=readable)

    def train(self, data_root, out_dir, target_cate):
        data_path = os.path.join(data_root, 'data.h5py')
        data = h5py.File(data_path, 'r')
        self.weight_fname = out_dir + '/' + target_cate + '_weights'
        self.model_fname = out_dir + '/' + target_cate + '_model'
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        self.logger.info('# of classes: %s' % opt.y_vocab_len[target_cate])
        self.num_classes = opt.y_vocab_len[target_cate]

        train = data['train']
        dev = data['dev']

        self.logger.info('# of train samples: %s' % train['y'].shape[0])
        self.logger.info('# of dev samples: %s' % dev['y'].shape[0])

        checkpoint = ModelCheckpoint(self.weight_fname, monitor='loss', save_best_only=True, mode='min', period=1)

        shopnet = ShopNet()
        if target_cate == 'b':
            model = shopnet.get_b_model(self.num_classes)
        elif target_cate == 'm':
            model = shopnet.get_m_model(self.num_classes)
        elif target_cate == 's':
            model = shopnet.get_s_model(self.num_classes)
        elif target_cate == 'd':
            model = shopnet.get_d_model(self.num_classes)
        else:
            self.logger.info('unvalid input category parameter, try again.')
            return
        self.logger.info('getting classifier model of ' + target_cate)

        total_train_samples = train['y'].shape[0]
        train_gen = self.get_sample_generator(train, batch_size=opt.batch_size, target_cate=target_cate)
        self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))

        # total_dev_samples = dev['y'].shape[0]
        # dev_gen = self.get_sample_generator(dev, batch_size=opt.batch_size, target_cate=target_cate)
        # self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))

        model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            # validation_data=dev_gen,
                            # validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=[checkpoint],
                            verbose=1)

        model.load_weights(self.weight_fname) # loads from checkout point if exists
        open(self.model_fname + '.json', 'w').write(model.to_json())
        model.save(self.model_fname + '.h5')


class ThreadsafeIter(object):
    def __init__(self, it):
        self._it = it
        self._lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            return next(self._it)

    def next(self):
        with self._lock:
            return self._it.next()


if __name__ == '__main__':
    clsf = Classifier()
    fire.Fire({'train': clsf.train,
               'predict': clsf.predict})

