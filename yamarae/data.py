import os
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import pickle
import datetime
import time
import traceback
# from collections import Counter
from multiprocessing import Pool

import gensim
import tqdm
import fire
import h5py
import numpy as np
# import mmh3
import six
# from keras.utils.np_utils import to_categorical
from six.moves import cPickle

from misc import get_logger, Option
opt = Option('./config.json')


class Reader(object):
    def __init__(self, data_path_list, div, begin_offset, end_offset):
        self.div = div
        self.data_path_list = data_path_list
        self.begin_offset = begin_offset
        self.end_offset = end_offset

    def is_range(self, i):
        if self.begin_offset is not None and i < self.begin_offset:
            return False
        if self.end_offset is not None and self.end_offset <= i:
            return False
        return True

    def get_size(self):
        offset = 0
        count = 0
        for data_path in self.data_path_list:
            h = h5py.File(data_path, 'r')
            sz = h[self.div]['pid'].shape[0]
            if not self.begin_offset and not self.end_offset:
                offset += sz
                count += sz
                continue
            if self.begin_offset and offset + sz < self.begin_offset:
                offset += sz
                continue
            if self.end_offset and self.end_offset < offset:
                break
            for i in range(sz):
                if not self.is_range(offset + i):
                    continue
                count += 1
            offset += sz
        return count

    def get_class(self, h, i):
        b = h['bcateid'][i]
        m = h['mcateid'][i]
        s = h['scateid'][i]
        d = h['dcateid'][i]
        return '%s>%s>%s>%s' % (b, m, s, d)

    def generate(self):
        offset = 0
        for data_path in self.data_path_list:
            h = h5py.File(data_path, 'r')[self.div]
            sz = h['pid'].shape[0]
            if self.begin_offset and offset + sz < self.begin_offset:
                offset += sz
                continue
            if self.end_offset and self.end_offset < offset:
                break
            for i in range(sz):
                if not self.is_range(offset + i):
                    continue
                class_name = self.get_class(h, i)
                yield h['pid'][i], class_name, h, i
            offset += sz

    def get_y_vocab(self, data_path):
        y_vocab = {}
        h = h5py.File(data_path, 'r')[self.div]
        sz = h['pid'].shape[0]
        for i in tqdm.tqdm(range(sz), mininterval=1):
            class_name = self.get_class(h, i)
            if class_name not in y_vocab:
                y_vocab[class_name] = len(y_vocab)
        return y_vocab


def preprocessing(data):
    try:
        cls, data_path_list, div, out_path, begin_offset, end_offset = data
        data = cls()
        data.load_y_vocab()
        data.preprocessing(data_path_list, div, begin_offset, end_offset, out_path)
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def build_y_vocab(data):
    try:
        data_path, div = data
        reader = Reader([], div, None, None)
        y_vocab = reader.get_y_vocab(data_path)
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))
    return y_vocab


class Data:
    y_vocab_path = './data/y_vocab.cPickle' if six.PY2 else './data/y_vocab.py3.cPickle'
    price_quantile_dict_path = './data/price_quantile_dict.pickle'
    time_aging_dict_path = './data/time_aging_dict.pickle'
    valid_tag_dict_path = './data/valid_tag_dict.pickle'
    b2v_dict_path = './data/b2v_dict.pickle'
    b2v_model_path = './data/b2v.model'
    tmp_chunk_tpl = 'tmp/base.chunk.%s'

    def __init__(self):
        self.logger = get_logger('data')
        self.price_quantile_dict = pickle.load(open(self.price_quantile_dict_path, 'rb'))
        self.time_aging_dict = pickle.load(open(self.time_aging_dict_path, 'rb'))
        self.valid_tag_dict = pickle.load(open(self.valid_tag_dict_path, 'rb'))
        self.b2v_dict = pickle.load(open(self.b2v_dict_path, 'rb'))
        self.b2v_model = gensim.models.Word2Vec.load(self.b2v_model_path)

    def load_y_vocab(self):
        self.y_vocab = cPickle.loads(open(self.y_vocab_path, 'rb').read())

    def build_y_vocab(self):
        pool = Pool(opt.num_workers)
        try:
            rets = pool.map_async(build_y_vocab,
                                  [(data_path, 'train')
                                   for data_path in opt.train_data_list]).get(99999999)
            pool.close()
            pool.join()
            y_vocab = set()
            for _y_vocab in rets:
                for k in six.iterkeys(_y_vocab):
                    y_vocab.add(k)
            self.y_vocab = {y: idx for idx, y in enumerate(y_vocab)}
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise
        self.logger.info('size of y vocab: %s' % len(self.y_vocab))
        cPickle.dump(self.y_vocab, open(self.y_vocab_path, 'wb'), 2)

    def _split_data(self, data_path_list, div, chunk_size):
        total = 0
        for data_path in data_path_list:
            h = h5py.File(data_path, 'r')
            sz = h[div]['pid'].shape[0]
            total += sz
        chunks = [(i, min(i + chunk_size, total))
                  for i in range(0, total, chunk_size)]
        return chunks

    def preprocessing(self, data_path_list, div, begin_offset, end_offset, out_path):
        self.div = div
        reader = Reader(data_path_list, div, begin_offset, end_offset)
        rets = []
        for pid, label, h, i in reader.generate():
            y, x = self.parse_data(label, h, i, div)
            if y is None:
                continue
            rets.append((pid, y, x))
        self.logger.info('sz=%s' % (len(rets)))
        open(out_path, 'wb').write(cPickle.dumps(rets, 2))
        self.logger.info('%s ~ %s done. (size: %s)' % (begin_offset, end_offset, end_offset - begin_offset))

    def _preprocessing(self, cls, data_path_list, div, chunk_size):
        chunk_offsets = self._split_data(data_path_list, div, chunk_size)
        num_chunks = len(chunk_offsets)
        self.logger.info('split data into %d chunks, # of classes=%s' % (num_chunks, len(self.y_vocab)))
        pool = Pool(opt.num_workers)
        try:
            pool.map_async(preprocessing, [(cls,
                                            data_path_list,
                                            div,
                                            self.tmp_chunk_tpl % cidx,
                                            begin,
                                            end)
                                           for cidx, (begin, end) in enumerate(chunk_offsets)]).get(9999999)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise
        return num_chunks

    def _check_valid_keyword(self, keyword):
        if keyword is np.nan:
            return False
        else:
            if any(exception in keyword for exception in opt.exception_word_list):
                return False
            elif keyword in opt.exception_tag_list:
                return False
            else:
                return True

    def _get_encoded_tag(self, key):
        if key in self.valid_tag_dict:
            return self.valid_tag_dict[key]
        else:
            return self.valid_tag_dict["-1"]

    def _get_encoded_raw_tag(self, key):
        if key in self.b2v_dict:
            return self.b2v_dict[key]
        else:
            return self.b2v_dict["-1"]

    def _get_trimed_tag(self, brand, maker, raw_flag=False):
        brand_valid = self._check_valid_keyword(brand)
        maker_valid = self._check_valid_keyword(maker)
        if brand_valid:
            if raw_flag:
                return self._get_encoded_raw_tag(brand)
            else:
                return self._get_encoded_tag(brand)
        else:
            if maker_valid:
                if raw_flag:
                    return self._get_encoded_raw_tag(brand)
                else:
                    return self._get_encoded_tag(brand)
        return self.valid_tag_dict["-1"]

    def _get_b2v(self, tag):
        if tag in self.b2v_model.wv.vocab:
            return self.b2v_model.wv[tag]
        else:
            return np.zeros(opt.b2v_feat_len)

    def _get_price_level(self, price):
        if price == -1:
            return 2
        elif price == np.nan:
            return 2
        else:
            if price < self.price_quantile_dict['quantile_1']:
                return 1
            elif price > self.price_quantile_dict['quantile_2']:
                return 3
            else:
                return 2

    def _get_unix_time_aging(self, stand_unix_time, time_str, div):
        date_str = time_str[2:10]
        unix_time = time.mktime(datetime.datetime.strptime(date_str, "%Y%m%d").timetuple())
        aging = stand_unix_time - unix_time
        a_min = self.time_aging_dict[div]['min']
        a_max = self.time_aging_dict[div]['max']
        norm_aging = (aging - a_min) / (a_max - a_min)
        return norm_aging

    def parse_data(self, label, h, i, div):
        Y = self.y_vocab.get(label)
        if Y is None and self.div in ['dev', 'test']:
            Y = -1

        tag = self._get_trimed_tag(h['brand'][i].decode('utf-8'), h['maker'][i].decode('utf-8'), raw_flag=False)
        raw_tag = self._get_trimed_tag(h['brand'][i].decode('utf-8'), h['maker'][i].decode('utf-8'), raw_flag=True)
        b2v = self._get_b2v(str(raw_tag))
        img_feat = h['img_feat'][i]
        price_lev = self._get_price_level(h['price'][i])
        div_stand_unix_time = self.time_aging_dict[div]['stand_unix_time']
        aging = self._get_unix_time_aging(div_stand_unix_time, str(h['updttm'][i]), div)
        # word_feat = h['']
        # return Y, (tag, b2v, img_feat, price_lev, aging, word_feat)
        return Y, (tag, b2v, img_feat, price_lev, aging)

    def create_dataset(self, g, size):
        g.create_dataset('y', (size,), chunks=True, dtype=np.int32) # encoded vocab
        g.create_dataset('tag', (size,), chunks=True, dtype=np.int32) # 1 ~ 15000
        g.create_dataset('b2v', (size, opt.b2v_feat_len), chunks=True, dtype=np.float32) # [0, 0.8, ... 0.9, 0.1]
        g.create_dataset('img_feat', (size, opt.img_feat_len), chunks=True, dtype=np.float32) # [0, 0.8, ... 0.9, 0.1]
        g.create_dataset('price_lev', (size,), chunks=True, dtype=np.int32) # 1,2,3
        g.create_dataset('aging', (size,), chunks=True, dtype=np.float32)
        # g.create_dataset('word_feat')
        g.create_dataset('pid', (size,), chunks=True, dtype='S12')

    def init_chunk(self, chunk_size):
        chunk = {}
        chunk['y'] = np.zeros(shape=chunk_size, dtype=np.int32)
        chunk['tag'] = np.zeros(shape=chunk_size, dtype=np.int32)
        chunk['b2v'] = np.zeros(shape=(chunk_size, opt.b2v_feat_len), dtype=np.float32)
        chunk['img_feat'] = np.zeros(shape=(chunk_size, opt.img_feat_len), dtype=np.float32)
        chunk['price_lev'] = np.zeros(shape=chunk_size, dtype=np.int32)
        chunk['aging'] = np.zeros(shape=chunk_size, dtype=np.float32)
        # chunk['word_feat'] = np.zeros()
        chunk['pid'] = []
        chunk['num'] = 0
        return chunk

    def copy_chunk(self, dataset, chunk, offset, with_pid_field=False):
        num = chunk['num']
        dataset['y'][offset:offset + num] = chunk['y'][:num]
        dataset['tag'][offset:offset + num] = chunk['tag'][:num]
        dataset['b2v'][offset:offset + num, :] = chunk['b2v'][:num]
        dataset['img_feat'][offset:offset + num, :] = chunk['img_feat'][:num]
        dataset['price_lev'][offset:offset + num] = chunk['price_lev'][:num]
        dataset['aging'][offset:offset + num] = chunk['aging'][:num]
        # dataset['word_feat'][offset:offset + num] = chunk['word_feat'][:num]
        if with_pid_field:
            dataset['pid'][offset:offset + num] = chunk['pid'][:num]

    def get_train_indices(self, size, train_ratio):
        train_indices = np.random.rand(size) < train_ratio
        train_size = int(np.count_nonzero(train_indices))
        return train_indices, train_size

    def make_db(self, data_name, output_dir='data/train', train_ratio=0.8):
        if data_name == 'train':
            div = 'train'
            data_path_list = opt.train_data_list
        elif data_name == 'dev':
            div = 'dev'
            data_path_list = opt.dev_data_list
        elif data_name == 'test':
            div = 'test'
            data_path_list = opt.test_data_list
        else:
            assert False, '%s is not valid data name' % data_name

        all_train = train_ratio >= 1.0
        all_dev = train_ratio == 0.0

        np.random.seed(17)
        self.logger.info('make database from data(%s) with train_ratio(%s)' % (data_name, train_ratio))

        self.load_y_vocab()
        num_input_chunks = self._preprocessing(Data,
                                               data_path_list,
                                               div,
                                               chunk_size=opt.chunk_size)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        data_fout = h5py.File(os.path.join(output_dir, 'data.h5py'), 'w')
        meta_fout = open(os.path.join(output_dir, 'meta'), 'wb')

        reader = Reader(data_path_list, div, None, None)
        tmp_size = reader.get_size()
        train_indices, train_size = self.get_train_indices(tmp_size, train_ratio)

        dev_size = tmp_size - train_size
        if all_dev:
            train_size = 1
            dev_size = tmp_size
        if all_train:
            dev_size = 1
            train_size = tmp_size

        train = data_fout.create_group('train')
        dev = data_fout.create_group('dev')
        self.create_dataset(train, train_size)
        self.create_dataset(dev, dev_size)
        self.logger.info('train_size ~ %s, dev_size ~ %s' % (train_size, dev_size))

        sample_idx = 0
        dataset = {'train': train, 'dev': dev}
        num_samples = {'train': 0, 'dev': 0}
        chunk_size = opt.db_chunk_size
        chunk = {'train': self.init_chunk(chunk_size),
                 'dev': self.init_chunk(chunk_size)}
        chunk_order = list(range(num_input_chunks))
        np.random.shuffle(chunk_order)
        
        for input_chunk_idx in chunk_order:
            path = os.path.join(self.tmp_chunk_tpl % input_chunk_idx)
            self.logger.info('processing %s ...' % path)
            data = list(enumerate(cPickle.loads(open(path, 'rb').read())))
            np.random.shuffle(data)
            for data_idx, (pid, y, x) in data:
                if y is None:
                    continue
                tag, b2v, img_feat, price_lev, aging = x
                is_train = train_indices[sample_idx + data_idx]
                if all_dev:
                    is_train = False
                if all_train:
                    is_train = True
                c = chunk['train'] if is_train else chunk['dev']
                idx = c['num']
                c['y'][idx] = y
                c['tag'][idx] = tag
                c['b2v'][idx] = b2v
                c['img_feat'][idx] = img_feat
                c['price_lev'][idx] = price_lev
                c['aging'][idx] = aging
                # c['word_feat'][idx] = word_feat
                c['num'] += 1
                if not is_train:
                    c['pid'].append(np.string_(pid))
                for t in ['train', 'dev']:
                    if chunk[t]['num'] >= chunk_size:
                        self.copy_chunk(dataset[t], chunk[t], num_samples[t],
                                        with_pid_field=t == 'dev')
                        num_samples[t] += chunk[t]['num']
                        chunk[t] = self.init_chunk(chunk_size)
            sample_idx += len(data)
        for t in ['train', 'dev']:
            if chunk[t]['num'] > 0:
                self.copy_chunk(dataset[t], chunk[t], num_samples[t],
                                with_pid_field=t == 'dev')
                num_samples[t] += chunk[t]['num']

        data_fout.close()
        meta = {'y_vocab': self.y_vocab}
        meta_fout.write(cPickle.dumps(meta, 2))
        meta_fout.close()

        self.logger.info('# of classes: %s' % len(meta['y_vocab']))
        self.logger.info('# of samples on train: %s' % num_samples['train'])
        self.logger.info('# of samples on dev: %s' % num_samples['dev'])
        self.logger.info('data: %s' % os.path.join(output_dir, 'data.h5py'))
        self.logger.info('meta: %s' % os.path.join(output_dir, 'meta'))


if __name__ == '__main__':
    data = Data()
    fire.Fire({'make_db': data.make_db,
               'build_y_vocab': data.build_y_vocab})