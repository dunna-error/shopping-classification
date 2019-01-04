from six.moves import cPickle
import pickle
import fire
import six
import json


def get_reverse_dict(data_dir, data_name):
    meta = cPickle.loads(open(data_dir + data_name + '/meta', 'rb').read())['y_vocab']
    reversed_meta = {}
    for key, value in meta.items():
        reversed_meta[value] = key
    with open('./data/reverse_y_vocab', 'wb') as f:
        pickle.dump(reversed_meta, f, pickle.HIGHEST_PROTOCOL)


def get_cate_index_dict(source_dir):
    cate_dict = json.loads(open(source_dir + 'cate1.json', 'rb').read().decode('utf-8'))
    y_vocab = cPickle.loads(open(source_dir + 'y_vocab.py3.cPickle', 'rb').read())

    cate_index_dict = {} # ex: dict["b"][37] = index
    struct_cate_index_dict = {} # ex: dict["b"][4213] = b's index
    predict_encoder = {}

    # using for make dicts below
    for cate_code, value in cate_dict.items():
        index = 0
        cate_index_dict[cate_code] = {}
        struct_cate_index_dict[cate_code] = {}
        predict_encoder[cate_code] = {}
        for cate_name, cate_number in cate_dict[cate_code].items():
            cate_index_dict[cate_code][cate_number] = index
            index = index + 1

    # using for train encoding
    for y_comb, y_index in y_vocab.items():
        b, m, s, d = y_comb.split(">")
        struct_cate_index_dict["b"][y_index] = cate_index_dict["b"][int(b)]
        struct_cate_index_dict["m"][y_index] = cate_index_dict["m"][int(m)]
        struct_cate_index_dict["s"][y_index] = cate_index_dict["s"][int(s)]
        struct_cate_index_dict["d"][y_index] = cate_index_dict["d"][int(d)]
    with open(source_dir + 'cate_index_dict.pickle', 'wb') as f:
        pickle.dump(struct_cate_index_dict, f, pickle.HIGHEST_PROTOCOL)

    # using for predict encoding
    for cate_code, value in cate_index_dict.items():
        for cate_actual_y, cate_one_hot_idx in cate_index_dict[cate_code].items():
            predict_encoder[cate_code][cate_one_hot_idx] = cate_actual_y
    with open(source_dir + 'predict_encoder.pickle', 'wb') as f:
        pickle.dump(predict_encoder, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    fire.Fire({'reverse_meta': get_reverse_dict, 'cate_index': get_cate_index_dict})