import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Input, Reshape, Dropout, Activation, LSTMCell
from keras.layers.merge import dot
from misc import get_logger, Option
import tensorflow as tf
import os
opt = Option('./config.json')


class Decoder:
    def __init__(self):
        self.logger = get_logger('Decoder')
        self.encoder_vector_size = opt.encoder_vector_size     #TODO

        self.checkpoint_dir = None
        self.n_hidden = None
        self.n_b_cate = None
        self.n_m_cate = None
        self.n_s_cate = None
        self.n_d_cate = None

        self.encoder_inputs = tf.placeholder(tf.int32, shape=[None, 4], name="encoder_inputs")
        self.decoder_inputs = tf.placeholder(tf.int32, shape=[None, 4], name="decoder_inputs")
        self.decoder_outputs = tf.placeholder(tf.int64, [None, 4], name="decoder_outputs")

        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'model')
        self.global_step = tf.Variable(0, name = 'global_step', trainable = False)

        # self.dec_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden)
        # self.dec_cell.call(self.Wxbh, state=)
        # self.target_weights = tf.placeholder(tf.float32, [None, None], name="target_weights")
        self.dec_cell = tf.Variable([None, self.n_hidden], name='dec_cell')

        self.Whh = tf.Variable(tf.float32, [self.n_hidden, self.n_hidden], name="Whh")
        self.bh = tf.Variable(tf.float32, [self.n_hidden, self.n_hidden], name="bh")

        self.Wxbh = tf.Variable(tf.float32, [self.n_b_cate, self.n_hidden], name="Wxbh")
        self.Whyb = tf.Variable(tf.float32, [self.n_hidden, self.n_b_cate], name="Whyb")
        self.byb = tf.Variable(tf.float32, [1, self.n_b_cate], name="Whyb")
        self.ybHat = tf.sigmoid(tf.add(tf.matmul(X, W), B))  # 4x1

        self.dec_cell = tf.add(tf.)


        self.Wxmh = tf.Variable(tf.float32, [self.n_m_cate, self.n_hidden], name="Wxmh")
        self.Wxsh = tf.Variable(tf.float32, [self.n_s_cate, self.n_hidden], name="Wxsh")
        self.Wxdh = tf.Variable(tf.float32, [self.n_d_cate, self.n_hidden], name="Wxdh")

        self.Whym = tf.Variable(tf.float32, [self.n_hidden, self.n_m_cate], name="Whym")
        self.bym = tf.Variable(tf.float32, [1, self.n_m_cate], name="Whym")
        self.Whys = tf.Variable(tf.float32, [self.n_hidden, self.n_s_cate], name="Whys")
        self.bys = tf.Variable(tf.float32, [1, self.n_s_cate], name="Whys")
        self.Whyd = tf.Variable(tf.float32, [self.n_hidden, self.n_d_cate], name="Whyd")
        self.byd  = tf.Variable(tf.float32, [1, self.n_d_cate], name="Whyd")


    def get_model(self, num_classes, activation='sigmoid'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1



        with tf.device('/gpu:0'):

            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')


            lstm_cell = LSTMCell(128)
            b_input = Input(shape=(self.encoder_vector_size,))
            # b_output = Dense(num_classes, activation=activation)(relu)
            # b_model = Model(inputs=[b_input, start_input], outputs=b_output)

            t_uni = Input((max_len,), name="input_1")
            t_uni_embd = embd(t_uni)  # token

            w_uni = Input((max_len,), name="input_2")
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1)
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat)

            embd_out = Dropout(rate=0.5)(uni_embd)
            relu = Activation('relu', name='relu1')(embd_out)
            outputs = Dense(num_classes, activation=activation)(relu)
            model = Model(inputs=[t_uni, w_uni], outputs=outputs)
            optm = keras.optimizers.Nadam(opt.lr)
            model.compile(
                loss='binary_crossentropy',
                optimizer=optm,
                metrics=[top1_acc]
            )
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model


class TextOnly:
    def __init__(self):
        self.logger = get_logger('textonly')

    def get_model(self, num_classes, activation='sigmoid'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        with tf.device('/gpu:0'):
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')

            t_uni = Input((max_len,), name="input_1")
            t_uni_embd = embd(t_uni)  # token

            w_uni = Input((max_len,), name="input_2")
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1)
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat)

            embd_out = Dropout(rate=0.5)(uni_embd)
            relu = Activation('relu', name='relu1')(embd_out)
            outputs = Dense(num_classes, activation=activation)(relu)
            model = Model(inputs=[t_uni, w_uni], outputs=outputs)
            optm = keras.optimizers.Nadam(opt.lr)
            model.compile(loss='binary_crossentropy',
                        optimizer=optm,
                        metrics=[top1_acc])
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model