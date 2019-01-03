import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Input, Reshape, Dropout, Activation, LSTMCell
from keras.layers.merge import dot
# from misc import get_logger, Option
import tensorflow as tf
import os
import numpy as np
# opt = Option('./config.json')


class Decoder:
    def __init__(self):
        # self.logger = get_logger('Decoder')
        # self.encoder_vector_size = opt.encoder_vector_size     #TODO
        self.encoder_vector_size = 128     #TODO

        self.checkpoint_dir = None
        self.encoder_states_size = 1024
        self.n_hidden = 1024
        self.n_b_cate = 50
        self.n_m_cate = 500
        self.n_s_cate = 3000
        self.n_d_cate = 1000
        self.checkpoint_dir = 'nomatterwhatitis'

        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'model')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.time_step = 4
        # self.encoder_states = None      # TODO encoder_states
        self.batch_size = 2      # TODO self.batch_size
        self.encoder_states = tf.Variable(tf.random_normal((self.batch_size, self.encoder_states_size)))  # TODO self.batch_size
        # self.decoder_inputs = tf.placeholder(tf.int32, shape=[None, 4], name="decoder_inputs")# TODO decoder_inputs
        # self.decoder_outputs = tf.placeholder(tf.int64, [None, 4], name="decoder_outputs")      # TODO decoder_outputs
        self.decoder_inputs = tf.Variable([[1,341,2521],[253,1221,256]], trainable=False, name='decoder_inputs')
        self.decoder_outputs = tf.Variable([[1,341,2521,231], [42,253,1221,256]], trainable=False, name='decoder_outputs', dtype=tf.int64)
        self.n_class = [self.n_b_cate, self.n_m_cate, self.n_s_cate, self.n_d_cate]

        def forward_propagation(self, x):
            # The total number of time steps
            T = len(x)
            # During forward propagation we save all hidden states in s because need them later.
            # We add one additional element for the initial hidden, which we set to 0
            s = np.zeros((T + 1, self.hidden_dim))
            s[-1] = np.zeros(self.hidden_dim)
            # The outputs at each time step. Again, we save them for later.
            o = np.zeros((T, self.word_dim))
            # For each time step...
            for t in np.arange(T):
                # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
                s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t - 1]))
                o[t] = softmax(self.V.dot(s[t]))
            return [o, s]


        def bptt(self, x, y):
            T = len(y)
            # Perform forward propagation
            o, s = self.forward_propagation(x)
            # We accumulate the gradients in these variables
            dLdU = np.zeros(self.U.shape)
            dLdV = np.zeros(self.V.shape)
            dLdW = np.zeros(self.W.shape)
            delta_o = o
            delta_o[np.arange(len(y)), y] -= 1.
            # For each output backwards...
            for t in np.arange(T)[::-1]:
                dLdV += np.outer(delta_o[t], s[t].T)
                # Initial delta calculation: dL/dz
                delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
                # Backpropagation through time (for at most self.bptt_truncate steps)
                for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                    # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                    # Add to gradients at each previous step
                    dLdW += np.outer(delta_t, s[bptt_step - 1])
                    dLdU[:, x[bptt_step]] += delta_t
                    # Update delta for next step dL/dz at t-1
                    delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
            return [dLdU, dLdV, dLdW]


        self.Whh = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.n_hidden, self.n_hidden], name="Whh", dtype=tf.float32)
        self.bh = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.n_hidden], name="bh", dtype=tf.float32)

        ''' 1st state to classfy big category'''

        self.Wxbh = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.encoder_states_size, self.n_hidden], name="Wxbh", dtype=tf.float32)
        self.Whyb = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.n_hidden, self.n_b_cate], name="Whyb", dtype=tf.float32)
        self.byb = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.n_b_cate], name="byb", dtype=tf.float32)

        ''' 2nd state to classfy medium category'''
        self.Wxmh = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.n_b_cate, self.n_hidden], name="Wxmh", dtype=tf.float32)
        self.Whym = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.n_hidden, self.n_m_cate], name="Whym", dtype=tf.float32)
        self.bym = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.n_m_cate], name="bym", dtype=tf.float32)

        ''' 3rd state to classfy small category'''
        self.Wxsh = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                    shape=[self.n_m_cate, self.n_hidden], name="Wxsh")
        self.Whys = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                    shape=[self.n_hidden, self.n_s_cate], name="Whys", dtype=tf.float32)
        self.bys = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.n_s_cate], name="bys", dtype=tf.float32)
        ''' 4th state to classfy detail category'''
        self.Wxdh = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                    shape=[self.n_s_cate, self.n_hidden], name="Wxdh", dtype=tf.float32)
        self.Whyd = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                    shape=[self.n_hidden, self.n_d_cate], name="Whyd", dtype=tf.float32)
        self.byd = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.n_d_cate], name="byd", dtype=tf.float32)

        self.U = [self.Wxbh, self.Wxmh, self.Wxsh, self.Wxdh]
        self.W = [self.Whh, self.Whh, self.Whh, self.Whh]
        self.Wb = [self.bh, self.bh, self.bh, self.bh]
        self.V = [self.Whyb, self.Whym, self.Whys, self.Whyd]
        self.Vb = [self.byb, self.bym, self.bys, self.byd]

        self.decoder_states = [self.encoder_states] * 4
        self.decoder_inputs_v2 = [
            self.encoder_states,
            tf.one_hot(self.decoder_inputs[:, 0], depth=self.n_b_cate),
            tf.one_hot(self.decoder_inputs[:, 1], depth=self.n_m_cate),
            tf.one_hot(self.decoder_inputs[:, 2], depth=self.n_s_cate)
        ]
        self.logits = [None] * 4
        self.pred = [None] * 4
        self.cross_ent = [None] * 4
        self.cost = [None] * 4

        self.forward()
        self.bptt()
        self.gogo()

    def gogo(self):
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        bptt = sess.run([self.bptt])

    def forward(self):
        for t in range(self.time_step):
            self.decoder_states[t] = tf.nn.tanh(
                tf.add(
                    tf.add(
                        tf.matmul(self.decoder_inputs_v2[t], self.U[t]),
                        tf.matmul(self.decoder_states[t-1], self.W[t])),
                    self.Wb[t]
                )
            )

            self.logits[t] = tf.nn.softmax(tf.add(tf.matmul(self.decoder_states[t], self.V[t]), self.Vb[t]))
            self.pred[t] = tf.argmax(self.logits[t], axis=1)
            self.cross_ent[t] = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=tf.squeeze(self.logits[t]), labels=self.decoder_outputs[:, t])
            # TODO self.decoder_outputs[0]
            self.cost[t] = tf.reduce_mean(self.cross_ent[t])


    def bptt(self):
        dWhh = tf.zeros(shape=self.Whh.shape, name="Whh", dtype=tf.float32)
        dbh = tf.zeros(shape=self.bh.shape, name="bh", dtype=tf.float32)

        ''' 1st state to classfy big category'''
        dWxbh = tf.zeros(self.Wxbh.shape, name="dWxbh", dtype=tf.float32)
        dWhyb = tf.zeros(self.Whyb.shape, name="dWhyb", dtype=tf.float32)
        dbyb = tf.zeros(self.byb.shape, name="dbyb", dtype=tf.float32)

        ''' 2nd state to classfy medium category'''
        dWxmh = tf.zeros(self.Wxmh.shape, name="dWxmh", dtype=tf.float32)
        dWhym = tf.zeros(self.Whym.shape, name="dWhym", dtype=tf.float32)
        dbym = tf.zeros(self.bym.shape, name="dbym", dtype=tf.float32)

        ''' 3rd state to classfy small category'''
        dWxsh = tf.zeros(self.Wxsh.shape, name="dWxsh")
        dWhys = tf.zeros(self.Whys.shape, name="dWhys", dtype=tf.float32)
        dbys = tf.zeros(self.bys.shape, name="dbys", dtype=tf.float32)

        ''' 4th state to classfy detail category'''
        dWxdh = tf.zeros(self.Wxdh.shape, name="dWxdh", dtype=tf.float32)
        dWhyd = tf.zeros(self.Whyd.shape, name="dWhyd", dtype=tf.float32)
        dbyd = tf.zeros(self.byd.shape, name="dbyd", dtype=tf.float32)

        dU = [dWxbh, dWxmh, dWxsh, dWxdh]
        dW = [dWhh, dWhh, dWhh, dWhh]
        dWb = [dbh, dbh, dbh, dbh]
        dV = [dWhyb, dWhym, dWhys, dWhyd]
        dVb = [dbyb, dbym, dbys, dbyd]
        delta_o = [logit for logit in self.logits]
        # delta_t = [None] * 4

        # For each output backwards...
        for t in np.arange(self.time_step)[::-1]:
            delta_o[t] -= tf.one_hot(indices=self.decoder_outputs[:, t], depth=self.n_class[t])
            dV[t] += tf.matmul(tf.transpose(self.decoder_states[t]), delta_o[t])
            dVb[t] -= delta_o[t]
            # Initial delta calculation: dL/dz
            dstates = tf.matmul(delta_o[t], tf.transpose(self.V[t])) * (1-(tf.square(self.decoder_states[t])))

            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(0, t+1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                # Add to gradients at each previous step
                dW[t] += tf.matmul(tf.transpose(self.decoder_states[bptt_step-1]), dstates)
                dWb[t] -= dstates
                dU[bptt_step] += tf.matmul(tf.transpose(self.decoder_inputs_v2[bptt_step]), dstates)
                dstates= tf.matmul(dstates, tf.transpose(self.W[t])) * (1 - tf.square(self.decoder_states[bptt_step-1]))
        return [dU, dV, dW]



if __name__ == '__main__':
    decoder = Decoder()