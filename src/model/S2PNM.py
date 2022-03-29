"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import tensorflow as tf

import module.sequential as S
from model.Base import Sequential, FeedForward, layernorm
from module.coding import Embedding, PositionCoding


class S2PNM(Sequential):
    """ Implementation of the paper ---
        Chen C, Li D, Yan J, Yang X.
        Modeling dynamic user preference via dictionary learning for sequential recommendation.
        TKDE 2021.
    """

    def __init__(self, num_items, FLAGS):
        super().__init__(num_items, FLAGS)

        with tf.variable_scope("S2PNM"):
            self.item_embs = Embedding(self.num_items, self.num_units, self.l2_reg,
                                       zero_pad=True, scale=True, scope="item_embs")
            self.pcoding = PositionCoding(self.seqslen, self.num_units, self.l2_reg, scope="spatial_embs")

            self._cudnn_rnn = tf.contrib.cudnn_rnn.CudnnGRU(
                num_layers=1, num_units=self.num_units, direction='unidirectional',
                kernel_initializer=tf.orthogonal_initializer(), name='S2PNM/GRU')

            self.attention = S.MultiHeadAttention(self.num_units, self.num_heads, self.attention_probs_dropout_rate)
            self.fforward = FeedForward([self.num_units, self.num_units], self.hidden_dropout_rate)
            self.output_bias = self.output_bias(inf_pad=True)

    def __call__(self, features, is_training):
        seqs_id = features['seqs_i']
        seqs_inputs = self.item_embs(seqs_id)

        # Dropout
        seqs_units = tf.layers.dropout(seqs_inputs, rate=self.hidden_dropout_rate,
                                       training=tf.convert_to_tensor(is_training))
        seqs_masks = tf.expand_dims(tf.to_float(tf.not_equal(seqs_id, 0)), -1)

        # Recurrency
        with tf.variable_scope("S2PNM/Reccurency"):
            h, _ = self._cudnn_rnn(tf.transpose(seqs_units, [1, 0, 2]))
            h = tf.transpose(h, [1, 0, 2])  # mask the hidden states as dynamic_rnn did

        # Position coding and Mask
        seqs_units = self.pcoding(h)
        seqs_units = seqs_units * seqs_masks

        # Attention
        with tf.variable_scope("S2PNM/Attention"):
            seqs_units = self.attention(layernorm(seqs_units), seqs_units, is_training, causality=True)
            g = self.fforward(layernorm(seqs_units), is_training)

        # Dictionary Learning
        with tf.variable_scope("S2PNM/Dictionary"):
            seqs_outs = layernorm(tf.concat([g, h, g - h, g * h], axis=-1))
            seqs_outs = tf.layers.dense(seqs_outs, 2 * self.num_units, activation=tf.nn.sigmoid)
            seqs_outs = tf.layers.dense(seqs_outs, self.num_units)
            seqs_outs += seqs_inputs

        if is_training:
            seqs_outs = tf.reshape(seqs_outs, [tf.shape(seqs_id)[0] * self.seqslen, self.num_units])
        else:
            # only using the latest representations for making predictions
            seqs_outs = tf.reshape(seqs_outs[:, -1], [tf.shape(seqs_id)[0], self.num_units])

        # compute logits
        logits = tf.matmul(seqs_outs, self.item_embs.lookup_table, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.output_bias)
        return logits
