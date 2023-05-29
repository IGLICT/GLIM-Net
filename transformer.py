# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Transformer network
'''
import tensorflow as tf

# from data_load import load_vocab
from modules import ff, year_positional_encoding, multihead_attention_time
from tqdm import tqdm
import logging
import numpy as np

from icecream import ic

logging.basicConfig(level=logging.INFO)

class Transformer:
    '''
    xs: tuple of
        x: int32 tensor. (N, T1) vocab idx
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,) sentences
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    training: boolean.
    '''
    def __init__(self):
        self.maxlen1 = 5
        self.d_model = 512
        self.dropout_rate = 0.5
        self.num_blocks = 6
        self.num_heads = 8
        self.d_ff = 2048
        self.maxlen2 = 5
        self.vocab_size = 32000
        self.lr = 0.0003
        self.warm_steps = 4000

    # def encode(self, xs, training=True):
    def encode(self, xs, mask, delta_year, time_matrix, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model) (batch_size, length, hidden_size)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            # x, seqlens, sents1 = xs

            src_masks = mask

            enc = xs
            enc += year_positional_encoding(enc, self.maxlen1,delta_year)
            enc = tf.layers.dropout(enc, self.dropout_rate, training=training)
            ## Blocks
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention_time(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              key_masks=src_masks,
                                              delta_year=time_matrix,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self.d_ff, self.d_model])

        memory = enc
        # return memory, sents1, src_masks
        return memory, src_masks

    def decode(self, ys, memory, src_masks, delta_year, time_matrix, training=True):
        '''
        memory: encoder outputs. (N, T1, d_model)
        src_masks: (N, T1)

        Returns
        logits: (N, T2, V). float32.
        y_hat: (N, T2). int32
        y: (N, T2). int32
        sents2: (N,). string.
        '''
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            # decoder_inputs, y, seqlens, sents2 = ys

            # tgt_masks
            mask = np.array([[False, False, False, False, False], 
                             [False, False, False, False, False], 
                             [False, False, False, False, False],
                             [False, False, False, False, False]])
            tgt_masks = tf.convert_to_tensor(mask)


            # PE module to be completed
            dec = ys
            dec = tf.cast(dec, dtype=tf.float32)
            start = tf.constant([[-1],[-1],[-1],[-1]],dtype=tf.float32)
            dec = tf.concat([start,dec],1)
            dec = dec[:,:5]
            dec = tf.expand_dims(dec, 2)
            dec = tf.tile(dec,[1,1,512])

            dec += year_positional_encoding(dec, self.maxlen2, delta_year)
            dec = tf.layers.dropout(dec, self.dropout_rate, training=training)

            # Blocks
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):

                    dec = multihead_attention_time(queries=dec,
                                              keys=dec,
                                              values=dec,
                                              key_masks=tgt_masks,
                                              delta_year=time_matrix,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.dropout_rate,
                                              training=training,
                                              causality=True,
                                              scope="self_attention")

                    dec = multihead_attention_time(queries=dec,
                                              keys=memory,
                                              values=memory,
                                              key_masks=src_masks,
                                              delta_year=time_matrix,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.dropout_rate,
                                              training=training,
                                              causality=False,
                                              scope="vanilla_attention")
                    ### Feed Forward
                    dec = ff(dec, num_units=[self.d_ff, self.d_model])


        stacked_outputs_0 = tf.layers.dense(dec, 256,name='fc0')
        stacked_outputs_0 = tf.layers.dropout(stacked_outputs_0, self.dropout_rate, training=training,name='do0')
        stacked_outputs_1 = tf.layers.dense(stacked_outputs_0, 2,name='fc1')

        # return logits, y_hat, y, sents2
        return stacked_outputs_1, stacked_outputs_1, ys




