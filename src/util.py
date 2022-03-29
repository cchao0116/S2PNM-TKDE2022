"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import logging

import numpy as np

import dataloader as D


class EarlyStopping(object):
    def __init__(self, FLAGS, patience=10):
        self.model = FLAGS.model
        self.patience = patience

        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
        elif acc < self.best_acc:  # count the best acc is not achieved
            self.counter += 1
            logging.info(f'EarlyStopping {self.model} counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop


def ranking(FLAGS):
    if FLAGS.model == "S2PNM":
        from model.S2PNM import S2PNM
        return S2PNM(FLAGS.num_items, FLAGS)
    else:
        raise NotImplementedError("The ranking model: {0} not implemented".format(FLAGS.model))


def reader(FLAGS, file_pattern, is_training: bool):
    if FLAGS.model == 'BERT4REC':
        seqslen = FLAGS.seqslen + 1
        mask = FLAGS.num_items + 1
        return D.InputReader(file_pattern, is_training=is_training,
                             decoder=D.TfExampleDecoder(seqslen),
                             processor=D.MaskedPostProcessor(seqslen, FLAGS.masklen, mask, is_training))
    elif FLAGS.model == 'GREC':
        seqslen = FLAGS.seqslen + 1
        mask = FLAGS.num_items + 1
        return D.InputReader(file_pattern, is_training=is_training,
                             decoder=D.TfExampleDecoder(seqslen),
                             processor=D.GRECPostProcessor(seqslen, FLAGS.masklen, mask, is_training))
    elif FLAGS.model == 'TimelyREC':
        return D.InputReader(file_pattern, is_training=is_training,
                             decoder=D.TfExampleDecoder(FLAGS.seqslen + 1, has_datetime=True),
                             processor=D.RegressivePostProcessor(has_datetime=True))
    else:
        return D.InputReader(file_pattern, is_training=is_training,
                             decoder=D.TfExampleDecoder(FLAGS.seqslen + 1),
                             processor=D.RegressivePostProcessor())
