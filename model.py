# -*- coding: utf-8 -*-
"""
@author: bai
"""
from data_prepare import pad_sequences, next_batch
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from evaluation import conlleval
from utils import get_logger
import os
import sys

class Encoder(object):
    def __init__(self, args, embeddings, config):
        self.hidden_dim = args.hidden_dim
        self.embeddings = embeddings
        self.trainable = args.trainable
        self.dropout = args.dropout
        self.batch_size = args.batch_size
        self.outputs = None
        self.state = None
        self.config = config
        self.encoder_fw_cell = None
        self.encoder_bw_cell = None
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
    def lookup_layer(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings,
                                           dtype=tf.float32,
                                           trainable=self.trainable,
                                           name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.word_ids,
                                                     name="word_embeddings")
        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout_pl)
    def _encoder(self,inputs):
        with tf.variable_scope("encoder"):
            self.encoder_fw_cell = LSTMCell(self.hidden_dim)
            self.encoder_bw_cell = LSTMCell(self.hidden_dim)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self.encoder_fw_cell,
                cell_bw=self.encoder_fw_cell,
                inputs=inputs,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            outputs = tf.concat(outputs, axis=-1)
            state = (tf.reduce_mean((state[0][0], state[1][0]), axis=0), tf.reduce_mean((state[0][1], state[1][1]), axis=0))
            outputs = tf.nn.dropout(outputs, self.dropout_pl)
        return outputs,state  
    def build(self):
        self.lookup_layer()
        self.outputs,self.state = self._encoder(inputs=self.word_embeddings)
        
class Decoder(object):
    def __init__(self,encoder,args, tag2label, vocab, paths, config):
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.hidden_dim = args.hidden_dim
        self.decode_method = args.decode_method
        self.trainable = args.trainable
        self.dropout = args.dropout
        self.embedding_dim=args.embedding_dim
        self.lr = args.lr
        self.clip_grad = args.clip
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab
        self.shuffle = args.shuffle
        self.model_path = paths['save_path']
        self.result_path = paths['result_path']
        self.logger = get_logger(paths['log_path'])
        self.config = config
        self.encoder=encoder
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
    def get_prop(self,output):
        with tf.variable_scope("predict"):
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)
            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2*self.hidden_dim])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])
    def _loss(self):
        if self.decode_method==1:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
            mask = tf.sequence_mask(self.encoder.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)
        if self.decode_method==0:
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                   tag_indices=self.labels,
                                                                   sequence_lengths=self.encoder.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)      
    def _predict(self):
        if self.decode_method==1:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1) 
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)

    def _optimize(self):
        with tf.variable_scope("Optimizing"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)
    def init(self):
        self.init_op = tf.global_variables_initializer()
    def train(self, train, dev):
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            for epoch in range(self.epoch_num):
                self.train_one_epoch(sess, train, dev, self.tag2label, epoch, saver)

    def test(self, test):
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            print('---------- testing -------')
            saver.restore(sess, self.model_path)
            label_list, seq_len_list = self.dev_one_epoch(sess, test)
            self.evaluate(label_list, seq_len_list, test)

    def demo_one(self, sess, sent):
        label_list = []
        for seqs, labels in next_batch(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, _ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        tag = [label2tag[label] for label in label_list[0]]
        return tag

    def train_one_epoch(self, sess, train, dev, tag2label, epoch, saver):
        num_batches = (int)(len(train)/self.batch_size) 
        batches = next_batch(train, self.batch_size, self.vocab, self.tag2label, shuffle=self.shuffle)
        for step, (seqs, labels) in enumerate(batches):
            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            feed_dict, _ = self.update(seqs, labels, self.lr, self.dropout)
            _, loss_train = sess.run([self.train_op, self.loss],feed_dict=feed_dict)
            if step + 1 == num_batches:
                self.logger.info('+++++++++++++epoch {}, loss: {:.4}++++++++++'.format(epoch + 1,loss_train))
                saver.save(sess, self.model_path, global_step=epoch)
        self.logger.info('------------------validation----------------')
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)

    def update(self, seqs, labels=None, lr=None, dropout=None):
        word_ids, seq_len_list = pad_sequences(seqs,pad_mark=0)

        feed_dict = {self.encoder.word_ids: word_ids,
                     self.encoder.sequence_lengths: seq_len_list,
                     }
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.encoder.dropout_pl] = dropout

        return feed_dict, seq_len_list

    def dev_one_epoch(self, sess, dev):
        label_list, seq_len_list = [], []
        for seqs, labels in next_batch(dev, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):
        feed_dict, seq_len_list = self.update(seqs, dropout=1.0)
        if self.decode_method==1:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list
        if self.decode_method==0:
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

       
    def evaluate(self, label_list, seq_len_list, data, epoch=None):
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label

        model_predict = []
        for label_, (sent, tag) in zip(label_list, data):
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        epoch_num = str(epoch+1) if epoch != None else 'test'
        label_path = os.path.join(self.result_path, 'label_' + epoch_num)
        metric_path = os.path.join(self.result_path, 'result_metric_' + epoch_num)
        for _ in conlleval(model_predict, label_path, metric_path):
            self.logger.info(_)


class softmaxOrCRFDecoder(Decoder):
    def build(self):
        self.get_prop(output=self.encoder.outputs)
        self._predict()
        self._loss()
        self._optimize()
        self.init()
        
class LSTMDecoder(Decoder):
    pass
