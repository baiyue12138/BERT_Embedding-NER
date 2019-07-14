# -*- coding: utf-8 -*-
"""
@author: bai
"""
import tensorflow as tf
import numpy as np
import os
import sys
import argparse
import model
import json
from utils import str2bool, get_entity, get_logger
from data_prepare import load_data, load_vocab, tag2label

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# hyperparameters
parser = argparse.ArgumentParser(description='Chinese NER')
parser.add_argument('--train_data', type=str, default='data', help='directory of train data')
parser.add_argument('--test_data', type=str, default='data', help='directory of test data')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
parser.add_argument('--epoch', type=int, default=50, help='epoch of training')
parser.add_argument('--hidden_dim', type=int, default=768, help='dim of LSTM cell hidden state')
parser.add_argument('--decode_method', type=int, default=1, help='0 is CRF 1 is Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--trainable', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pre_embedding', type=str, default='vector.npy', help='file of pretrained char embedding')
parser.add_argument('--embedding_dim', type=int, default=768, help='char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--model', type=int, default=0, help='0 for train. 1 for test, 2 for demo')
parser.add_argument('--model_path', type=str, default='model', help='model path for test and demo')
args = parser.parse_args()
def get_embedding():
    word2id = load_vocab(os.path.join('.', args.train_data, 'word2id.pkl'))
    embedding_path=os.path.join('.',args.train_data, args.pre_embedding)
    if os.path.exists(embedding_path):
        embeddings = np.array(np.load(embedding_path), dtype='float32')
        embedding_method='pretrained'    
    else:
        embeddings = np.random.uniform(-0.25, 0.25, (len(word2id), args.embedding_dim))
        embeddings = np.float32(embeddings)
        embedding_method='random'
    return word2id, embeddings, embedding_method

def get_model(embeddings,word2id,paths):
    encoder = model.Encoder(args, embeddings,config=config)
    encoder.build()
    if args.decode_method==0 or args.decode_method==1:
        decoder=model.softmaxOrCRFDecoder(encoder,args, tag2label, word2id, paths, config)
    else:
        print("Invalid argument! Please use valid arguments!")
    decoder.build()
    return decoder
def save_result(saver,sess,decoder,ckpt_file):
    saver.restore(sess, ckpt_file)
    f = open("./data/case/input.txt","r")
    sentences = f.readlines()
    f.close() 
    result=[]
    for step, sen in enumerate(sentences):
        sys.stdout.write(' processing: {} sentence / {} sentences.'.format(step + 1, len(sentences)) + '\r')
        instance={}
        sen_ = list(sen.strip().strip('\r\n'))
        char_list = [(sen_, ['O'] * len(sen_))]
        tag = decoder.demo_one(sess, char_list)
        PER, LOC, ORG = get_entity(tag, sen_)
        #print(sen)
        #print("PER",PER)
        #print("LOC",LOC)
        #print("ORG",ORG)
        instance['sen']=sen
        instance['PER']=PER
        instance['LOC']=LOC
        instance['ORG']=ORG
        result.append(instance)
    with open('./data/case/result.json','w',encoding='utf-8')as fw:
        fw.write(json.dumps(result,ensure_ascii=False))
    print("********The result is saved in the ./data/case/result.json*********"+ '\r')        
if __name__=='__main__':
    word2id, embeddings, embedding_method = get_embedding()
    if args.model != 2:
        train_path = os.path.join('.', args.train_data, 'train_data')
        test_path = os.path.join('.', args.test_data, 'test_data')
        train_data = load_data(train_path)
        test_data =  load_data(test_path)
        test_size = len(test_data)
    paths = {}
    if args.decode_method==0:
        decode='CRF'
    else:
        decode='Softmax'
    output_path = os.path.join('.', args.model_path, decode+"_"+embedding_method)
    if not os.path.exists(output_path): 
        os.makedirs(output_path)
    model_path = os.path.join(output_path, "checkpoints/")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if args.model==0:
        ckpt_prefix = os.path.join(model_path, "model")
        paths['save_path'] = ckpt_prefix
    if args.model==1 or args.model==2:
        ckpt_file = tf.train.latest_checkpoint(model_path)
        paths['save_path'] = ckpt_file
    result_path = os.path.join(output_path, "results")
    paths['result_path'] = result_path
    if not os.path.exists(result_path): 
        os.makedirs(result_path)
    log_path = os.path.join(result_path, "log.txt")
    paths['log_path'] = log_path

    decoder = get_model(embeddings,word2id,paths)
    # train
    if args.model == 0:
        decoder.train(train=train_data, dev=test_data)
    # test
    elif args.model == 1:
        print("test data: {}".format(test_size))
        decoder.test(test_data)
    # demo
    elif args.model == 2:
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            save_result(saver,sess,decoder,ckpt_file)
