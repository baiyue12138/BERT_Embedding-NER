# -*- coding: utf-8 -*-
"""
@author: bai
"""
import numpy as np
import os
import pickle
from bert_serving.client import BertClient
def read_dictionary(vocab_path):
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id
if __name__=='__main__':
    word2id={}
    words=[]
    vecs=[]
    words2id_filename='./data/word2id.pkl'
    words2vec_filename='./data/vector.npy'
    word2id=read_dictionary(words2id_filename)
    for word, Id in  word2id.items():
        words.append(word)
    print("vocab_size：",len(words))
    bc = BertClient()
    vecs=bc.encode(words)
    print("vocab_size: ",len(vecs))
    np.save(words2vec_filename,vecs)
    
