# BiLSTM-CRF+ Bert Embedding for Chinese NER

We implemented a BiLSTM-CRF model for NER, which identifies three types of entities: PERSON, LOCATION and ORGANIZATION.

## Requirement:

- Python: 3.6 
- TensorFlow: 1.8

## Dataset
Dataset comes from [MSRA corpus](http://sighan.cs.uchicago.edu/bakeoff2006/).

### Folder Description:

The directory `./data` contains:

- the preprocessed data files, `train_data` and `test_data` 
- a vocabulary file `word2id.pkl` 
- a char embedding file `vector.npy`   #This file is a pre-trained char embeddings. Randomly initializes the embedding when this file does not exist.


The directory `./data/case` contains:
- a original file'input.txt',# When you need to use our model for NER, you need to put these sentences into 'input.txt'
- a 'result.json' # save all identified entities

The directory `./model` is used to save the trained model file

## Input format:

 We use BIO tag scheme, with each character its label for one line. Sentences are splited with a null line.
```
我	O
们	O
是	O
受	O
到	O
郑	B-PER
振	I-PER
铎	I-PER
先	O
生	O
、	O
阿	B-PER
英	I-PER
先	O
生	O
著	O
作	O
的	O
启	O
示	O
```
## Pretrained Embeddings

We use Dr. Xiao Han’s[bert-as-service](https://github.com/hanxiao/bert-as-service)to obtain the BERT embeddings for NER, which is used as follows:
- 1.Environmental requirements: python version >=3.5, tensorflow version >=1.10
- 2.pip install  bert-serving-server; 
- 3.pip install bert-serving-client
- 4.Download the trained BERT Chinese model[chinese_L-12_H-768_A-12.zip](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)
- 5.start up bert-as-service : bert-serving-start -model_dir /home/chinese_L-12_H-768_A-12 -num_worker=2
- 6.Get the word vector using getBertVector.py

## Model

we use __BiLSTM__, can efficiently use *both past and future* input information and extract features automatically.

we use __CRF__ and a __Softmax__,  label the tag for each character in one sentence. 

## How to run the code?

`python main.py --model=0`

the model can be 0 ,1 or 2.
"--model=0" means train "--model=1" means test and "--model=2" means application.
If you use "--model=2", you can identify the entities in some original sentences.
First, you need to put the original sentence into the ./data/case/input.txt file;
Then run the command `python main.py --model=2` and the recognized result will be saved to ./data/case/ Result.json


The best test performance of the model is:
```
ccuracy:  98.64%; precision:  90.87%; recall:  87.13%; F1:  88.96
        LOC: precision:  93.41%; recall:  90.16%; F1:  91.76  2777
        ORG: precision:  85.64%; recall:  84.67%; F1:  85.15  1316
        PER: precision:  90.78%; recall:  84.38%; F1:  87.46  1844
```



## References

\[1\] [https://github.com/Determined22/zh-NER-TF](https://github.com/Determined22/zh-NER-TF)

\[2\] [https://github.com/hanxiao/bert-as-service](https://github.com/hanxiao/bert-as-service)

\[3\] [https://github.com/google-research/bert](https://github.com/google-research/bert)