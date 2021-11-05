import pandas as pd
import numpy as np
from scipy.sparse.construct import rand
from sklearn.neural_network import MLPClassifier
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import gensim
from sklearn.utils import _determine_key_type


word2vec = gensim.models.KeyedVectors.load_word2vec_format("~/Documents/Assignments/sem5/NLP/A3/GoogleNews-vectors-negative300.bin",binary=True)


def getWT(word):
    ind = word.rfind('/')
    return [word[:ind],word[ind+1:]]

def getEnc(tag):
    if tag in enc_dict.keys():
        return enc_dict[tag]
    else:
        i = len(enc_dict)
        enc = i+1
        enc_dict[tag] = i+1
        return enc


tagged = (open("/home/karan/Documents/Assignments/sem5/NLP/A3/Brown_tagged_train.txt",'r').readlines())

text = list()
tag_text = list()
emb = list()
tag_enc = list()
enc_dict = {}
tag_set = set()
mx = 0

for sent in tagged:
    tagged_words = sent.split()
    mx = max(mx,len(tagged_words))
    tags = list()
    words = list()
    emb_words = list()
    enc_tags = list()
    for word in tagged_words:
        l = getWT(word)
        wd = l[0]
        t = l[1]
        words.append(wd)
        tags.append(t)
        encoded_tag = getEnc(t)
        enc_tags.append(encoded_tag)
        try:
            emb_words.append(word2vec[wd])
        except KeyError:
            emb_words.append(word2vec[1])
    
    for i in range(386-len(emb_words)):
        emb_words.append(np.zeros(300))
        enc_tags.append(0)

    text.append(words)
    tag_text.append(tags)
    emb.append(emb_words)
    
    tag_enc.append(enc_tags)

from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(emb, tag_enc, test_size=0.2, random_state=0) 

classifier = MLPClassifier(hidden_layer_sizes=(30,30))
classifier.fit(trainX,trainY)