import nltk
from nltk.corpus import reuters
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import gensim
import os
import collections
import smart_open
import random


cats = reuters.categories()
l = [len(reuters.fileids(cats[i])) for i in range(90)]
l2 = [len(reuters.fileids(cats[j])) for j in range(90)]
l.sort(reverse=True)
categories = [cats[l2.index(l[k])] for k in range(10)]
#categories is the list of top 10 categories

length = len(reuters.fileids())
modids = [0]*length
c=0
for ind in range(length):
    id = reuters.fileids()[ind]
    check=0
    cat = reuters.categories(id)
    for ca in cat:
        if ca in categories:
            check+=1
    if check!=0:
        modids[c]=id
        c=c+1

# c is the number of docs belonging to top 10 ids
Top10catids = [modids[ii] for ii in range(c)] # ids of those c docs
test = [d for d in Top10catids if d.startswith('test/')]#test documents fileids list
train = [d for d in Top10catids if d.startswith('training/')]

x1 = [0]*len(train)
x2 = [0]*len(test)

for x11 in range(len(train)):
    id1 = train[x11]
    l1 = reuters.categories(id1)
    t1=0
    for x12 in l1:
        if x12 in categories:
            t1 = t1+1# number of categories that this doc(id1) belongs to and that are present in top 10 classes
    tt1=[""]*t1
    tin=0
    for x12 in l1:
        if x12 in categories:
            tt1[tin]=x12
            tin+=1
    x1[x11]=tt1

# x1: list of list that contains all categories(out of top 10) that our training docs belong to

for x21 in range(len(test)):
    id2 = test[x21]
    l2 = reuters.categories(id2)
    t2=0
    for x22 in l2:
        if x22 in categories:
            t2 = t2+1# number of categories that this doc(id1) belongs to and that are present in top 10 classes
    tt2=[""]*t2
    tin2=0
    for x22 in l2:
        if x22 in categories:
            tt2[tin2]=x22
            tin2+=1
    x2[x21]=tt2



Stop_Words = stopwords.words("english")


def tokenize(text):
    clean_txt = re.sub('[^a-z\s]+',' ',text)  # replacing spcl chars, punctuations by space
    clean_txt = re.sub('(\s+)',' ',clean_txt)  # replacing multiple spaces by single space
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(clean_txt))  # tokenizing, lowercase
    words = [word for word in words if word not in Stop_Words]  # filtering stopwords
    words = filter(lambda t: len(t)>=min_length, words)  # filtering words of length <=2
    tokens =(list(map(lambda token: PorterStemmer().stem(token),words)))  # stemming tokens
    return tokens


n_classes = 10
labels = categories
stop_words = stopwords.words("english")


mlb = MultiLabelBinarizer()
docs = {}
docs['train'] = [reuters.raw(doc_id) for doc_id in train]
docs['test'] = [reuters.raw(doc_id) for doc_id in test]

trd = docs['train']
tstd = docs['test']

y_tr = mlb.fit_transform(x1)
y_tst = mlb.fit_transform(x2)

t_d_tr = [tokenize(dd) for dd in trd]#tokenized training docs
t_d_tst = [tokenize(ddd) for ddd in tstd]


def read_corpus(fname):
    for i, line in enumerate(fname):
        yield gensim.models.doc2vec.TaggedDocument(line, [i])


train_corp = list(read_corpus(t_d_tr))

#model training:-
model = gensim.models.doc2vec.Doc2Vec(vector_size=70, min_count=2, epochs=100)
model.build_vocab(train_corp)
model.train(train_corp,  epochs=100, total_examples=model.corpus_count)


#model evaluation:-

r_precs = []

for c in categories:
  file_i = reuters.fileids(c)
  tr_ids = [ii for ii in file_i if ii.startswith('training/')]
  tst_ids = [iii for iii in file_i if iii.startswith('test/')]
  dR = len(tst_ids)#same_cattrade_len

  q_id = random.choice(tst_ids)
  indd = test.index(q_id)
  inferred_vector = model.infer_vector(t_d_tst[indd])

  sims = model.docvecs.most_similar([inferred_vector], topn=dR)
  pos=0
  for tt in range(dR):
    rnk_tt_id = train[sims[tt][0]]
    if(rnk_tt_id in tr_ids):
      pos = pos+1
  r_precs.append(pos/dR)

# print(r_precs)
print(sum(r_precs)/len(r_precs))











