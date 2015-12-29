__author__ = 'vahan'

#http://radimrehurek.com/gensim/tut2.html

import gensim
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora, models, similarities, matutils
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os


filename="oct1516/text/text_extractedTweetsFromUnited Kingdom.txt"
documents=[]

for line in open(filename):
    print line;
    #remove all https
    line=re.sub("(?P<url>https?://[^\s]+)", "", line)
    #leave words only
    line = re.sub(r"[^A-Za-z\s]", "", line.strip())
    words = line.split()
    documents=documents+words

# to remove duplicated words
documents=list(set(documents))
print len(documents)

# remove common words and tokenize
#stoplist = set('for a of the and to in'.split())

STOP_WORDS = nltk.corpus.stopwords.words('english')
texts = [[word for word in document.lower().split() if word not in STOP_WORDS] for document in documents]

#empty Arrays Removed from Texts
texts=[text for text in texts if len(text)!=0]


#make one aray
all_tokens = sum(texts, [])

# remove words that appear only once
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)

#eliminate words with one letter
tokens_single_letter = set(word for word in set(all_tokens) if len(word) == 1)
#tokens_once = set(word for word in set(all_tokens) if len(word) >1)

texts = [[word for word in text if word not in tokens_once and word not in tokens_single_letter]for text in texts]

#empty Arrays Removed from Texts
texts=[text for text in texts if len(text)!=0]

#lemmatize words
lmtzr = WordNetLemmatizer()
texts=[[lmtzr.lemmatize(text[0])] for text in texts]
#######################################################################################################

dictionary = corpora.Dictionary(texts)
dictionary.save('oct1516/text/deerwester.dict') # store the dictionary, for future reference
print(dictionary)

print(dictionary.token2id)


corpus = [dictionary.doc2bow(text) for text in texts]

corpora.MmCorpus.serialize('oct1516/text/deerwester.mm', corpus) # store to disk, for later use

print(corpus)

tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model

corpus_tfidf = tfidf[corpus]

for doc in corpus_tfidf:
    print(doc)

#######################################################################################################################################
lda = gensim.models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=5, update_every=1, chunksize=10000, passes=1)

lda.print_topics(5)


########################################################################################################################################
#
#
#
#
#
# lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=30) # initialize an LSI transformation
# corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
#
# lsi.print_topics(2, topn=5)
#
# for doc in corpus_lsi: # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
#     print(doc)
#
#
#
#
# ###############################################################################################################
# # http://sujitpal.blogspot.ae/2014/08/topic-modeling-with-gensim-over-past.html
#
# MODELS_DIR='/home/vahan/temp2'
# # write out coordinates to file
# fcoords = open(os.path.join(MODELS_DIR, "coords.csv"), 'wb')
# for vector in lsi[corpus]:
#     # if len(vector) != 2:
#     #     continue
#     # print((vector[0][1], vector[1][1]))
#     fcoords.write("%6.4f\t%6.4f\n" % (vector[0][1], vector[1][1]))
# fcoords.close()
#
#
# #clustering topics
# # http://sujitpal.blogspot.ae/2014/08/topic-modeling-with-gensim-over-past.html
# # Source: num_topics.py
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
#
# MODELS_DIR='/home/vahan/temp2'
# MAX_K = 10
#
# X = np.loadtxt(os.path.join(MODELS_DIR, "coords.csv"), delimiter="\t")
# ks = range(1, MAX_K + 1)
#
# inertias = np.zeros(MAX_K)
# diff = np.zeros(MAX_K)
# diff2 = np.zeros(MAX_K)
# diff3 = np.zeros(MAX_K)
# for k in ks:
#     kmeans = KMeans(k).fit(X)
#     inertias[k - 1] = kmeans.inertia_
#     # first difference
#     if k > 1:
#         diff[k - 1] = inertias[k - 1] - inertias[k - 2]
#     # second difference
#     if k > 2:
#         diff2[k - 1] = diff[k - 1] - diff[k - 2]
#     # third difference
#     if k > 3:
#         diff3[k - 1] = diff2[k - 1] - diff2[k - 2]
#
# elbow = np.argmin(diff3[3:]) + 3
#
# plt.plot(ks, inertias, "b*-")
# plt.plot(ks[elbow], inertias[elbow], marker='o', markersize=12,
#          markeredgewidth=2, markeredgecolor='r', markerfacecolor=None)
# plt.ylabel("Inertia")
# plt.xlabel("K")
# plt.show()
#
#
# # re-ran the KMeans algorithm with K=5 and generated the clusters.
#
# # Source: viz_topics_scatter.py
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
#
# MODELS_DIR='/home/vahan/temp2'
# NUM_TOPICS = 5
#
# X = np.loadtxt(os.path.join(MODELS_DIR, "coords.csv"), delimiter="\t")
# kmeans = KMeans(NUM_TOPICS).fit(X)
# y = kmeans.labels_
#
# colors = ["b", "g", "r", "m", "c"]
# for i in range(X.shape[0]):
#     plt.scatter(X[i][0], X[i][1], c=colors[y[i]], s=10)
# plt.show()
#
#
#
# ###############################################################################################################
#
# #LDA
# model = models.ldamodel.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=5)
#
# corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
#
# lsi.print_topics(2)
#
#
#
# from os import path
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# all_tokens = sum(texts, [])
#
# doc=''
# #concatenate all our texts:
# for token in all_tokens:
#     doc=doc+' '+token
#
#
# wordcloud = WordCloud().generate(doc)
# # Open a plot of the generated image.
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.show()
#
#
#
#
#
# # Finding the natural number of topics for Latent Dirichlet Allocation from
# # http://blog.cigrainger.com/2014/07/lda-number.html
#
# # Define KL function
# def sym_kl(p,q):
#     return np.sum([stats.entropy(p,q),stats.entropy(q,p)])
#
# l = np.array([sum(cnt for _, cnt in doc) for doc in corpus])
# def arun(corpus,dictionary,min_topics,max_topics,step=1):
#     kl = []
#     for i in range(min_topics,max_topics,step):
#         lda = models.ldamodel.LdaModel(corpus=corpus,
#             id2word=dictionary,num_topics=i)
#         m1 = lda.expElogbeta
#         U,cm1,V = np.linalg.svd(m1)
#         #Document-topic matrix
#         lda_topics = lda[corpus]
#         m2 = matutils.corpus2dense(lda_topics, lda.num_topics).transpose()
#         cm2 = l.dot(m2)
#         cm2 = cm2 + 0.0001
#         cm2norm = np.linalg.norm(l)
#         cm2 = cm2/cm2norm
#         kl.append(sym_kl(cm1,cm2))
#     return kl
#
# kl = arun(corpus_tfidf,dictionary,min_topics=1, max_topics=10)
#
# #plot natural number of topics
# plt.plot(kl)
# plt.ylabel('Symmetric KL Divergence')
# plt.xlabel('Number of Topics')
# plt.savefig('kldiv.png', bbox_inches='tight')
# plt.show()
# #######################################################################################################################
#
#
# # set up logging so we see what's going on
# import logging
# import os
# from gensim import corpora, models, utils
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#
# def iter_documents(reuters_dir):
#     """Iterate over Reuters documents, yielding one document at a time."""
#     for fname in os.listdir(reuters_dir):
#         # read each document as one big string
#         document = open(os.path.join(reuters_dir, fname)).read()
#         # parse document into a list of utf8 tokens
#         yield utils.simple_preprocess(document)
#
# class ReutersCorpus(object):
#     def __init__(self, reuters_dir):
#         self.reuters_dir = reuters_dir
#         self.dictionary = corpora.Dictionary(iter_documents(reuters_dir))
#         self.dictionary.filter_extremes()  # remove stopwords etc
#
#     def __iter__(self):
#         for tokens in iter_documents(self.reuters_dir):
#             yield self.dictionary.doc2bow(tokens)
#
# # set up the streamed corpus
# corpus = ReutersCorpus('/home/vahan/nltk_data/corpora/reuters/training/')
# # INFO : adding document #0 to Dictionary(0 unique tokens: [])
# # INFO : built Dictionary(24622 unique tokens: ['mdbl', 'fawc', 'degussa', 'woods', 'hanging']...) from 7769 documents (total 938238 corpus positions)
# # INFO : keeping 7203 tokens which were in no less than 5 and no more than 3884 (=50.0%) documents
# # INFO : resulting dictionary: Dictionary(7203 unique tokens: ['yellow', 'four', 'resisted', 'cyprus', 'increase']...)
#
# # train 10 LDA topics using MALLET
# mallet_path = '/home/vahan/mallet-2.0.7/bin/mallet'
# model = models.LdaMallet(mallet_path, corpus, num_topics=10, id2word=corpus.dictionary)
# # 0	5	bank market rate dollar stg exchange rates banks interest money yen central currency federal today fed week pct dealers
# # 1	5	mln cts net loss shr dlrs profit qtr revs year note oper avg shrs includes sales gain jan tax
# # 2	5	wheat department grain corn agriculture year program report usda loan soviet week export crop farm union association farmers production
# # 3	5	international meeting ec agreement price world market coffee talks prices european west brazil countries told stock producers export current
# # 4	5	trade japan government japanese foreign told united officials states economic official industry tax president imports house minister countries world
# # 5	5	oil prices production gas gold crude canadian price energy canada opec petroleum bpd day barrels resources reserves industry output
# # 6	5	pct billion year february january rose rise december fell growth compared increase earlier figures current month quarter end deficit
# # 7	5	tonnes march april record dividend pay sugar div prior split qtly sets total june china traders quarterly board week
# # 8	5	shares company stock offer corp group share common stake shareholders acquisition merger buy american pct board management outstanding bid
# # 9	5	dlrs mln company year share quarter earnings sale unit business sales corp expects results assets operations income operating reported
#
# #
# # <1000> LL/token: -7.5002
# #
# # Total time: 34 seconds
#
# # now use the trained model to infer topics on a new document
# doc=''
#
# #make one aray
# all_tokens = sum(texts, [])
#
# #concatenate all our texts:
# for token in all_tokens:
#     doc=doc+' '+token
#
# bow = corpus.dictionary.doc2bow(utils.simple_preprocess(doc))
# print model[bow]  # print list of (topic id, topic weight) pairs
# #[(0, 0.0641025641025641),
# # (1, 0.0641025641025641),
# # (2, 0.10398860398860399),
# # (3, 0.1737891737891738),
# # (4, 0.1467236467236467),
# # (5, 0.0811965811965812),
# # (6, 0.0641025641025641),
# # (7, 0.08974358974358974),
# # (8, 0.07264957264957266),
# # (9, 0.1396011396011396)]
#
#
#
# # ALSO VERY NICE VISUALIZATION:
# # http://diging.github.io/tethne/doc/0.6.1-beta/tutorial.mallet.html

