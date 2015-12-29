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

import csv

stratum=1



ifile  = open('HDICountriesClusters.csv', "rb")
reader = csv.reader(ifile)


#extract countries for particular cluster
countries=[]
rownum=0
for row in reader:
    if rownum==0:
        header = row
    else:
        if int(row[3])==stratum :
            countries.append(row[1])
    rownum+=1
ifile.close()



######################################################################################################
#save to file for http://www.wordle.net/create


textFile= open("wordsForStratum_"+str(stratum)+".txt", 'w')
documents=[]

for country in countries:
    filename="combineTranslated/text_extractedTweetsFrom"+country+".txt"


    for line in open(filename):
        #remove all https
        line=re.sub("(?P<url>https?://[^\s]+)", "", line)
        #leave words only
        line = re.sub(r"[^A-Za-z\s]", "", line.strip())
        words = line.split()
        documents=documents+words

    print country

    # to remove duplicated words
    documents=list(set(documents))

    print "length of documents now"
    print len(documents)


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
    # dictionary.save('oct1516/text/deerwester.dict') # store the dictionary, for future reference
    # print(dictionary)
    #
    # print(dictionary.token2id)


    corpus = [dictionary.doc2bow(text) for text in texts]

    # corpora.MmCorpus.serialize('oct1516/text/deerwester.mm', corpus) # store to disk, for later use

    # print(corpus)

tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model

corpus_tfidf = tfidf[corpus]

# for doc in corpus_tfidf:
#     print(doc)

#######################################################################################################################################
lda = gensim.models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=5, update_every=1, chunksize=10000, passes=1)

lda.print_topics(5)
