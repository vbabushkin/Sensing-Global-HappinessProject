__author__ = 'vahan'
__author__ = 'vahan'

from os import path
import matplotlib.pyplot as plt
from wordcloud import WordCloud
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

doc=''
for country in countries:
    documents=[]
    filename="combineTranslated/text_extractedTweetsFrom"+country+".txt"

    print country
    print filename
    for line in open(filename):
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

    all_tokens = sum(texts, [])

    #AGAIN eliminate words with one letter
    tokens_single_letter = set(word for word in set(all_tokens) if len(word) == 1)
    #tokens_once = set(word for word in set(all_tokens) if len(word) >1)

    texts = [[word for word in text if word not in tokens_once and word not in tokens_single_letter]for text in texts]

    #empty Arrays Removed from Texts
    texts=[text for text in texts if len(text)!=0]

    all_tokens = sum(texts, [])


    #concatenate all our texts:
    for token in all_tokens:
        doc=doc+' '+token


    #######################################################################################################

    for token in all_tokens:
        textFile.write(token+"\n")

    print len(doc)


textFile.close()

#for Japanese characters use https://tagul.com/cloud/1
#for others http://www.wordle.net/create


wordcloud = WordCloud().generate(doc)
# Open a plot of the generated image.
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
