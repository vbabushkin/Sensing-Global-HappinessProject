__author__ = 'vahan'
#best wordcloud generator works with japan characters as well: http://www.jasondavies.com/wordcloud/#
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


folder = 'sampleStratum3'

files=os.listdir (folder)
# #TRANSLATION
for filename in files:
    documents=[]

    for line in open(folder+'/'+filename):
        #remove all https
        line=re.sub("(?P<url>https?://[^\s]+)", "", line)
        words = line.split()
        documents=documents+words


    documents=list(set(documents))
    print len(documents)


    # STOP_WORDS = nltk.corpus.stopwords.words('english')
    texts = [[word for word in document.lower().split()] for document in documents]

    #empty Arrays Removed from Texts
    texts=[text for text in texts if len(text)!=0]


    #make one aray
    all_tokens = sum(texts, [])


    #to eliminate words hashtags
    tokens_single_hashtag = set(word for word in set(all_tokens) if word[0] =='#')


    #######################################################################################################

    all_tokens = sum(texts, [])

    hashedDoc=''
    #concatenate all hashtags:
    for token in tokens_single_hashtag:
        hashedDoc=hashedDoc+' '+token
    ######################################################################################################


    textFile= open(folder+"/processed"+filename, 'w')
    for token in tokens_single_hashtag:
        textFile.write(token+"\n")
    textFile.close()
