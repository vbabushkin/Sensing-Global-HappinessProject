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


country="Nigeria"
filename="oct1516/textOnly/text_extractedTweetsFrom"+country+".txt"





documents=[]

for line in open(filename):
    print line;
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
#save to file for http://www.wordle.net/create
folder="oct1516/hashtagsFromAllTweetsByCountries"

textFile= open(folder+"/text_"+country+".txt", 'w')
for token in tokens_single_hashtag:
    textFile.write(token+"\n")
textFile.close()

#for Japanese characters use https://tagul.com/cloud/1
#for others http://www.wordle.net/create


wordcloud = WordCloud().generate(hashedDoc)
# Open a plot of the generated image.
plt.imshow(wordcloud)
plt.axis("off")
plt.show()