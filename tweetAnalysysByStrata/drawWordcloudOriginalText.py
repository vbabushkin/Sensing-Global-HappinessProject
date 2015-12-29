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
country="Spain"
filename="oct1516/textOnly/text_extractedTweetsFrom"+country+".txt"
documents=[]

for line in open(filename):
    print line;
    #remove all https
    line=re.sub("(?P<url>https?://[^\s]+)", "", line)
    line = re.sub(r'[?|$|.|!|@|#]',r'', line)
    #leave words only
    #line = re.sub(r"[^A-Za-z\s]", "", line.strip())
    words = line.split()
    documents=documents+words

# to remove duplicated words
documents=list(set(documents))
print len(documents)

# remove common words and tokenize
#stoplist = set('for a of the and to in'.split())

#STOP_WORDS = nltk.corpus.stopwords.words('english')
texts = [[word.decode('utf-8') for word in document.lower().split()] for document in documents]

#empty Arrays Removed from Texts
texts=[text for text in texts if len(text)!=0]


#make one aray
all_tokens = sum(texts, [])

# # remove words that appear only once
# tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
#
# #eliminate words with one ore two letters
# tokens_single_letter = set(word for word in set(all_tokens) if len(word) <= 2)
#tokens_once = set(word for word in set(all_tokens) if len(word) >1)

# texts = [[word for word in text if word not in tokens_once and word not in tokens_single_letter]for text in texts]
#
# #empty Arrays Removed from Texts
# texts=[text for text in texts if len(text)!=0]

# #lemmatize words
# lmtzr = WordNetLemmatizer()
# texts=[[lmtzr.lemmatize(text[0])] for text in texts]
#######################################################################################################
doc=''
#concatenate all our texts:
for token in all_tokens:
    doc=doc+' '+token

######################################################################################################
#save to file for http://www.wordle.net/create
folder="oct1516/originalTextFromAllTweetsByCountries"

textFile= open(folder+"/text_"+country+".txt", 'w')
for token in all_tokens:
    print token
    textFile.write(token.encode('utf-8')+"\t")
textFile.close()


wordcloud = WordCloud().generate(doc)
# Open a plot of the generated image.
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
