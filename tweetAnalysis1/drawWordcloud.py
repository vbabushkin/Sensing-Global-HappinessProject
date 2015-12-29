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
country="United Arab Emirates"
filename="oct1516/text/text_extractedTweetsFrom"+country+".txt"
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

all_tokens = sum(texts, [])

#AGAIN eliminate words with one letter
tokens_single_letter = set(word for word in set(all_tokens) if len(word) == 1)
#tokens_once = set(word for word in set(all_tokens) if len(word) >1)

texts = [[word for word in text if word not in tokens_once and word not in tokens_single_letter]for text in texts]

#empty Arrays Removed from Texts
texts=[text for text in texts if len(text)!=0]

all_tokens = sum(texts, [])

doc=''
#concatenate all our texts:
for token in all_tokens:
    doc=doc+' '+token

######################################################################################################
#save to file for http://www.wordle.net/create
folder="oct1516/textFromAllTweetsByCountries"

textFile= open(folder+"/text_"+country+".txt", 'w')
for token in all_tokens:
    textFile.write(token+"\t")
textFile.close()


wordcloud = WordCloud().generate(doc)
# Open a plot of the generated image.
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
