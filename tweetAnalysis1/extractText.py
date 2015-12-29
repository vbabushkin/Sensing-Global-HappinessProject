__author__ = 'vahan'
import sys
import json
import time
import ast
import goslate
import os
gs = goslate.Goslate()
t0 = time.time()


folder = 'oct1516/tweetsByCountriesFull'

files=os.listdir (folder)
# #TRANSLATION
for currentFile in files:
    # load each tweet as json
    textFile= open("oct1516/text/text_"+currentFile, 'w')
    filename=folder+'/'+currentFile
    print filename
    for line in open(filename):
        tweet_json = ast.literal_eval(line)
        tweet_text = tweet_json['text'].encode('utf8')#.split()
        #print tweet_text
        try:
            translatedText=gs.translate(str(tweet_text), 'en')
            textFile.write(str(translatedText.encode('ascii', 'ignore'))+"\n")
        except:
            pass
        #print tweet_text
        #print translatedText
    textFile.close()

# for currentFile in files:
#     # load each tweet as json
#     textFile= open("oct1516/text/text_"+currentFile, 'w')
#     filename=folder+'/'+currentFile
#     print filename
#     for line in open(filename):
#         tweet_json = ast.literal_eval(line)
#         tweet_text = tweet_json['text'].encode('utf8')#.split()
#         #translatedText=gs.translate(str(tweet_text), 'en')
#         textFile.write(str(tweet_text)+"\n")
#         #print tweet_text
#     textFile.close()

t1 = time.time()

total = t1-t0

print "TOTAL TIME REQUIRED: "
print total