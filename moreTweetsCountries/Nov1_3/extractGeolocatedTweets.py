#!/usr/bin/env python

# @dpmehta02
# Coursera Data Science HW1 - Script analyzes tweets for sentiment (negativity/positivity)
# Requires sentiment file (e.g., AFINN-111.txt: https://code.google.com/p/fb-moody/source/browse/trunk/AFINN/AFINN-111.txt?spec=svn2&r=2)
# USAGE: $ python tweet_sentiment.py <sentiment_file> <tweet_file>

import sys
import json
import time
import ast

#nov 5-6
OUTPUT_ID_START=517
OUTPUT_ID_END=573

def main():
    t0 = time.time()

    # load each tweet as json
    filtered = open("/media/vahan/DATAPART1/tweets/moreTweetsCountries/Nov1_3/filteredGeolocatedTweets_Nov_1_3", 'w')


    #CHANGE IT BACK
    for i in range(OUTPUT_ID_START,OUTPUT_ID_END):
        name="/media/vahan/DATAPART1/tweets/collectedTweets/output"+str(i)
        print name
        for line in open(name):
            tweet_json = json.loads(line)
            # only accept records with a 'text' field
            if tweet_json.get('coordinates'):
                filtered.write(str(tweet_json)+"\n")
    filtered.close()

    # #read the saved dictionaries:
    # for line in open('filteredGeolocatedTweets', 'r'):
    #         tweet_json = ast.literal_eval(line)
    #         tweet_text = tweet_json['text'].encode('utf8')#.split()
    #         tweet_coord= tweet_json['coordinates']['coordinates']
    #         print tweet_text
    #         print "############################################"
    #         print tweet_coord


    #save coordinates only in separate file


    coordFile = open("/media/vahan/DATAPART1/tweets/moreTweetsCountries/Nov1_3/collectedTweetsCoordinates_Nov_1_3.txt", 'w')
    for line in open('/media/vahan/DATAPART1/tweets/moreTweetsCountries/Nov1_3/filteredGeolocatedTweets_Nov_1_3', 'r'):
            tweet_json = ast.literal_eval(line)
            tweet_coord= tweet_json['coordinates']['coordinates']
            coordFile.write(str(tweet_coord[0])+"\t"+str(tweet_coord[1])+"\n")

    t1 = time.time()

    total = t1-t0

    print "TOTAL TIME REQUIRED: "
    print total

if __name__ == '__main__':
  main()
