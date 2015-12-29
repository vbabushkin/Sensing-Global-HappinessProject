__author__ = 'vahan'
from collections import Counter
import re
import matplotlib.pyplot as plt

folder = 'sampleStratum2'
country='Japan'
filename='processedrawTextCombinedFor'+country+'.txt'
tweetFile=folder+'/'+filename


words = re.findall('\w+', open(tweetFile).read().lower().decode('utf-8') )
Counter(words).most_common(50)


