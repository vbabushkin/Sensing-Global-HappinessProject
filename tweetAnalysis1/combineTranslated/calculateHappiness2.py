__author__ = 'vahan'


readfile = open("happiness_index_rm4to6.txt", "r")
ToBeHappy = open("text_extractedTweetsFromAlgeria.txt", "r")


dictList={}
outputHappiness = []

for row in readfile: # in the main words ranking!
    field=row.strip()
    field=field.split("\t") # seperated by tab? words perhaps
    dictList[field[0]]=float(field[1]) #ok the word and its scale!

print dictList

for line in ToBeHappy:
    wordScore = 0
    for word in line.split():
        if word in dictList:
               wordScore = wordScore+  dictList[word]
               print word
               print dictList[word]
               print wordScore
    outputHappiness.append(wordScore)
print outputHappiness
len(outputHappiness)