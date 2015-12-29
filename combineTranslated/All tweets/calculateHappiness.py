__author__ = 'vahan'

import re
#input is each countries translated tweets
datainput=open("text_extractedTweetsFromAlgeria.txt", "r")


readfile = open("happiness_index_rm4to6.txt", "r")
outfile = open('result-it-orinal-Algeria.txt','w')
outfile2 = open('result-it-orinal-Algeria-englishTweets.txt','w')
dictList={}
saveid=[]
countemty=0
count0=0

#check whether keyid is a number?
def check1000(keyid):
    if not keyid.isdigit():
        return False
    else:
        if int(keyid)>1000:
            return True
        else:
            return False

#Return a copy of the string with the leading and trailing characters removed
for row in readfile: # in the main words ranking!
    field=row.strip()
    field=field.split("\t") # seperated by tab? words perhaps
    dictList[field[0]]=float(field[1]) #ok the word and its scale!

for mystr in datainput: # here is the main text files
    wordList = re.sub("[^\w]", " ",  mystr.lower()).split() #re.sub(pattern, repl, string, count=0, flags=0) Return the string obtained by replacing the leftmost non-overlapping occurrences of pattern in string by the replacement repl.
    #print wordList
    tweetid=wordList[0]

    while  not check1000(tweetid):
        #print wordList
        wordList=wordList[1:]
        tweetid=wordList[0]


    saveid.append(wordList[0])
    temp=0
    temppo=0
    tempne=0
    count=0
    countpo=0
    countne=0
    savepo=[]
    savene=[]
    if len(wordList)<3:#no content tweets
        countemty=countemty+1
    else:
        for i in range(1,len(wordList)):
            if dictList.has_key(wordList[i]):
                temp=temp+dictList[wordList[i]]
                count=count+1
                if dictList[wordList[i]]>=0:
                    countpo=countpo+1
                    temppo=temppo+dictList[wordList[i]]
                    savepo.append(wordList[i])
                else:
                    countne=countne+1
                    tempne=tempne+dictList[wordList[i]]
                    savene.append(wordList[i])


            else:
                pass
        if count==0:
            count0=count0+1
        else:
            average=(temp/count)
            #outfile.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (wordList[0],average,temp,temppo,tempne,count,countpo,countne,savepo,savene))
            outfile.write("%s,%s,%s,%s\n" % (wordList[0],average,count,savepo))
            outfile2.write("%s,%s\n" % (wordList[0],average))

outfile.write('countemty= %s\n' %countemty)
outfile.write("count0= %s\n" %count0)
outfile2.write('countemty= %s\n' %countemty)
outfile2.write("count0= %s\n" %count0)
outfile.close()
outfile2.close()
readfile.close()
datainput.close()