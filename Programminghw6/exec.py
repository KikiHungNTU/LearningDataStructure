# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 00:46:18 2018

@author: Ouch
"""

#讀檔
data = []
idx = -1
with open('input', 'r') as inputFile:
    for i in inputFile.readlines() :
        if i !=[]:
            i = i.strip('\n')
            data.append([])
            idx = idx+1
            data[idx] = i
#抓k、card
k=[]
card = []
number = len(data)
for i in range(number):
    k.append( data[i].split()[0] )
    card.append([])
    card[i] = data[i].split()[1:]
    card[i].sort(key = int)

#找第k個min
def computeAns(cards, k):
    afterCompute = []
    for i in range(len(cards)):
        for j in range(i+1,len(cards)):
            answer = int(cards[j]) - int(cards[i])
            afterCompute.append(answer)
    afterCompute.sort(key = int)
    k_minimum = afterCompute[k-1]
    return k_minimum

#寫答案
answers = []
outputFile = open('output','w') 
for i in range(number):
    a = computeAns(card[i],int(k[i]) )
    answers.append(a)
    if i < number-1:
        outputFile.write(str(a)+'\n')
    else:
        outputFile.write(str(a))
outputFile.close() 
