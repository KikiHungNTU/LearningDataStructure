# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 20:55:30 2017

@author: Ouch
"""
def numDecodings(s):
    if s=="" or s[0]=='0': return 0
    dp=[1,1]
    for i in range(2,len(s)+1):
        if int(s[i-2:i])==10 or int(s[i-2:i])==20:
            dp.append(dp[i-2])
        elif 10 <=int(s[i-2:i]) <=26 and s[i-1]!='0':
            dp.append(dp[i-1]+dp[i-2])
        elif s[i-1]!='0':
            dp.append(dp[i-1])
        else:
            return 0
    return dp[len(s)]

import csv
data = []
Output = []
#讀檔
with open('input', 'r') as inputFile:
    row = csv.reader(inputFile)
    for i in row :
        data.append(i)
#Output        
for i in range(len(data)):    
    Output.append(numDecodings(data[i][0]))
outRow = len(Output)
outputFile = open('Output','w') 
for i in range(outRow):
    if i <outRow-1:
        outputFile.write(str(Output[i])+"\n")
    else:
        outputFile.write(str(Output[i]))
outputFile.close() 