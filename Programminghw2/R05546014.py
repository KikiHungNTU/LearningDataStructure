# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 20:09:20 2017

@author: Ouch
"""

import csv

#讀檔
with open('Input', 'r') as inputFile:
    case = [[int(x) for x in data] for data in csv.reader(inputFile, delimiter=',')]
Number = len(case)
#height = case[0]
def ComputeAnswer(height):
    stack=[]; i=0; area=0
    while i<len(height):
      #如果stack null or i 的高度 高於現存最大高度
      #stack[len(stack)-1]  高才存進來 矮抓走
      if stack==[] or height[i]>height[stack[len(stack)-1]]:
          stack.append(i)
      else:
          curr=stack.pop()
          width=i if stack==[] else i-stack[len(stack)-1]-1
          area=max(area,width*height[curr])
          i-=1
      i+=1
      #stack算完了
    while stack!=[]:
           curr=stack.pop()
           width=i if stack==[] else len(height)-stack[len(stack)-1]-1
           area=max(area,width*height[curr])
    return area

 #寫答案      
outputFile = open('Output.txt', 'w')
for i in range(0 ,Number):
    if i == Number-1:
        outputFile.write(str(ComputeAnswer(case[i])))
    else:
        outputFile.write(str(ComputeAnswer(case[i])) + '\n')
    
 