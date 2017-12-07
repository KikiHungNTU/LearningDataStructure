# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 15:20:45 2017

@author: Ouch
"""

import numpy as np
#讀檔
stockInf = []
idx = -1
with open('Input', 'r') as inputFile:
    for i in inputFile.readlines() :
        if i !=[]:
            i = i.strip('\n')
            stockInf.append([])
            idx = idx+1
            stockInf[idx] = i
#分割每筆資料                  
def get_stock(stockInf,number):
    stock = []
    stock.append(stockInf[number].split(','))
    return stock

#計算利潤
def computeProfit(stock):
    stock = np.array(stock)
    minPrice = int(stock[0,0])
    bestProfit = 0
    day = len(stock[0])
    #找最低股價買
    for i in range(day):
        stock_today = int(stock[0,i])
        if stock_today < minPrice:
            minPrice = stock_today
        #找最高點賣
        elif (stock_today - minPrice) > bestProfit:
            bestProfit = stock_today - minPrice
            sellDay = i
    #回去對是哪天買的
    for i in range(day-1,-1,-1):
        stock_today = int(stock[0,i])
        stock_sell = int(stock[0,sellDay])
        if stock_today == stock_sell - bestProfit:
            buyDay = i
    #answer = []
    #answer.append(buyDay)
    #answer.append(sellDay)
    #answer.append(bestProfit)
    return buyDay,sellDay,bestProfit

numberofData = len(stockInf)                    
#answer = []
#idx = 0
outputFile = open('Output','w') 
for i in range(numberofData):
    stock = get_stock(stockInf,i)
    #answer.append([])
    #answer[idx] = str(computeProfit(stock))
    b , s , p = computeProfit(stock)
    #idx = idx+1
    if i < numberofData-1:
        outputFile.write(str(b)+','+str(s)+','+str(p)+'\n')
    else:
        outputFile.write(str(b)+','+str(s)+','+str(p))
outputFile.close()

