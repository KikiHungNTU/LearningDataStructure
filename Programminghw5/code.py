# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 21:37:29 2017

@author: Ouch
"""

def factorial(f): #計算階層
	if f == 0:
		return 1
	else :
		temp = 1
		for i in range(1,f+1):
			temp = i*temp
		return temp

def number(n): #計算出可能的數量
	Catalan = factorial(2*n)/(factorial(n+1)*factorial(n))
	return int(Catalan)

def dynamic_programming(i,j):
	temp = 1
    #root
	for x in range(i,j+1):
		for left in range(1,number(x-i)+1):
			for right in range(1,number(j-x)+1):
				if tree[i][x-1][left] != None and tree[x+1][j][right]!= None:
					tree[i][j][temp] = str(x) + "," + tree[i][x-1][left] + "," + tree[x+1][j][right]
				elif tree[i][x-1][left] == None and tree[x+1][j][right]!= None:
					tree[i][j][temp] = str(x) + "," + tree[x+1][j][right]
				elif tree[i][x-1][left] != None and tree[x+1][j][right]== None:
					tree[i][j][temp] = str(x) + "," + tree[i][x-1][left]
        #都空 就是自己
				else:
					tree[i][j][temp] = str(x)
				temp = temp + 1

def build_dynamic_programming(k):
	for i in range(1,k+1):
		for j in range(k,i-1,-1):
			dynamic_programming(k-j+1,k-j+i)

outputFile = open('put','w')
with open("Input", "r") as inputFile:
	for line in inputFile:
		k = int(line)
    #長寬k+1，高Catalan
		tree = [[[None]*(number(k)+1)  for i in range(k+2)]for j in range(k+2)]
		build_dynamic_programming(k)
		outputFile.write(str(number(k)) + '\n')
		for i in range(1,number(k)+1):
			outputFile.write(str(tree[1][k][i]) + '\n')
outputFile.close()