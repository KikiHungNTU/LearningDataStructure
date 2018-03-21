# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 11:49:43 2018

@author: Ouch
"""
"""
Practicing for quiz only with standard library
"""
import itertools as itert
import math

A = [3, 5, 7, 9, 11, 2, 1]
##Q0
#Given A
#Find smallest missing posistive value with O(N)
def solution(A):
    nums = A
    n = len(nums)
    for i in range(n):
        while nums[i] > 0 and nums[i] <= n and nums[i] != i + 1 and nums[i] != nums[nums[i] - 1]:
            nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
        # print nums
    for i in range(n):
        if i + 1 != nums[i]:
            return i + 1
    return n + 1

##Q1
#Given 4 numbers
#Find the possible max distance between two points
def find_missing(A):   
    permutate =itert.permutations(A,4)
    dis=[]
    for a , b , c ,d in permutate:
        dis.append( math.sqrt((a - c)**2 +( d-b)**2) ) 
    ans1 = max(dis)
    return ans1

##Q2
#Given uncontinous sequence
#Find possible max sum of K consecutive and L consecutive numbers without overlapped
K = 3
L = 2
def find_consecutive_sum(A,K,L):   
    kSums = []
    lSums = []
    k_l_Sum=[]
    if K + L > len(A):
        ans2 = -1
    else:
        for k in range(len(A) - K+1):
            kSums.append( [ k, (k+ k-1), sum(A[k:k+k]) ] )
        for l in range(len(A) - L+1):
            lSums.append( [ l, (l+ l-1), sum(A[l:l+l]) ] )
        for kSum, lSum in itert.product(kSums, lSums):
            k_start, k_end, k_sum = kSum
            l_start, l_end, l_sum = lSum               
            if k_start > l_end or l_start > k_end:
                k_l_Sum.append(k_sum + l_sum)
        ans2 = max(k_l_Sum)
    return ans2
    

##Q3
#Given graph T
#find out all distances between the start and visited
import collections
def bfs_distance(T):
    adj_list = collections.defaultdict(list)
    for i in range(len(T)):
        if T[i] == i:
            capital = i
        adj_list[i].append( T[i] )
        adj_list[ T[i] ].append( i )
    ans3 = [ 0 for _ in T ]  
    visited , q = set(), [(0,capital)]
    while q:
        dis, vertex = q.pop(0)
        ans3[dis] += 1
        visited.add(vertex)
        for v in adj_list[vertex]:
            if v not in visited:
                q.append( ( dis+1, v ) )
    ans3.pop(0)
    return ans3