# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 23:56:00 2017

@author: Ouch
"""

import numpy as np

outputFile = open('output.txt', 'a')
##讀檔
with open('input', 'r') as inputFile:
    for line in inputFile.readlines():
        data = line.split()

        #讀手機、樓層數
        phone = data[0];
        floor = data[1];
        print(phone, floor);
        
        total_Floor = 0;
        count = 0;
        
        #把字串轉為數字
        phone = int(phone);
        floor = int(floor);
        
        while total_Floor < floor:
            count = count + 1;
            f = np.zeros((phone+1, count+1), dtype=np.int64)
            trial = np.zeros(count+1, dtype=np.int64)
    
            #手機壞了,只剩(手機-1)摔壞、f(手機,摔次數)=樓層
            for i in range(1, count+1):
                for j in range(1, phone+1):
                    #每一樓層 手機從1支到phone支
                    f[j][i] = f[j-1][i-1] + f[j][i-1] + 1;
    
            #存路徑 摔壞1支 少1次
            for k in range(count-1, -1, -1):
                trial[k] = f[phone-1][k] + 1;
                    
            for k in range(count-2, -1, -1):
                trial[k] = trial[k] + trial[k+1];
            

    
            total_Floor = f[phone][count]           
                        
            if total_Floor >= floor:
                print(count, '\n')
                outputFile.write(str(count) + '\n')
                
                for i in range(count-1, 0, -1):
                    print(trial[i], ' ')
                    outputFile.write(str(trial[i]) + ' ')
                    
                print(trial[0])
                outputFile.write(str(trial[0]) + '\n')