import torch as pt
from random import random


if pt.cuda.is_available():
    device = pt.device("cuda:0")
else:
    device = pt.device("cpu")

dic = {}
#tokenize data dict

def tokenize(data):
    Data = []
    for i in data:
        if i not in dic.keys():
            x = 1
            while x in dic.values():
                x = (float(random())*10000)%10000
            dic[i] = x
        Data.append(dic[i])
    return Data

def getData(file_n,n):
    in_data = []
    DATA = []
    notes = open(file_n,"r")
    for lines in notes:
        DATA.append(lines)
    
    in_data = DATA[0] 
    #right now all the songs are in one line so... 
    
    F_DATA = []
    #for in_data in DATA:
    # when many songs seprated by lines
    
    data = []
    for i in range(1,len(in_data)-1):
        alph = ''
        while (in_data[i]!='\'' or in_data[i]!=' ' or in_data[i]!=',') and i>(len(in_data)-1):
            alph = alph + in_data[i]
            i = i + 1
        data.append(alph)
    data = tokenize(data)
    final_data = []
    for i in range(0, len(data),n):
        alph = []
        for z in range(i,i+n):
            if z>=len(data):
                break
            alph.append(data[z])
        final_data.append(alph)
    
    if len(final_data[-2])>len(final_data[-1]):
        del final_data[-1]
    #F_DATA.append(final_data)
    #for loop to be closes for many song data

    F_DATA = in_data
    F_DATA =  pt.Tensor(F_DATA)
    F_DATA.to(device)
    return F_DATA

file_n="data/ocarina.txt"
n=100 
#lenght of set of data
Data = getData(file_n,n)


