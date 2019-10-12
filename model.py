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
    data = []
    in_data = DATA[0] 
    #right now all the songs are in one line so... 

    for i in range(1,len(in_data)-1):
        alph = ''
        while (in_data[i]!='\'' or in_data[i]!=' ' or in_data[i]!=',') and i>(len(in_data)-1):
            alph = alph + in_data[i]
            i = i + 1
        data.append(alph)
    data = tokenize(data)
    in_data = []
    for i in range(0, len(data),n):
        alph = []
        for z in range(i,i+n):
            if z>=len(data):
                break
            alph.append(data[z])
        in_data.append(alph)
    if len(in_data[-2])>len(in_data[-1]):
        del in_data[-1]
    in_data =  pt.Tensor(in_data)
    in_data.to(device)
    return in_data

file_n="data/ocarina.txt"
n=100 
#lenght of set of data
Data = getData(file_n,n)


