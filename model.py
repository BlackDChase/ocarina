import torch as pt
from random import random
import sys

if pt.cuda.is_available():
    device = pt.device("cuda:0")
else:
    device = pt.device("cpu")

dic = {}
#tokenize data dict

def tokenize(data):
    print("yo",data)
    Data = []
    for i in data:
        print("\n>>",i)
        sys.exit()
        if i not in dic.keys():
            x = 1
            while x in dic.values():
                x = (float(random())*10000)%10000
            dic[i] = x
        Data.append(dic[i])
    return Data
#Using random Function to make data more distinct


def getData(file_n,n):
   # in_data = []

    notes = []
    #Every line will have notes of 1 midi file

    f_notes = open(file_n,"r")
    for lines in f_notes:
        notes.append(lines)
    #print(len(notes)) 
    inputData = []
    for in_data in notes:
        # when many songs seprated by lines
        if len(in_data) < (n+(n//2)):
            continue
        music = []
        for i in range(0,len(in_data)):
            beat = ''
            while (in_data[i]!='\'' or in_data[i]!=' ' or in_data[i]!=',' or in_data[i]!='[' or in_data[i]!=']') and i>(len(in_data)-1):
                beat = beat + in_data[i]
                i = i + 1
            music.append(beat)
        nMusic = tokenize(music)
        music = []
        # len(music) is number of label available for each music
        for i in range(len(nMusic) - n):
            workset = []
            for j in range(i,i+n):
                workset.append(nMusic[j])
            music.append(workset)
        for i in range(0,(len(music)//(n//2 + 1)),(n//2 + 1) ):
            label = []
            if (i + n//2 + 1)>len(music):
                continue
            for k in range(i,i+(n//2)+1):
                label.append(music[k])
            inputData.append(label)
    #for loop to be closes for many song data
    
    print(len(inputData),len(inputData[0]),len(inputData[0][0]),inputData[0][0])
    print(type(inputData),type(inputData[0]),type(inputData[0][0]))
    print(dic)
    #inputData =  pt.Tensor(inputData)
    #print(inputData.size(),type(inputData))
    #inputData.to(device)
    return inputData

file_n="data/ocarina.txt"
n=100 
#lenght of set of data
Data = getData(file_n,n)


