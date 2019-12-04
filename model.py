import torch.nn as nn
import torch as pt
import convert
import numpy as np
from random import randint

#import torchvision.transform as transforms
#import torchvision.datasets as dsets

from random import random
import sys
import re
import music21

if pt.cuda.is_available():
    device = pt.device("cuda:0")
    print("Running on CUDA")
else:
    device = pt.device("cpu")
#GPU


### start
dic = {}

#def getData(notes,n):
    













'''
class LSTMModel(nn.Module):
    #Source : https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        
        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out
'''

def make_dictonary(keys,dic):
    for i in range(len(keys)):
        temp = []
        for z in range(len(keys)):
            if(z==i):
                temp.append(1)
            else:
                temp.append(0)
        dic[str(keys[i])] = temp

def transform(unorganised_data):
    data = []
    it = -1
    for music_file in unorganised_data:
        diced_data = []
        randomized_data = []
        check_list = []
        it +=1
        if len(music_file)<199:
            #to be able to have atleast 100 sets of 100 notes
            print("File too small : ",it)
            continue
        print("Tokenising file no : ",it)
        tokenized_notes = []
        for notes in music_file:
            tokenized_notes.append(dic[str(notes)])
        for i in range(len(music_file)-99):
            #for making sets of 100 notes
            noteSet = []
            for j in range(i,i+100):
                noteSet.append(tokenized_notes[j])
            diced_data.append(noteSet)
        for i in range(100):
            random_number = randint(0,len(diced_data)-1)
            while(random_number in check_list):
                random_number = randint(0,len(diced_data)-1)                
            check_list.append(random_number)
            randomized_data.append(diced_data[random_number])
        data.append(randomized_data)
    print("Convertion into Tensor")    
    data = pt.Tensor(data)
    print("Transferring to device")
    data.to(device)
    return data


n_files_to_load = 10
keys = []
data, keys = convert.parse_to_notes(n_files_to_load)
#if no argument passed, it will do it for all values

make_dictonary(keys,dic)
data = transform(data)

print("Tensor Shape : ",data.shape)
sys.exit()






batch = 5           #why??
n_iters = 1000      #why??
num_epochs = int(n_iters/ len(Data) / batch)
#print(num_epochs)
train_loader = pt.utils.data.DataLoader(dataset=Data, batch_size=batch, shuffle=True)
                    #test loader??
input_dim = 100     #why??
hidden_dim = 100    #why??
layer_dim = 1       #why??
output_dim = 1    #the number of notes found through dict = 286, but we are asked to have only 1, to forward it

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1 #why??
optimizer = pt.optim.SGD(model.parameters(), lr=learning_rate) 
                    #why??
'''
for i in range(len(list(model.parameters()))):
    print(">>",list(model.parameters())[i].size())
# Number of steps to unroll
'''
seq_dim = 51        #why?? he used 28 , input dim he chose 28.. so i am choosing acordingly
codes = []
for i, (notes, labels) in enumerate(train_loader):

    notes = notes.view(-1, seq_dim, input_dim).requires_grad_()
    optimizer.zero_grad()
    outputs = model(notes)
    codes.append(output)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    break
print(codes)
playMusic(codes)

'''
### The orignal implimentation of LSTM
# This is where the model is trained
iter = 0
for epoch in range(num_epochs):
    for i, (notes, labels) in enumerate(train_loader):
        # Load images as a torch tensor with gradient accumulation abilities
        notes = notes.view(-1, seq_dim, input_dim).requires_grad_()

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        # outputs.size() --> 100, 10
        outputs = model(notes)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        if iter % 200 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for notes, labels in test_loader:
                # Resize images
                notes = notes.view(-1, seq_dim, input_dim)

                # Forward pass only to get logits/output
                outputs = model(notes)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += labels.size(0)

                # Total correct predictions
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
'''
