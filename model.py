import torch.nn as nn
import torch
from random import randint
import sys
#import torchvision.transform as transforms
#import torchvision.datasets as dsets

### GPU

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on CUDA")
else:
    device = torch.device("cpu")


def creator(data,n_keys):
    print("Converting to tensor tensor")
    data = torch.Tensor(data)
    print("Tensor Shape : ",data.shape)
    print("Moving files to device")
    data.to(device)
    
    ### Instantiate the model
    print("Instantiating Model")
    output_dim = n_keys
    input_dim = n_keys
    layer_dim = 1                       # why?
    hidden_dim = 100
    model = LSTMModel(input_dim,hidden_dim,layer_dim,output_dim)
    model.to(device)
    
    ### Instantiate loss class
    criterion = nn.CrossEntropyLoss()

    ### Instantiate Optimizer Class
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    ### Instantiate Training
    print("Training model")
    train(model,criterion,optimizer,data,n_keys)
    
    ### Generating the new Data
    print("Generating data")
    new_data = generate(model,n_keys)
    return new_data.tolist()


class LSTMModel(nn.Module):
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
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # One time step
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out

### Train the model
def train(model,criterion,optimizer,data,n_keys):
    epochs = 500
    for epoch in range(epochs):
        # Iterate over dataset
        for datapoint in data:
            # Prepare data points
            label = datapoint[-1]
            datapoint = datapoint[:-1]
            datapoint.requires_grad_()
            #print(datapoint.shape)
            datapoint = datapoint.view(-1,99,n_keys)
            datapoint = datapoint.to(device)
            #label = label.view(-1,n_keys)
            index_label = index_finder(label)
            index_label = torch.Tensor([index_label])
            index_label = index_label.type(torch.LongTensor)
            index_label = index_label.to(device)

            # Clean optimizer
            optimizer.zero_grad()
            
            # Get output from model
            output = model(datapoint)
            #output = output.squeeze()

            # Calculate loss
            #print(output.shape,index_label)
            loss = criterion(output,index_label)

            # Calculate gradient
            loss.backward()

            # Update Gradient
            optimizer.step()
        print("Epoch : ",epoch," Done")
    return

### Generate a raw data
def generate(model,n_keys):
    random_nintynine = []
    # The first 99 which are randomly made, and on basiis of model it will genrate new ones

    for i in range(99):
        pos = randint(0,n_keys-1)
        #print(pos)
        temp = []
        for j in range(n_keys):
            if j==pos:
                temp.append(1)
            else:
                temp.append(0)
        random_nintynine.append(temp)
    random_nintynine = torch.Tensor(random_nintynine)
    random_nintynine.to(device)
    for i in range(901):
        model_input = random_nintynine[i:]
        #print(model_input.shape)
        model_input = model_input.view(-1,99,n_keys)
        model_input = model_input.to(device)
        #print(model_input.shape)
        generated_one = model(model_input)
        generated_one = clean_output(generated_one)
        #print(generated_one.shape,random_nintynine.shape)
        random_nintynine = torch.cat((random_nintynine,generated_one),dim=0)
        #adding the newly generated tokenized note to the tokenized music set
    return random_nintynine

### Output cleaner
# It cleans the output to give the desired dic value
def clean_output(output):
    maxF = 0
    output = output[0].tolist()
    for i in range(len(output)):
        if output[i]>output[maxF]:
            maxF=i
    new_output = []
    for i in range(len(output)):
        if i == maxF:
            new_output.append(1)
        else:
            new_output.append(0)
    return torch.Tensor(new_output).unsqueeze(dim=0)

def index_finder(label):
    maxF = 0
    label = label.tolist()
    if len(label)==1:
        label = label[0]
    for i in range(len(label)):
        if label[i]>label[maxF]:
            maxF = i
    return maxF
