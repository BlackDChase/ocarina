import torch.nn as nn
import torch as pt
import numpy as np

#import torchvision.transform as transforms
#import torchvision.datasets as dsets

### GPU
if pt.cuda.is_available():
    device = pt.device("cuda:0")
    print("Running on CUDA")
else:
    device = pt.device("cpu")


def creator(data,n_keys):
    print("Converting to tensor tensor")
    data = pt.Tensor(data)
    print("Tensor Shape : ",data.shape)
    print("Moving files to device")
    data.to(device)
    
    ### Instantiate the model
    output_dim = n_keys
    input_dim = n_keys
    layer_dim = 1                       # why?
    hidden_dim = 100
    model = LSTMModel(input_dim,hidden_dim,layer_dim,output_dim)

    ### Instantiate loss class
    criterion = nn.CrossEntropyLoss()

    ### Instantiate Optimizer Class
    learning_rate = 0.1
    optimizer = pt.optim.SGD(model.parameters(), lr=learning_rate)
    
    ### Instantiate Training
    new_data = train(model,criterion,optimizer,data)

    print("Converting tensor to list")
    new_data.tolist()
    return new_data           


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
def train(model,criterion,optimizer,data):
    #code
    return data
### Wrapper function
