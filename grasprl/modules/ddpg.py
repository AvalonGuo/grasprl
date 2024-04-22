import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from prettytable import PrettyTable
from collections import namedtuple
import numpy as np
import random
import time

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

simple_Transition = namedtuple("simple_Transition", ("state", "action", "reward"))


def Transform_Image(means, stds):
    return T.Compose([T.ToTensor(), T.Normalize(means, stds)])




class ReplayBuffer(object):
    def __init__(self, size, simple=False):
        self.size = size
        self.memory = []
        self.position = 0
        self.simple = simple
        random.seed(20)

    def push(self, *args):
        if len(self.memory) < self.size:
            self.memory.append(None)
        if self.simple:
            self.memory[self.position] = simple_Transition(*args)
        else:
            self.memory[self.position] = Transition(*args)
        # If replay buffer is full, we start overwriting the first entries
        self.position = (self.position + 1) % self.size

    def sample(self, batch_size):
        rand_samples = random.sample(self.memory, batch_size - 1)
        rand_samples.append(self.memory[self.position - 1])
        return rand_samples

    def get(self, index):
        return self.memory[index]

    def __len__(self):
        return len(self.memory)





def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params} ({total_params/1000000:.2f}M)")

class Actor(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Actor,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=state_dim,out_channels=256,kernel_size=8,stride=4,padding=2) 
        self.conv2 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=4,stride=2)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=3)

        self.fc1 = nn.Linear(in_features=64*8*8,out_features=512)
        self.fc2 = nn.Linear(in_features=512,out_features=action_dim)
        self.relu = nn.ReLU()
        # self.max_action = max_action
    def forward(self,x):
        """
        @size_transformation
        -input: [batch_num,4,200,200]
        -conv1: [batch_num,256,50,50]
        -conv2: [batch_num,128,24,24]
        -conv3: [batch_num,64,8,8]
        -fc1: [batch_num,512]
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x).unsqueeze(1)
        return x.float()
    
class Critic(nn.Module):
    def __init__(self,state_dim):
        super(Critic,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=state_dim,out_channels=160,kernel_size=20,stride=8)  #[1,160,23,23]
        self.conv2 = nn.Conv2d(in_channels=160,out_channels=80,kernel_size=9,stride=7)        #[1,80,3,3]

        self.fc1 = nn.Linear(in_features=241*3,out_features=81)
        self.fc2 = nn.Linear(in_features=81,out_features=1)
        self.relu = nn.ReLU()
    def forward(self,state,action,batch_size):
        state = self.relu(self.conv1(state))
        state = self.relu(self.conv2(state)).view(batch_size,-1,3)

        q = self.relu(self.fc1(torch.cat([state,action],1).view(batch_size,-1)))
        q = self.relu(self.fc2(q))
        return q





