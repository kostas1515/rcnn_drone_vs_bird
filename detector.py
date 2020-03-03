import faster_rcnn
import train_faster_rcnn as trainer
import torchvision
import torch.optim as optim
import pandas as pd
import time
import sys
import timeit
import torch.autograd
from dataset import *
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np

net = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
net.to(device)

drone_size='large+medium'
print('training for '+ drone_size+'\n')
transformed_dataset=DroneDatasetCSV(csv_file='../annotations.csv',
                                           root_dir='../images/images/',
                                           drone_size=drone_size,
                                           transform=transforms.Compose([
                                               ResizeToTensor(800)
                                           ]))
try:
    PATH = './faster_rcnn.pth'
    weights = torch.load(PATH)
    net.load_state_dict(weights)
except FileNotFoundError:
    print('going vanilla')

optimizer = optim.Adam(net.parameters(), lr=0.00001,weight_decay=0.0005)

dataset_len=(len(transformed_dataset))
print('Length of dataset is '+ str(dataset_len)+'\n')
batch_size=4

dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)
epochs =100
for i in range(epochs):
    print(i)
    trainer.train_one_epoch(net,optimizer,dataloader,device,i,20)
    torch.save(net.state_dict(), PATH)