import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms as T
from torchvision import models
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from time import sleep

from utils import get_accuracy_bin

#train the model
def trainSiamens(net, tresholds, epochs, train_dataloader, device, optimizer, criterion, sch):
    print(net)
    loss=[] 
    counter=[]
    acc = np.zeros([len(tresholds),epochs])
    iteration_number = 0
    for epoch in range(1,epochs):
        acc_ep = np.zeros(len(tresholds))
        items = 0
        batch = 0
        for batch_id, smpl in enumerate(tqdm(train_dataloader)):
            sleep(0.1)
            img0, img1, label = smpl[0], smpl[1], smpl[2]
            img0, img1, label = img0.to(device), img1.to(device) , label.to(device)
            optimizer.zero_grad()
            output1,output2 = net(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()
            euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
            for m in range(len(tresholds)):
                label_t = euclidean_distance<tresholds[m]
                acc_ep[m] = accuracy_score(label, label_t)*len(label) + acc_ep[m]
            items = items + len(label)
            batch = batch + 1
        print("Epoch {}\n Current loss {}".format(epoch,loss_contrastive.item()))
        for m in range(len(tresholds)):
            print("Current accuracy {} for {}\n".format(acc_ep[m]/items,tresholds[m]))
            acc[m][epoch] = acc_ep[m]/items
        iteration_number += 1
        counter.append(iteration_number)
        loss.append(loss_contrastive.item())
        sch.step()
    plt.plot(counter, loss, label='Loss') 
    for m in range(len(tresholds)):
        plt.plot(counter, acc, label=str(tresholds[m]))
    plt.legend()
    return net

def trainBinary(net, tresholds, epochs, train_dataloader, device, optimizer, criterion, sch):
    print(net)
    loss=[] 
    counter=[]
    acc = np.zeros([len(tresholds),epochs])
    iteration_number = 0
    tresholds = [0.25, 0.5, 0.75]
    for epoch in range(1,epochs):
        acc_ep = np.zeros(len(tresholds))
        items = 0
        batch = 0
        for batch_id, smpl in enumerate(tqdm(train_dataloader)):
            sleep(0.1)
            img0, img1, label = smpl[0], smpl[1], smpl[2]
            img0, img1, label = img0.to(device), img1.to(device) , label.to(device)
            optimizer.zero_grad()
            output = net(img0,img1)
            loss_contrastive = criterion(output,label)
            loss_contrastive.backward()
            optimizer.step()
            for m in range(len(tresholds)):
                acc_ep[m] = get_accuracy_bin(label, output, tresholds[m])*len(label) + acc_ep[m]
            items = items + len(label)
            batch = batch + 1
        print("Epoch {}\n Current loss {}".format(epoch,loss_contrastive.item()))
        for m in range(len(tresholds)):
            print("Current accuracy {} for {}\n".format(acc_ep[m]/items,tresholds[m]))
            acc[m][epoch] = acc_ep[m]/items
        iteration_number += 1
        counter.append(iteration_number)
        loss.append(loss_contrastive.item())
        sch.step()
    plt.plot(counter, loss, label='Loss') 
    for m in range(len(tresholds)):
        plt.plot(counter, acc, label=str(tresholds[m]))
    plt.legend()
    return net

def trainClass(net, tresholds, epochs, train_dataloader, device, optimizer, criterion, sch):
    print(net)
    loss=[] 
    counter=[]
    acc = []
    iteration_number = 0
    for epoch in range(1,epochs):
        acc_ep = 0
        items = 0
        batch = 0
        for batch_id, smpl in enumerate(tqdm(train_dataloader)):
            sleep(0.1)
            img0, label = smpl[0], smpl[1]
            img0, label = img0.to(device), label.to(device)
            optimizer.zero_grad()
            output = net(img0)
            loss_contrastive = criterion(output,torch.reshape(label,[label.size()[0]]).type(torch.LongTensor))
            loss_contrastive.backward()
            optimizer.step()
            sm = torch.nn.Softmax()
            outputS = sm(output)
            _, predicted = torch.max(outputS, 1)
            acc_ep += (predicted == label).sum().item()
            items = items + len(label)
            batch = batch + 1
        print("Epoch {}\n Current loss {}".format(epoch,loss_contrastive.item()))
        print("Current accuracy {}".format(acc_ep/items))
        acc.append(acc_ep/items)
        iteration_number += 1
        counter.append(iteration_number)
        loss.append(loss_contrastive.item())
        sch.step()
    plt.plot(counter, acc)
    plt.plot(counter, loss)
    return net