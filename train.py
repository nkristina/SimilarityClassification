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

from sklearn import  metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from utils import get_accuracy_bin

def trainSiamens(net, tresholds, epochs, train_dataloader, device, optimizer, criterion, sch, val_dataloader, best_tr=1):
    loss=[] 
    loss_v=[]
    counter=[]
    counter_v=[]
    early_stop = 0
    acc = np.zeros([len(tresholds),epochs])
    acc_val = np.zeros([len(tresholds),epochs])
    iteration_number = 0
    iteration_number_v = 0
    out1, out2, out3, out4, out5, lab = [], [], [], [], [], []
    for epoch in range(0,epochs):
        acc_ep = np.zeros(len(tresholds))
        items = 0
        batch = 0
        out1.clear(), out2.clear(), out3.clear(), out4.clear(), out5.clear()
        lab.clear()
        net.train()
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
            out1.append(euclidean_distance<0.75), out2.append(euclidean_distance<1.0), out3.append(euclidean_distance<1.25)
            out4.append(euclidean_distance<1.5), out5.append(euclidean_distance<1.75), lab.append(label)
        print("Epoch {}\n Current loss {}".format(epoch,loss_contrastive.item()))
        for m in range(len(tresholds)):
            print("Current accuracy {} for {}\n".format(acc_ep[m]/items,tresholds[m]))
            acc[m][epoch] = acc_ep[m]/items
        iteration_number += 1
        counter.append(epoch)
        loss.append(loss_contrastive.item())

        outputs = np.concatenate(out3)
        targets = np.concatenate(lab)
        f1 = f1_score(outputs, targets)
        print("F1 score {} for 1".format(f1))

        sch.step()

        net.eval()
        acc_v = np.zeros(len(tresholds))
        items = 0
        batch = 0
        for o in range(0,3):
            for batch_id, smpl in enumerate(tqdm(val_dataloader)):
                sleep(0.1)
                img0, img1, label = smpl[0], smpl[1], smpl[2]
                img0, img1, label = img0.to(device), img1.to(device) , label.to(device)
                output1,output2 = net(img0,img1)
                loss_contrastive = criterion(output1,output2,label)
                euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
                for m in range(len(tresholds)):
                    label_t = euclidean_distance<tresholds[m]
                    acc_v[m] = accuracy_score(label, label_t)*len(label) + acc_v[m]
                items = items + len(label)
                batch = batch + 1
        print("Epoch {}\n Current val loss {}".format(epoch,loss_contrastive.item()))
        for m in range(len(tresholds)):
            print("Current val accuracy {} for {}\n".format(acc_v[m]/items,tresholds[m]))
            acc_val[m][epoch] = acc_v[m]/items
        iteration_number_v += 1
        counter_v.append(epoch)
        if(not epoch==0 and loss_v[-1]<loss_contrastive.item()):
            early_stop = early_stop + 1
        else:
            early_stop = 0
        loss_v.append(loss_contrastive.item())
        
        if early_stop>=3:
            break

    plt.Figure()
    plt.plot(counter, loss, label='Loss Trening') 
    plt.plot(counter_v, loss_v, label='Loss Validation') 
    plt.title("Training and validation loss over epochs")
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

    plt.Figure()
    for m in range(len(tresholds)):
        plt.plot(counter, acc[m][0:len(counter)], label=str(tresholds[m]))
    plt.title("Training accuracy over epochs for different tresholds")
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

    plt.Figure()
    for m in range(len(tresholds)):
        plt.plot(counter_v, acc_val[m][0:len(counter_v)], label=str(tresholds[m]))
    plt.title("Validation accuracy over epochs for different tresholds")
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

    plt.Figure()
    plt.bar(tresholds, acc_val[:,len(counter)-1], width=0.2)
    plt.title("Validation accuracy for different tresholds")
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

    outputs = np.concatenate(out3)
    targets = np.concatenate(lab)
    f1 = f1_score(outputs, targets)
    print("F1 score {} for 1".format(f1))

    plt.Figure()
    confusion_matrix = metrics.confusion_matrix(targets, outputs)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
    cm_display.plot()
    plt.show()

    p = np.zeros([5,1])
    r = np.zeros([5,1])

    outputs = np.concatenate(out1) 
    p[0]=precision_score(outputs,targets)
    r[0]=recall_score(outputs, targets)
    outputs = np.concatenate(out2) 
    p[1]=precision_score(outputs,targets)
    r[1]=recall_score(outputs, targets)
    outputs = np.concatenate(out3) 
    p[2]=precision_score(outputs,targets)
    r[2]=recall_score(outputs, targets)
    outputs = np.concatenate(out4) 
    p[3]=precision_score(outputs,targets)
    r[3]=recall_score(outputs, targets)
    outputs = np.concatenate(out5) 
    p[4]=precision_score(outputs,targets)
    r[4]=recall_score(outputs, targets)

    plt.Figure()
    plt.scatter(p, r)
    plt.title("Precision - recall graph for different tresholds]")
    plt.xlabel('Precision')
    plt.xlabel('Recall')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()

    print('Precision:')
    print(p)
    print('Recall:')
    print(r)

    return net

def test(net, tresholds, test_dataloader, device, criterion, best_tr=1):
    net.eval()
    acc_v = np.zeros(len(tresholds))
    items = 0
    batch = 0
    out1, out2, out3, out4, out5, lab = [], [], [], [], [], []
    for batch_id, smpl in enumerate(tqdm(test_dataloader)):
        sleep(0.1)
        img0, img1, label = smpl[0], smpl[1], smpl[2]
        img0, img1, label = img0.to(device), img1.to(device) , label.to(device)
        output1,output2 = net(img0,img1)
        loss_contrastive = criterion(output1,output2,label)
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        for m in range(len(tresholds)):
            label_t = euclidean_distance<tresholds[m]
            acc_v[m] = accuracy_score(label, label_t)*len(label) + acc_v[m]
        items = items + len(label)
        batch = batch + 1
        out1.append(euclidean_distance<0.75), out2.append(euclidean_distance<1.0), out3.append(euclidean_distance<1.25)
        out4.append(euclidean_distance<1.5), out5.append(euclidean_distance<1.75), lab.append(label)
    print("Test loss {}".format(loss_contrastive.item()))
    for m in range(len(tresholds)):
        print("Test accuracy {} for {}\n".format(acc_v[m]/items,tresholds[m]))

    outputs = np.concatenate(out3)
    targets = np.concatenate(lab)
    f1 = f1_score(outputs, targets)
    print("F1 score {} for 1".format(f1))

    confusion_matrix = metrics.confusion_matrix(targets, outputs)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
    cm_display.plot()
    plt.show()

    p = np.zeros([5,1])
    r = np.zeros([5,1])

    outputs = np.concatenate(out1) 
    p[0]=precision_score(outputs,targets)
    r[0]=recall_score(outputs, targets)
    outputs = np.concatenate(out2) 
    p[1]=precision_score(outputs,targets)
    r[1]=recall_score(outputs, targets)
    outputs = np.concatenate(out3) 
    p[2]=precision_score(outputs,targets)
    r[2]=recall_score(outputs, targets)
    outputs = np.concatenate(out4) 
    p[3]=precision_score(outputs,targets)
    r[3]=recall_score(outputs, targets)
    outputs = np.concatenate(out5) 
    p[4]=precision_score(outputs,targets)
    r[4]=recall_score(outputs, targets)

    plt.Figure()
    plt.scatter(p, r)
    plt.title("Precision - recall graph for different tresholds]")
    plt.xlabel('Precision')
    plt.xlabel('Recall')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()

    print('Precision:')
    print(p)
    print('Recall:')
    print(r)


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
            acc_ep += (predicted == torch.reshape(label,[label.size()[0]])).sum().item()
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