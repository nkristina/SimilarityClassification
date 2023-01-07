import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms as T
from torchvision import models
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    "Contrastive loss function"

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(
            (label) * torch.pow(euclidean_distance, 2)
            + (1-label)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )/2

        return loss_contrastive
    # VEROVATNO GRESKA za batch
    # pairwise distance -> keepdim na false


#create a siamese network
class SiameseNetworkSimple(nn.Module):
    def __init__(self):
        super(SiameseNetworkSimple, self).__init__()
        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(10, 120, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),
        )
        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(20280,1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128,2))
        
    def forward_once(self, x):
        # Forward pass 
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2

class Efficient():
    def __init__(self, embeding_size):
        self.model = models.efficientnet_b0(pretrained=True)
        for params in self.model.parameters():
            params.requires_grad = False
        
        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=embeding_size)

    def Model(self):
        return self.model

class EfficientSiemens(nn.Module):
    def __init__(self, embeding_size):
        super(EfficientSiemens, self).__init__()
        print("INIT")
        self.model = models.efficientnet_b0(pretrained=True)

        for params in self.model.parameters():
            params.requires_grad = False
        
        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=embeding_size)

        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(pytorch_total_params)

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.model(input1)
        # forward pass of input 2
        output2 = self.model(input2)
        return output1, output2