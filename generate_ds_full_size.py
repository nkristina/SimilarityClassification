from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision import datasets
import os
from PIL import Image

from generate_dataset import TestingDatasetImageNet, TreningDatasetImageNet, NxNTreningDatasetImageNet


def generate_test(batch_size, use_cuda, num_class, NxN=False):
    # Define main data directory
    DATA_DIR = 'tiny-imagenet-200-test' # Original images come in shapes of [3,64,64]

    # Define training and validation data paths
    TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
    VALID_DIR = os.path.join(DATA_DIR, 'val')

    preprocess_transform_pretrain = T.Compose([
                T.Resize(224), # Resize images to 256 x 256
                #T.CenterCrop(224), # Center crop image
                #T.RandomHorizontalFlip(),
                T.ToTensor(),  # Converting cropped images to tensors
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])

    simple = T.Compose([T.ToTensor(),  # Converting cropped images to tensors
    ])

    if use_cuda:
        kwargs = {"pin_memory": True, "num_workers": 6}
    else:
        kwargs = {}

    val_img_dir = os.path.join(VALID_DIR, 'images')
    folder_dataset = datasets.ImageFolder(root=val_img_dir)

    idx = []
    for i in range(len(folder_dataset)):
            if folder_dataset.targets[i]<num_class:
                name = folder_dataset.imgs[i][0]
                if name.rsplit(os.sep,1)[-1][0]!='.' and Image.open(name).mode=='RGB':
                    idx.append(i)

    siamese_dataset = TestingDatasetImageNet(imageFolderDataset=folder_dataset,idx=idx,transform=preprocess_transform_pretrain)
    vis_dataloader = DataLoader(siamese_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return vis_dataloader


def generate_training(batch_size, use_cuda, num_class, NxN=False, val=False):
    # Define main data directory
    DATA_DIR = 'tiny-imagenet-200-test' # Original images come in shapes of [3,64,64]

    # Define training and validation data paths
    TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
    VALID_DIR = os.path.join(DATA_DIR, 'val')

    preprocess_transform_pretrain = T.Compose([
                T.Resize(224), # Resize images to 256 x 256
                #T.CenterCrop(224), # Center crop image
                #T.RandomHorizontalFlip(),
                T.ToTensor(),  # Converting cropped images to tensors
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])

    simple = T.Compose([T.ToTensor(),  # Converting cropped images to tensors
    ])

    if use_cuda:
        kwargs = {"pin_memory": True, "num_workers": 4}
    else:
        kwargs = {}

    folder_dataset = datasets.ImageFolder(root=TRAIN_DIR)

    idx = []
    for i in range(len(folder_dataset)):
            if folder_dataset.targets[i]<num_class:
                name = folder_dataset.imgs[i][0]
                if name.rsplit(os.sep,1)[-1][0]!='.' and Image.open(name).mode=='RGB':
                    idx.append(i)

    if NxN:
        siamese_dataset = NxNTreningDatasetImageNet(imageFolderDataset=folder_dataset,idx=idx,transform=preprocess_transform_pretrain)
    else:
        siamese_dataset = TreningDatasetImageNet(imageFolderDataset=folder_dataset,idx=idx,transform=preprocess_transform_pretrain, val=val)

    vis_dataloader = DataLoader(siamese_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    return vis_dataloader