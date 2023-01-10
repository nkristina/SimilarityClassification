import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import random

from torch.utils.data import Dataset
import torch


def generate_ds(name, num_class=20, DATA_DIR='tiny-imagenet-200-test', ds_size=None, val=False):
    # Define main data directory

    # Define training and validation data paths
    TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
    VALID_DIR = os.path.join(DATA_DIR, 'val')

    # Open and read val annotations text file
    fp = open(os.path.join(DATA_DIR, 'wnids.txt'), 'r')
    data = fp.readlines()

    # Create dictionary to store img filename (word 0) and corresponding
    # label (word 1) for every line in the txt file (as key value pair)
    #classes_enum = {}
    classes = []
    for line in data:
        words = line.split('\n')
        classes.append(words[0])
    fp.close()

    # SECOND STEP
    classes = classes[0:num_class]

    first_img = []
    second_img = []
    label = []

    for i in range(ds_size):
        class_type = np.random.rand()
        if class_type>0.5:
            clas1, clas2 = 1, 1
            while clas1==clas2:
                clas1 = classes[np.random.randint(0,num_class)]
                clas2 = classes[np.random.randint(0,num_class)]
            label.append('0')
        else:
            clas1 = classes[np.random.randint(0,num_class)]
            clas2 = clas1
            label.append('1')

        img_dir1 = os.path.join(TRAIN_DIR, clas1)
        fp1 = open(os.path.join(img_dir1, clas1 + '_boxes.txt'), 'r')
        data1 = fp1.readlines()

        if not val:
            stop = int(len(data1)*0.8)
            start = 0
        else:
            start = int(len(data1)*0.8)+1
            stop = len(data1)
        ok1 = False
        while not ok1:
            choise = data1[np.random.randint(start,stop)].split('\t')[0]
            path  = os.path.join(img_dir1,'images')
            path = os.path.join(path,choise)
            if(os.path.exists(path) and Image.open(path).mode=='RGB'):
                first_img.append(choise)
                ok1 = True
        fp1.close()

        img_dir2 = os.path.join(TRAIN_DIR, clas2)
        fp2 = open(os.path.join(img_dir2, clas2 + '_boxes.txt'), 'r')
        data2 = fp2.readlines()

        if not val:
            stop = int(len(data2)*0.8)
            start = 0
        else:
            start = int(len(data2)*0.8)+1
            stop = len(data2)
        ok2 = False
        while not ok2:
            choise = data2[np.random.randint(start,stop)].split('\t')[0]
            path = os.path.join(img_dir2,'images')
            path = os.path.join(path,choise)
            if(os.path.exists(path) and Image.open(path).mode=='RGB'):
                second_img.append(choise)
                ok2 = True 
        fp2.close()

    new_data = {'First': first_img, 'Second': second_img, 'Label': label}
    df = pd.DataFrame(new_data)
    df.to_csv(name, index=False, header=None)
    
#preprocessing and loading the dataset
class SiameseDataset(Dataset):
    def __init__(self,training_csv=None,training_dir=None,transform=None, val=False):
        # used to prepare the labels and images path
        self.train_df=pd.read_csv(training_csv)
        self.train_dir = training_dir    
        self.transform = transform
        self.val = val

    def __getitem__(self,index):
        # getting the image path
        img1 = self.train_df.iat[index,0]
        class1 = img1.split('_')[0]
        if self.val:
            image1_path=os.path.join(self.train_dir,'images')
            image1_path=os.path.join(image1_path,class1)
            image1_path=os.path.join(image1_path,img1)
        else:
            image1_path=os.path.join(self.train_dir,class1)
            image1_path=os.path.join(image1_path,'images')
            image1_path=os.path.join(image1_path,img1)

        img2 = self.train_df.iat[index,1]
        class2 = img2.split('_')[0]
        if self.val:
            image2_path=os.path.join(self.train_dir,'images')
            image2_path=os.path.join(image2_path, class2)
            image2_path=os.path.join(image2_path,img2)
        else:
            image2_path=os.path.join(self.train_dir,class2)
            image2_path=os.path.join(image2_path,'images')
            image2_path=os.path.join(image2_path,img2)

        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        # img1.transpose((2, 0, 1))
        # img2.transpose((2, 0, 1))
        
        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        label = torch.from_numpy(np.array([int(self.train_df.iat[index,2])],dtype=np.float32))
        smpl = [img0, img1, label]
        return smpl

    def __len__(self):
        return len(self.train_df)


def generate_ds_class(name, num_class=20, ds_size=None):
    # Define main data directory
    DATA_DIR = 'tiny-imagenet-200-test' # Original images come in shapes of [3,64,64]

    # Define training and validation data paths
    TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
    VALID_DIR = os.path.join(DATA_DIR, 'val')

    # Open and read val annotations text file
    fp = open(os.path.join(DATA_DIR, 'wnids.txt'), 'r')
    data = fp.readlines()

    # Create dictionary to store img filename (word 0) and corresponding
    # label (word 1) for every line in the txt file (as key value pair)
    #classes_enum = {}
    classes = []
    for line in data:
        words = line.split('\n')
        classes.append(words[0])
    fp.close()

    # SECOND STEP
    classes = classes[1:num_class+1]

    first_img = []
    second_img = []
    label = []

    for i in range(ds_size):
        id = np.random.randint(0,num_class)
        clas1 = classes[id]
        label.append(str(id))

        img_dir1 = os.path.join(TRAIN_DIR, clas1)
        fp1 = open(os.path.join(img_dir1, clas1 + '_boxes.txt'), 'r')
        data1 = fp1.readlines()

        ok1 = False
        while not ok1:
            choise = data1[np.random.randint(0,len(data1))].split('\t')[0]
            path  = os.path.join(img_dir1,'images')
            path = os.path.join(path,choise)
            if(os.path.exists(path) and Image.open(path).mode=='RGB'):
                first_img.append(choise)
                ok1 = True
        fp1.close()

    new_data = {'First': first_img, 'Label': label}
    df = pd.DataFrame(new_data)
    df.to_csv(name, index=False, header=None)
    
#preprocessing and loading the dataset
class SiameseDataset(Dataset):
    def __init__(self,training_csv=None,training_dir=None,transform=None):
        # used to prepare the labels and images path
        self.train_df=pd.read_csv(training_csv)
        self.train_dir = training_dir    
        self.transform = transform

    def __getitem__(self,index):
        # getting the image path
        img1 = self.train_df.iat[index,0]
        class1 = img1.split('_')[0]
        image1_path=os.path.join(self.train_dir,class1)
        image1_path=os.path.join(image1_path,'images')
        image1_path=os.path.join(image1_path,img1)

        img2 = self.train_df.iat[index,1]
        class2 = img2.split('_')[0]
        image2_path=os.path.join(self.train_dir,class2)
        image2_path=os.path.join(image2_path,'images')
        image2_path=os.path.join(image2_path,img2)

        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        # img1.transpose((2, 0, 1))
        # img2.transpose((2, 0, 1))
        
        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        label = torch.from_numpy(np.array([int(self.train_df.iat[index,2])],dtype=np.float32))
        smpl = [img0, img1, label]
        return smpl

    def __len__(self):
        return len(self.train_df)

class TestingDatasetImageNet(Dataset):

    def __init__(self,imageFolderDataset, idx, transform=None):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.idx = idx
    
    def __getitem__(self,index):
        while True:
            ind = np.random.randint(0,len(self.idx))
            img0_tuple = self.imageFolderDataset.imgs[self.idx[ind]]
            if Image.open(img0_tuple[0]).mode=='RGB':
                break
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                ind = np.random.randint(0,len(self.idx))
                img1_tuple = self.imageFolderDataset.imgs[self.idx[ind]] 
                if img0_tuple[1]==img1_tuple[1] and Image.open(img1_tuple[0]).mode=='RGB':
                    break
        else:
            while True:
                #keep looping till a different class image is found
                ind = np.random.randint(0,len(self.idx))
                img1_tuple = self.imageFolderDataset.imgs[self.idx[ind]] 
                if img0_tuple[1] !=img1_tuple[1] and Image.open(img1_tuple[0]).mode=='RGB':
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]==img0_tuple[1])],dtype=np.float32))

    def __len__(self):
        return len(self.idx)


class TreningDatasetImageNet(Dataset):

    def __init__(self,imageFolderDataset, idx, transform=None, val=False):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        if val:
            self.idx = idx[0::5]
        else:
            del idx[0::5]
            self.idx = idx
        self.val = val

    def __getitem__(self,index):
        while True:
            ind = np.random.randint(0,len(self.idx))
            img0_tuple = self.imageFolderDataset.imgs[self.idx[ind]]
            if Image.open(img0_tuple[0]).mode=='RGB':
                break
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                ind = np.random.randint(0,len(self.idx))
                img1_tuple = self.imageFolderDataset.imgs[self.idx[ind]] 
                if img0_tuple[1]==img1_tuple[1] and Image.open(img1_tuple[0]).mode=='RGB':
                    break
        else:
            while True:
                #keep looping till a different class image is found
                ind = np.random.randint(0,len(self.idx))
                img1_tuple = self.imageFolderDataset.imgs[self.idx[ind]] 
                if img0_tuple[1] !=img1_tuple[1] and Image.open(img1_tuple[0]).mode=='RGB':
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]==img0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.idx)

class NxNTreningDatasetImageNet(Dataset):

    def __init__(self,imageFolderDataset, idx, transform=None):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.idx = idx

    def __getitem__(self,index):
        
        i1 = index//len(self.idx)
        i2 = index % len(self.idx)

        img0_tuple = self.imageFolderDataset.imgs[self.idx[i1]]
        img1_tuple = self.imageFolderDataset.imgs[self.idx[i2]] 

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]==img0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.idx)*len(self.idx)


#preprocessing and loading the dataset
class SDataset(Dataset):
    def __init__(self,training_csv=None,training_dir=None,transform=None):
        # used to prepare the labels and images path
        self.train_df=pd.read_csv(training_csv)
        self.train_dir = training_dir    
        self.transform = transform

    def __getitem__(self,index):
        # getting the image path
        img1 = self.train_df.iat[index,0]
        class1 = img1.split('_')[0]
        image1_path=os.path.join(self.train_dir,class1)
        image1_path=os.path.join(image1_path,'images')
        image1_path=os.path.join(image1_path,img1)

        img2 = self.train_df.iat[index,1]
        class2 = img2.split('_')[0]
        image2_path=os.path.join(self.train_dir,class2)
        image2_path=os.path.join(image2_path,'images')
        image2_path=os.path.join(image2_path,img2)

        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        # img1.transpose((2, 0, 1))
        # img2.transpose((2, 0, 1))
        
        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        label = torch.from_numpy(np.array([int(self.train_df.iat[index,2])],dtype=np.float32))
        smpl = [img0, img1, label]
        return smpl


#preprocessing and loading the dataset
class ClassificationDataset(Dataset):
    def __init__(self,training_csv=None,training_dir=None,transform=None):
        # used to prepare the labels and images path
        self.train_df=pd.read_csv(training_csv)
        self.train_dir = training_dir    
        self.transform = transform

    def __getitem__(self,index):
        # getting the image path
        img1 = self.train_df.iat[index,0]
        class1 = img1.split('_')[0]
        image1_path=os.path.join(self.train_dir,class1)
        image1_path=os.path.join(image1_path,'images')
        image1_path=os.path.join(image1_path,img1)

        # Loading the image
        img0 = Image.open(image1_path)
        # img1.transpose((2, 0, 1))
        # img2.transpose((2, 0, 1))
        
        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
        label = torch.from_numpy(np.array([int(self.train_df.iat[index,1])],dtype=np.float32))
        smpl = [img0, label]
        return smpl

    def __len__(self):
        return len(self.train_df)

if __name__=="__main__":
    generate_ds('train2000.csv',2,2000)
