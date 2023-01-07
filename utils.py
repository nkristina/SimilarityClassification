import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# Functions to display single or a batch of sample images
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.imshow(npimg)
    plt.show()
    
def show_batch(dataloader):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()    
    imshow(make_grid(images)) # Using Torchvision.utils make_grid function
    
def show_image(dataloader):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    random_num = np.random.randint(0, len(images)-1)
    imshow(images[random_num])
    label = labels[random_num]
    print(f'Label: {label}, Shape: {images[random_num].shape}')

def show_image_pair(dataloader):
    dataiter = iter(dataloader)
    images1, images2, labels = dataiter.next()
    random_num = np.random.randint(0, len(images1)-1)
    imshow(images1[random_num])
    imshow(images2[random_num])
    label = labels[random_num]
    print(f'Label: {label}, Shape: {images1[random_num].shape}')