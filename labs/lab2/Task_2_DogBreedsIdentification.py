#!/usr/bin/env python
# coding: utf-8

# __Complete all sub-tasks marked with ## TO DO! ## and submit the filled notebook on OLAT__ \
# __Using a GPU is recommended here__

# ### Transfer Learning ###
# Aim of this notebook is to implement the concept of transfer learning to train a bigger dataset. We try to compete on a well-known competiton on Kaggle known as Dog Breeds Identification. Read more about it here:
# 
# https://www.kaggle.com/c/dog-breed-identification/overview
# 
# 

# To train a model on the Dog breeds dataset using transfer learning and submit your results to Kaggle.
# Note: Below notebook gives some tips to run the code in pytorch. 

# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[17]:


import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import shutil
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from os.path import isfile, join
import numpy as np


# In[18]:


use_cuda = torch.cuda.is_available()


# In[19]:


from AlexNet import AlexNet
from train_test import start_train_test


# In[13]:


####################################################################################################
## TO DO! : Register on Kaggle With Your repective GroupName  (For example: WS19_VDL_GROUP_01)    ##
####################################################################################################


# In[8]:


####################################################################################################
## TO DO! : Download the Dog-Breeds dataset in folder "data"                                      ##
## from the Kaggle competition link mentioned above                                               ##
####################################################################################################


# In[8]:


####################################################################################################
## TO DO! : Make your dataset to and dataloaders for the  test data                                ##
####################################################################################################

class DogsDataset(Dataset):
    def __init__(self, labels, root_dir, subset=False, transform=None):
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = '{}.jpg'.format(self.labels.iloc[idx, 0])
        fullname = join(self.root_dir, img_name)
        image = Image.open(fullname)
        # labels = self.labels.iloc[idx, 1:].as_matrix().astype('float')
        labels = self.labels.iloc[idx, 1:].as_matrix()
        labels = np.argmax(labels)
        if self.transform:
            image = self.transform(image)
        return [image, labels]


# In[20]:


####################################################################################################
## TO DO! : Split train data into 20% validation set and make dataloaders for train and val split ##
####################################################################################################

INPUT_SIZE = 224
NUM_CLASSES = 120

data_dir = 'data/'
labels = pd.read_csv(join(data_dir, 'labels.csv'))

selected_breed_list = list(labels.groupby('breed').count().sort_values(by='id', ascending=False).head(NUM_CLASSES).index)
labels = labels[labels['breed'].isin(selected_breed_list)]
labels['target'] = 1
labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0)

trainset = labels_pivot.sample(frac=0.8)
testset = labels_pivot[~labels_pivot['id'].isin(trainset['id'])]

print(trainset.shape, testset.shape)

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

my_tr = transforms.Compose([
    transforms.Scale(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

train_set = DogsDataset(trainset, data_dir+'train/', transform=my_tr)
valid_set = DogsDataset(testset, data_dir+'train/', transform=my_tr)

trainloader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)
testloader = DataLoader(valid_set, batch_size=4, shuffle=True, num_workers=4)


# In[11]:


# HINT: 
# One can make their own custom dataset and dataloaders using the CSV file or
# Convert the Dog-breed training dataset into Imagenet Format, where all images of one class are in a
# folder named with class as in the below given format. Standard Pytorch Datasets and Dataloaders can then be applied
# over them
# Root
# |
# |---Class1 ___Img1.png
# |          ___Img2.png
# |
# |---Class2 ___Img3.png
# |          ___Img4.png
# |....
# |....


# __Train famous Alexnet model on Dog breeds dataset. It is not easy to train the alexnet model from 
# scratch on the Dog breeds data itself. Curious minds can try for once to train Alexnet from scratch. We adopt Transfer Learning here. We 
# obtain a pretrained Alexnet model trained on Imagenet and apply transfer learning to it to get better results.__

# ## Transfer Learning

# In[10]:


####################################################################################################
## TO DO! :  Freeze the weigths of the pretrained alexnet model and change the last classification layer
##from 1000 classes of Imagenet to 120 classes of Dog Breeds, only classification layer should be 
## unfreezed and trainable                                                                        ##
####################################################################################################
import torchvision.models as models
pretrained_alexnet = models.alexnet(pretrained=True)
criterion = torch.nn.CrossEntropyLoss()

for param in pretrained_alexnet.parameters()[:-1]:
    param.requires_grad = False

pretrained_alexnet.classifier[6] = nn.Linear(4096,NUM_CLASSES)

# Below function will directly train your network with the given parameters to 5 epochs
# You are also free to use function learned in task 1 to train your model here 
train_loss, test_loss = start_train_test(pretrained_alexnet, trainloader, testloader, criterion)


# ## Making Kaggle Submission

# In[11]:


from transform import transform_testing
import PIL.Image
import torch.nn.functional as F
import numpy as np


# In[14]:


### Not So optimal Code: This can take upto 2 minutes to run: You are free to make an optimal version :) ###
# It iterates over all test images to compute the softmax probablities from the last layer of the network
augment_image = transform_testing()
test_data_root = 'data/test/' 
test_image_list = os.listdir(test_data_root) # list of test files 
result = []
nett = pretrained_alexnet
for img_name in test_image_list:
    img = PIL.Image.open(test_data_root + img_name)
    img_tensor = augment_image(img)
    with torch.no_grad():
        output = nett(img_tensor.unsqueeze_(0).cuda())
        probs = F.softmax(output, dim=1)
    result.append(probs.cpu().numpy())
all_predictions = np.concatenate(result)
print(all_predictions.shape)


# In[15]:


df = pd.DataFrame(all_predictions)
file_list = os.listdir('data/train') # list of classes to be provided here
df.columns = sorted(file_list)

# insert clean ids - without folder prefix and .jpg suffix - of images as first column
test_data_root = 'data/test/' # list of all test files here
test_image_list = os.listdir(test_data_root)
df.insert(0, "id", [e[:-4] for e in test_image_list])
df.to_csv(f"sub_1_alexnet.csv", index=False)


# ### TO DO!: ###
# Submit the created CSV file to Kaggle, with a score(cross entropy loss) not more than __2.0__\
# Take a snapshot of your rank on Kaggle Public Leaderboard and include the image here ...
# For example :
# ![title](snp2.png)

# ## CHALLENGE  (optional)
# Compete against each other, Come up with creative ideas. Try beating the score of __0.3__. The group with minimum score gets a small prize at the time when the solutions are discussed. 
# 

# __Hints:__
# 
# 1. Instead of Alexnet use pretrained resnet 18 model for better accuracy
# 2. Instead of a just adding the last classification layer, try adding two layers to get a better loss
# 3. Train some more layers at the end of the network with a very very small learning rate
# 4. Add Batch Normalizations or Dropout to the layers you have added, (If not present)
# 5. Add more augmentation to your dataset, see tranform.py file and use auto autoaugment to apply more rigorous data augmentation techniques
