#!/usr/bin/env python
# coding: utf-8

# In[28]:


import cv2 
import numpy as np
import os
import copy


# In[20]:


def histogramEqualization(img):
    bins = np.zeros((256))
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            n = image[j][i]
            bins[n] = bins[n] + 1
            
    cumm_sum = 0 
    CFD = []
    
    for i in range(len(bins)):
        cumm_sum = cumm_sum + bins[i]
        CFD.append(cumm_sum)

    CFD = CFD/cumm_sum
    
    img2 = np.copy(image)
    
    for i in range(img2.shape[1]):
        for j in range(img2.shape[0]):
            n = img2[j][i]
            img2[j][i] = CFD[n]*255
    return img2


# In[32]:


def ahe(img, tile_size_r, tile_size_c):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed_img = copy.deepcopy(image)
    for r in range(0,img.shape[0], img.shape[0]//tile_size_r):
        for c in range(0,img.shape[1], img.shape[1]//tile_size_c):
            window = img[r:r+img.shape[0]//tile_size_r,c:c+img.shape[1]//tile_size_c]
            processed_img[r:r+img.shape[0]//tile_size_r,c:c+img.shape[1]//tile_size_c] = histogramEqualization(window)
    
    return processed_img


# In[7]:


tile_size_r = 8
tile_size_c = 8


# In[5]:


source_path = 'C:/Users/abhij/OneDrive/Desktop/sem2/perception/project2/adaptive_hist_data/'
destn_path = 'C:/Users/abhij/OneDrive/Desktop/sem2/perception/project2/results/histogram_equalization/'

for image in os.listdir(source_path):
    if (image.endswith(".png")):
        img = cv2.imread(os.path.join(source_path,image))
        processed_img = histogramEqualization(img)
        cv2.imwrite(os.path.join(destn_path, image), processed_img)


# In[7]:


source_path = 'C:/Users/abhij/OneDrive/Desktop/sem2/perception/project2/adaptive_hist_data/'
destn_path = 'C:/Users/abhij/OneDrive/Desktop/sem2/perception/project2/results/ahe/'

for image in os.listdir(source_path):
    if (image.endswith(".png")):
        img = cv2.imread(os.path.join(source_path,image))
        processed_img = ahe(img, tile_size_r, tile_size_c)
        cv2.imwrite(os.path.join(destn_path, image), processed_img)


# In[ ]:




