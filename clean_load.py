# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 08:55:19 2020

@author: Kunal
"""
import cv2
import os
import pandas as pd

folders = [x[0] for x in os.walk('WithFaces')]
folders.pop(0)

def load_images_from_folder(folder):
    images = []
    metadata = pd.DataFrame([], columns=['Size','dim1','dim2','label'])
    
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))        
        if img is not None:
            temp = pd.DataFrame({'Size':[os.path.getsize(os.path.join(folder,filename))], 'dim1':[img.shape[0]], 'dim2':[img.shape[1]], 'label':[folder.split('\\')[1]]})
            metadata = metadata.append(temp)
            img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
            images.append(img)
    return images, metadata

images = []
metadata = pd.DataFrame()
for folder in folders:
    temp_images, temp_metadata = load_images_from_folder(folder)
    images.extend(temp_images)
    metadata = metadata.append(temp_metadata)
   
import numpy as np
images = np.array(images)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(metadata['label'])
metadata['label'] = le.transform(metadata['label'])

metadata = np.array(metadata, dtype = np.int32)


import h5py
h5f = h5py.File('zip_data.h5', 'w')
h5f.create_dataset('Images', data=images)
h5f.create_dataset('Metadata', data = metadata)
h5f.close()




