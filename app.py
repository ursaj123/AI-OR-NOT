import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pickle
import re
import albumentations as A
from PIL import Image
from albumentations.pytorch import ToTensorV2

# assigning the title of webpage
st.title("Real Or Fake!!!?")

# first making the model
class fakeorreal(nn.Module):
    def __init__(self):
        super(fakeorreal, self).__init__()
        
        self.conv = nn.Sequential(
            # (-1, 3, 256, 256)
            nn.Conv2d(3, 8, kernel_size=3, padding='same'), 
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2,2),
            
            # (-1, 8, 128, 128)
            nn.Conv2d(8, 16, kernel_size=3, padding='same'), 
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2,2),
            
            # (-1, 16, 64, 64)
            nn.Conv2d(16, 32, kernel_size=3, padding='same'), 
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2),
            
            # (-1, 32, 32, 32)
            nn.Conv2d(32, 64, kernel_size=3, padding='same'), 
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2),
            
            # (-1, 64, 16, 16)
            nn.Conv2d(64, 128, kernel_size=3, padding='same'), 
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,2),
            
            # (-1, 128, 8, 8)
            nn.Conv2d(128, 256, kernel_size=3, padding='same'), 
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2,2)
        )
        self.classifier = nn.Sequential(
            # (-1, 256, 4, 4)
            nn.Flatten(),
            # (-1, 4096)
            nn.Dropout(0.25),
            nn.Linear(4096, 1),
            # (-1, 1)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x
         
    
   

# loading the model
model = fakeorreal()
checkpoint = torch.load('my_checkpoint.pth.tar', map_location='cpu')
model.load_state_dict(checkpoint["state_dict"])   

# now defining the basic UI
image_path = st.file_uploader(label="Upload your image here")
test_transform = A.Compose([
    A.Resize(width=256, height=256),
    A.Normalize(
    mean = [0, 0, 0],
    std = [1, 1, 1], 
    max_pixel_value = 255.0),
    ToTensorV2() 
])
if image_path is not None:
    st.image(image=image_path)
    image = np.array(Image.open(image_path).convert('RGB'))
    augs = test_transform(image=image)
    image = augs['image']
    c, h, w = image.shape
    image = image.reshape(1, c, h, w) # shape is (1, 3, 256, 256)
    op = model(image) # op shape is (1, 1)
    op = torch.sigmoid(op[0])
    prob = int(op[0]*1e4)//1e2
    pred_label = int(op[0]>0.5)
    if pred_label==0:
        prob = 100-prob # changing prob for label 0
    if pred_label==0:
        pred_label = 'ai generated'
    else:
        pred_label = 'real'
    st.write("This image isju {} with a probability of {}%".format(pred_label, prob))
    
    
    
