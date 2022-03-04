'''
Muhammad Arshad 
01/25/2022
'''

# for classes
from collections import OrderedDict

# for ML model load
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import transforms as ext_transforms
from enet import ENet

import numpy as np
import cv2

# Class mode imports 
import utils
import matplotlib.pyplot as plt
import os, time


'''
- This class loads the ENet model in the memory for infereces 
- Easier to use it like this then directly loading it in the ROS communication class (reusable and extendible) 
- Make sure to set the correct model path. 
'''
class ENetModelPredictions:

    def __init__(self, model_path = './../save/ENet_Rasberry_v2/ENet'):

        self.device = torch.device(0)
        
        curpth = os.path.abspath(os.getcwd())
        abcurpth = os.path.dirname(os.path.abspath(__file__))
        self.model_path = model_path
        
        torch.backends.cudnn.enabled = True
        cudaflag = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cudaflag else 'cpu')
        
        self.class_encoding = OrderedDict([
            ('unlabeled', (0, 0, 0)),
            ('strawberry', (255, 0, 0))
        ])

        self.label_to_rgb = transforms.Compose([
            ext_transforms.LongTensorToRGBPIL(self.class_encoding),
            transforms.ToTensor() 
        ])

        self.model = ENet(2).to(self.device)
        self.model.eval()    
        self.checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(self.checkpoint['state_dict'])

    def predict_live(self, image):

        in_image = image.transpose(2,0,1)
        in_image = torch.from_numpy( in_image ).unsqueeze(0)
        in_image = in_image.to(self.device).float()/255
        out_image = self.model(in_image)

        _, predictions = torch.max(out_image, 1)
        color_predictions = utils.batch_transform(predictions.cpu(), self.label_to_rgb)
        image, mask = self.imshow_batch( in_image.data.cpu(), color_predictions, 1 )


        return image, mask


    def imshow_batch(self, images, labels,num):
        """Displays two grids of images. The top grid displays ``images``
        and the bottom grid ``labels``

        Keyword arguments:
        - images (``Tensor``): a 4D mini-batch tensor of shape
        (B, C, H, W)
        - labels (``Tensor``): a 4D mini-batch tensor of shape
        (B, C, H, W)

        """

        # Make a grid with the images and labels and convert it to numpy
        images = torchvision.utils.make_grid(images).numpy()
        labels = torchvision.utils.make_grid(labels).numpy()

        img_rgb = np.transpose(images, (1, 2, 0))
        img_inst = np.transpose(labels, (1, 2, 0))
        
        return img_rgb, img_inst        


if __name__ == '__main__':

    try: 
        eNet = ENetModelPredictions()
        print ("Loaded ENet ML model in memory. Please use live_prediction function to get predictions")
    except: 
        print ("Unable to load ENet ML model in memory")