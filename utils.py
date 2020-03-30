from skimage.transform import resize
from skimage import io
import os 
import numpy as np
import matplotlib.pyplot as plt

#Utility to import the images from the pulmonary dataset. It imports them and resizes them acoording to the dimensions specified (dim1, dim2) to conserve memory. 
def import_pulmonary(path=r'C:/Users/Joe Krinke/Desktop/pulmonary-chest-xray-abnormalities/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png/',grayscale, dim1, dim2):
    #Create array to hold data
    lung_images = []
    labels = []
    #Read in negative images and resize them. Create filename, padding the number in the center to match the format.[CHNCXR_0001_0] Last 0 indicates no disease. 
    for i in range(326):
        number = str(i+1)
        number = number.rjust(4, '0')
        name = str(r'CHNCXR_') + number + r'_0.png' 
        unprocessed = io.imread(name,as_gray = grayscale)
        resized = resize(unprocessed, (dim1,dim2)) #Resize images to standard size
        lung_images.append(resized)
        labels.append(0) #Add label indicating they are negative for disease. 
    #Positive images go from 327 - 662 inclusive
    for i in range(327,662):
        number = str(i+1)
        number = number.rjust(4, '0')
        name = str(r'CHNCXR_') + number + r'_1.png'  #Create filename, padding the number in the center to match the format.[CHNCXR_0001_1] Last 1 indicates disease.  
        unprocessed = io.imread(name, as_gray = grayscale)
        resized = resize(unprocessed, (dim1,dim2))
        lung_images.append(resized)
        labels.append(1) 
    #Compile images + labels
    lung_images = np.stack(lung_images)
    return(lung_images)
