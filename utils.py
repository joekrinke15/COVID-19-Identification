"""
Utility to import the images from the pulmonary dataset. 
It imports them and resizes them acoording to the dimensions specified (dim1, dim2) to conserve memory. 
0 corresponds to healthy, 1 corresponds to people with TB, 2 corresponds to viral pneumonia, and 3 corresponds to coronavirus.
"""

from skimage.transform import resize
from skimage import io
import os
import numpy as np
import matplotlib.pyplot as plt


def import_tb(grayscale, dim1=300, dim2=300, path=r'C:/Users/Joe Krinke/Desktop/pulmonary-chest-xray-abnormalities/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png/'):
    os.chdir(path)
    # Create array to hold data
    lung_images = []
    labels = []
    # Read in negative images and resize them. 
    # Create filename, padding the number in the center to match the format.
    # [CHNCXR_0001_0] Last 0 indicates no disease.
    for i in range(326):
        number = str(i+1)
        number = number.rjust(4, '0')
        name = str(r'CHNCXR_') + number + r'_0.png'
        image = io.imread(name, as_gray=grayscale)
        resized = resize(image, (dim1, dim2))  # Resize images to standard size
        lung_images.append(resized)
        labels.append(0)  # Add label indicating they are negative for disease.
    # Positive images go from 327 - 662 inclusive
    for i in range(327, 662):
        number = str(i+1)
        number = number.rjust(4, '0')
        # Create filename, padding the number in the center to match the format.[CHNCXR_0001_1] Last 1 indicates TB.
        name = str(r'CHNCXR_') + number + r'_1.png'
        image = io.imread(name, as_gray=grayscale)
        resized = resize(image, (dim1, dim2))
        lung_images.append(resized)
        labels.append(1)
    # Compile images + labels
    lung_images = np.stack(lung_images)
    return(lung_images, labels)

# The data from the Kaggle source is separated into 3 different sets: 
# normal, covid-19, and viral pneumonia. 
# These functions assume all the data is contained within the same folder [the way they are when you download].


def import_covid(grayscale,dim1=300, dim2=300,path=r'C:\Users\Joe Krinke\Downloads\covid19-radiography-database\COVID-19 Radiography Database\COVID-19'):
    os.chdir(path)
    covid_images = []
    covid_labels = []
    for i in range(219):
        number = str(i+1)
        if i < 133:
            name = str(r'COVID-19 (') + number + r').png'
        else:
            name = str(r'COVID-19(') + number + r').png'
        image = io.imread(name, as_gray=grayscale)
        resized = resize(image, (dim1, dim2))
        covid_images.append(resized)
        covid_labels.append(0)
    covid_images = np.stack(covid_images)
    return(covid_images, covid_labels)


def import_normal(grayscale, dim1=300, dim2=300, path=r'C:\Users\Joe Krinke\Downloads\covid19-radiography-database\COVID-19 Radiography Database\NORMAL'):
    os.chdir(path)
    normal_images = []
    normal_labels = []
    for i in range(1090):
        number = str(i+1)
        name = str(r'NORMAL (') + number + r').png'
        image = io.imread(name, as_gray=grayscale)
        resized = resize(image, (dim1, dim2))
        normal_images.append(resized)
        normal_labels.append(3)
    normal_images = np.stack(normal_images)
    return(normal_images, normal_labels)


def import_pneumonia(grayscale, dim1=300, dim2=300, path=r'C:\Users\Joe Krinke\Downloads\covid19-radiography-database\COVID-19 Radiography Database\Viral Pneumonia'):
    os.chdir(path)
    pneumonia_images = []
    pneumonia_labels = []
    for i in range(1102):
        number = str(i+1)
        name = str(r'Viral Pneumonia (') + number + r').png'
        image = io.imread(name, as_gray=grayscale)
        resized = resize(image, (dim1, dim2))
        pneumonia_images.append(resized)
        pneumonia_labels.append(2)
    pneumonia_images = np.stack(pneumonia_images)
    return(pneumonia_images, pneumonia_labels)

#The following two functions use the tuples returned from the functions that read in the data. 

def create_dataset(tb_data, covid_data, normal_data, pneumonia_data, binary=True):
    image_dataset = np.concatenate((tb_data[0], covid_data[0], normal_data[0], pneumonia_data[0]))
    label_dataset = np.concatenate((tb_data[1], covid_data[1], normal_data[1], pneumonia_data[1]))
    if binary:
        label_dataset = label_conversion(label_dataset)
    return(image_dataset, label_dataset)

# This function turns the coronavirus labels to 1 and all other labels to 0 for binary classification. 

def label_conversion(labels):
    return(np.where(labels == 3, 1,0))
