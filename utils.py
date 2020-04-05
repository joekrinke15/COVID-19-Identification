from skimage.transform import resize
from skimage import io
import os 
import numpy as np
import matplotlib.pyplot as plt

#Utility to import the images from the pulmonary dataset. It imports them and resizes them acoording to the dimensions specified (dim1, dim2) to conserve memory. 0 corresponds to healthy, 1 corresponds to people with TB, 2 corresponds to viral pneumonia, and 3 corresponds to coronavirus. 
def import_tb(path=r'C:/Users/Joe Krinke/Desktop/pulmonary-chest-xray-abnormalities/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png/',grayscale, dim1, dim2):
    #Create array to hold data
    lung_images = []
    labels = []
    #Read in negative images and resize them. Create filename, padding the number in the center to match the format.[CHNCXR_0001_0] Last 0 indicates no disease. 
    for i in range(326):
        number = str(i+1)
        number = number.rjust(4, '0')
        name = str(r'CHNCXR_') + number + r'_0.png' 
        resized = resize(io.imread(name,as_gray = grayscale), (dim1,dim2)) #Resize images to standard size
        lung_images.append(resized)
        labels.append(0) #Add label indicating they are negative for disease. 
    #Positive images go from 327 - 662 inclusive
    for i in range(327,662):
        number = str(i+1)
        number = number.rjust(4, '0')
        name = str(r'CHNCXR_') + number + r'_1.png'  #Create filename, padding the number in the center to match the format.[CHNCXR_0001_1] Last 1 indicates TB.  
        resized = resize(io.imread(name, as_gray = grayscale), (dim1,dim2))
        lung_images.append(resized)
        labels.append(2)
    #Compile images + labels
    lung_images = np.stack(lung_images)
    return(lung_images)

#The data from the Kaggle source is separated into 3 different sets: normal, covid-19, and viral pneumonia. They are imported separately and then concatenated. These functions assume all the data is contained within the same folder [the way they are when you download]. 
def import_covid(path =r'C:\Users\Joe Krinke\Downloads\covid19-radiography-database.zip\COVID-19 Radiography Database\COVID-19' , dim_1, dim_2, grayscale)
    covid_images = []
    covid_labels = []
    for i in range(220):
        number = str(i)
        name = str(r'COVID-19 (' + number + ')'
        resized =  resize(io.imread(name, as_gray=grayscale), (dim1, dim2))
        covid_images.append(resize)
        covid_labels.append(0)
    covid_images = np.stack(covid_images)
    return(covid_images)
def import_normal(path =, dim_1, dim_2, grayscale):
    normal_images = []
    normal_labels = []
    for i in range(1341):
        number = str(i)
        name = (r'NORMAL ('+ number + ')'
        resized =  resize(io.imread(name, as_gray=grayscale), (dim1, dim2))
        normal_images.append(resized)
        normal_labels.append(3)
    normal_images = np.stack(normal_images)
    return(normal_images)       
def import_pneumonia(path =, dim_1, dim_2, grayscale):
    pneumonia_images = []
    covid_labels = []
    for i in range(1347):
        number = str(i)
        name = (r'Viral Penumonia (') + number +')'
        resized = resize(io.imread(name, as_gray = grayscale, (dim1, dim2)))
        penumonia_images.append(resized)
    pneumonia_images = np.stack(pneumonia_images)
    return(penumonia_images)
                
        
def construct_dataset(path, dim_1, dim_2, grayscale):

    full_data = np.concatenate((import_pneumonia(path, dim_1, dim_2, grayscale), import_covid(path, dim_1, dim_2, grayscale), 
                                import_tb(path, dim_1, dim_2, grayscale),import_normal(path , dim_1, dim_2, grayscale)))
    return(full_data)
