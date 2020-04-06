from utils import *
from sklearn.linear_model import LogisticRegression

tb_data = import_tb(True)
covid_data = import_covid(True)
normal_data = import_normal(True) 
pneumonia_data = import_pneumonia(True)

image_data , labels = create_dataset(tb_data, covid_data, normal_data, pneumonia_data, binary=True)


# Flatten the images.

n_samples = image_data.shape[0]
image_data = image_data.reshape((n_samples, -1))

#Create logistic Regression model.

clf = LogisticRegression(penalty='l1').fit(image_data, labels)
clf.score(image_data, labels)
