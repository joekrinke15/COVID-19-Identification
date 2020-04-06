from utils import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix

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

predictions = clf.predict_proba(image_data)


#Roc Curve plot
 fpr, tpr, thresholds = roc_curve(labels, predictions[:,1])
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
#Calculate AUC
print("The area under the curve for this model is:{}".format(auc(fpr,tpr)))
      
#Confusion Matrix
predictions = clf.predict(image_data)
confusion_matrix(labels, predictions)