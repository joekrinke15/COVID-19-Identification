# %%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.preprocessing import label_binarize

from skimage.io import imread

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# common parameters for generators
INPUT_SHAPE = (299, 299, 3)
BATCH_SIZE = 32
FLOW_PARAMS = {'target_size': INPUT_SHAPE[:2],
               'batch_size': BATCH_SIZE,
               'class_mode': 'categorical'}

val_gen = ImageDataGenerator(rescale=1./255)
test_flow = val_gen.flow_from_directory(
    f'./data/super_set/test', **FLOW_PARAMS, shuffle=False)
# %%
model1 = load_model('serialized/inception_trainable_weighted.h5')
model2 = load_model('serialized/xception_trainable_weighted.h5')

test_flow.reset()
pred1 = model1.predict_generator(test_flow, len(test_flow), verbose=1)

test_flow.reset()
pred2 = model2.predict_generator(test_flow, len(test_flow), verbose=1)

predictions = .4*pred1 + .6*pred2
y_true = test_flow.classes

y_pred = np.argmax(predictions, 1)

print(metrics.classification_report(y_true,
                            y_pred,
                            target_names=test_flow.class_indices))
print(metrics.confusion_matrix(y_true, y_pred))

print(metrics.roc_auc_score(y_true, predictions, multi_class='ovo', average='weighted'))


y_test = label_binarize(y_true, [0, 1, 2])
preds  = [pred1, pred2, predictions]


# %%
plt.rcParams.update({'font.size': 16})

labels = ['COVID-19', 'NORMAL', 'OTHER']

fig, ax = plt.subplots(1,3, figsize=(16, 8))

model_names = ['InceptionV3', 'Xception', 'Weighted Voting']
for i in range(3):
        sns.heatmap(metrics.confusion_matrix(y_true, np.argmax(preds[i], 1)),
                        annot=True,
                        ax=ax[i],
                        square=True,
                        linewidths=0.5,
                        cbar=False,
                        fmt='d',
                        cmap='Blues',
                        yticklabels=labels,
                        xticklabels=labels,
                        annot_kws={"size": 20})
        ax[i].set_xlabel('Predicted Labels')
        ax[i].set_title(model_names[i], fontsize=18)
ax[0].set_ylabel('Actual Labels')
fig.suptitle('Confusion Matrix Per Model', fontsize=22)
fig.tight_layout()
plt.subplots_adjust(top=0.99)
# %%

fig, ax = plt.subplots(1,3, figsize=(16, 8))
for m in range(3):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = 3
        y_test = label_binarize(y_true, [0, 1, 2])
        for i in range(n_classes):
                fpr[i], tpr[i], _ = metrics.roc_curve(y_test[:, i], preds[m][:, i])
                roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        alphas = [.9, .6, .6]
        # Plot of a ROC curve for a specific class
        for i in range(n_classes):
                ax[m].plot(fpr[i], 
                        tpr[i],
                        label=f'{labels[i]} (AUC = {roc_auc[i]:.2f})', 
                        alpha=alphas[i])

                ax[m].plot([0, 1], [0, 1], 'k--')
                ax[m].set_xlim([0.0, 1.0])
                ax[m].set_ylim([0.0, 1.05])
                ax[m].set_xlabel('False Positive Rate')
                ax[m].set_ylabel('True Positive Rate')
                ax[m].axis('square')
        ax[m].set_title(model_names[m])
        ax[m].legend(loc="lower right")

fig.suptitle('ROC Curves Per Model By Label', fontsize=22)
fig.tight_layout()
plt.subplots_adjust(top=0.99)
# %%

fig, ax = plt.subplots(3,3)
for i in range(3):
        for j in range(3):
                mask = np.logical_and(y_true == i, y_pred == j)
                ax_args = {'xticklabels': [],
               'yticklabels': [],
               'xticks': [],
               'yticks': []}
                ax[i][j].set(**ax_args)
                if mask.sum() == 0:
                        ax[i][j].axis('square')
                        continue
                impath = np.random.choice(np.array(test_flow.filenames)[mask])
                ax[i][j].imshow(imread(f'./data/super_set/test/{impath}')/255., 
                cmap='Greys')

for i, t in enumerate(labels):
        ax[i][0].set_ylabel(t)
        ax[0][i].set_title(t)
fig.text(.5, 
        .95, 
        'Predicted Labels', 
        ha='center', 
        va='center', fontsize=18)
fig.text(0.06, 
        0.5, 
        'Actual Labels', 
        ha='center', 
        va='center', 
        rotation='vertical', fontsize=18)
fig.tight_layout(pad=.5)
plt.subplots_adjust(top=0.85)
plt.show()