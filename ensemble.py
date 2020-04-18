import numpy as np
np.random.seed(42)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from keras.applications.inception_v3 import InceptionV3
from keras import layers as nn
from keras.models import Model, load_model
from keras import optimizers as optim
import os
import shutil
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight


# global training variables
INPUT_SHAPE = (299, 299, 3)
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4
LR_DECAY = LEARNING_RATE/EPOCHS
DATA_SRC = 'data/super_set' # source folder for the data
# load base model

def make_model(input_shape=INPUT_SHAPE, learning_rate=LEARNING_RATE, lr_decay=LR_DECAY):
    base_model = InceptionV3(weights="imagenet", include_top=False,
                            input_tensor=nn.Input(shape=input_shape))

    x = base_model.output
    x = nn.GlobalAveragePooling2D()(x)
    x = nn.Dropout(.5)(x)
    output = nn.Dense(3, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(loss='categorical_crossentropy',
                optimizer=optim.Adam(lr=learning_rate, decay=lr_decay), 
                metrics=['accuracy'])
    return model


# setup data generators


# preprocess validation and test data
val_gen = ImageDataGenerator(rescale=1./255)

# common parameters for generators
FLOW_PARAMS = {'target_size':INPUT_SHAPE[:2],
        'batch_size':BATCH_SIZE,
        'class_mode':'categorical'}

test_flow = val_gen.flow_from_directory(f'{DATA_SRC}/test', **FLOW_PARAMS, shuffle=False)


# load models model
model1 = load_model('serialized/inception_trainable_weighted.h5')
model2 = load_model('serialized/xception_trainable_weighted.h5')

test_flow.reset()
pred1 = model1.predict_generator(test_flow, len(test_flow), verbose=1)

test_flow.reset()
pred2 = model2.predict_generator(test_flow, len(test_flow), verbose=1)

predictions = .4*pred1 + .6*pred2
y_true = test_flow.classes
print(classification_report(y_true, 
                            np.argmax(predictions, 1), 
                            target_names=test_flow.class_indices))
print(confusion_matrix(y_true, np.argmax(predictions, 1)))
