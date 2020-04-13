import numpy as np

from glob import glob
from keras.applications.inception_v3 import InceptionV3
from keras import layers as nn
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers as optim
from keras.utils import to_categorical
import matplotlib.pyplot as plt


# set up global variables

# data paths
DATA_DIR = './Lung Images'
DATA_PATHS = glob(f'{DATA_DIR}/*/*.jpg')

# training variables
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3
LR_DECAY = LEARNING_RATE/EPOCHS

imagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False)


# load base model and freeze learning
base_model = InceptionV3(weights="imagenet", include_top=False,
                         input_tensor=nn.Input(shape=INPUT_SHAPE))
for layer in base_model.layers:
	layer.trainable = False

# add trainable layers
x = base_model.output
x = nn.AveragePooling2D(pool_size=(4, 4))(x)
x = nn.Flatten(name="flatten")(x)
x = nn.Dense(64, activation="relu")(x)
x = nn.Dropout(.5)(x)
output = nn.Dense(3, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.summary()