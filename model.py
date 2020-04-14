import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from keras.applications.inception_v3 import InceptionV3
from keras import layers as nn
from keras.models import Model, load_model
from keras import optimizers as optim

import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob

from skimage.io import imread
from skimage.transform import resize

from sklearn.metrics import classification_report
from sklearn.utils import class_weight


# training variables
INPUT_SHAPE = (299, 299, 3)
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
LR_DECAY = LEARNING_RATE/EPOCHS

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
model.compile(loss='categorical_crossentropy',
              optimizer=optim.Adamax(lr=LEARNING_RATE), 
              metrics=['accuracy', 'mse'])

# setup data generators
train_gen = ImageDataGenerator(rescale=1./255,
                                rotation_range=20,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                horizontal_flip=True)

val_gen = ImageDataGenerator(rescale=1./255)

FLOW_PARAMS = {'target_size':INPUT_SHAPE[:2],
        'batch_size':BATCH_SIZE,
        'class_mode':'categorical'}

train_flow = train_gen.flow_from_directory('data/train', **FLOW_PARAMS)
val_flow = val_gen.flow_from_directory('data/val', **FLOW_PARAMS, shuffle=False)
test_flow = val_gen.flow_from_directory('data/test', **FLOW_PARAMS, shuffle=False)

class_weights = class_weight.compute_class_weight('balanced',
                                    np.unique(train_flow.classes),
                                    train_flow.classes)

# callback functions
model_name = 'inception_64'
es_c = EarlyStopping(monitor='val_loss', patience=3, mode='min')
mc_c = ModelCheckpoint(f'serialized/{model_name}.h5',
                       monitor='val_loss',
                       save_best_only=True,
                       mode='min', verbose=1)
tb_c = TensorBoard(log_dir=f'./serialized/logs/{model_name}')


history = model.fit_generator(train_flow,
                            steps_per_epoch=len(train_flow),
                            epochs=EPOCHS,
                            callbacks=[es_c, mc_c, tb_c],
                            verbose=1, 
                            validation_data=val_flow,
                            validation_steps=len(val_flow))

model = load_model(f'serialized/{model_name}.h5')
predictions = model.predict_generator(val_flow, len(val_flow), verbose=1)

y_true = val_flow.classes
print(classification_report(y_true, np.argmax(predictions, 1)))


