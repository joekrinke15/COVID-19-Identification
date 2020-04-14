import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from keras.applications.vgg19 import VGG19
from keras import layers as nn
from keras.models import Model, load_model
from keras import optimizers as optim

import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob

from skimage.io import imread
from skimage.transform import resize

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight


# training variables
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
LR_DECAY = LEARNING_RATE/EPOCHS
DATA_SRC = 'data/single_src_no_dup' # source folder for the data
# load base model and freeze learning
base_model = VGG19(weights="imagenet", include_top=False,
                         input_tensor=nn.Input(shape=INPUT_SHAPE))
for layer in base_model.layers:
    layer.trainable = False

# add trainable layers
x = base_model.output
x = nn.GlobalAveragePooling2D()(x)
x = nn.Dense(128, activation="relu")(x)
x = nn.Dropout(.5)(x)
output = nn.Dense(3, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(loss='categorical_crossentropy',
              optimizer=optim.Adam(lr=LEARNING_RATE, decay=LR_DECAY), 
              metrics=['accuracy'])

# setup data generators
train_gen = ImageDataGenerator(rescale=1./255,
                                rotation_range=15,
                                horizontal_flip=True)

val_gen = ImageDataGenerator(rescale=1./255)

FLOW_PARAMS = {'target_size':INPUT_SHAPE[:2],
        'batch_size':BATCH_SIZE,
        'class_mode':'categorical'}

train_flow = train_gen.flow_from_directory(f'{DATA_SRC}/train', **FLOW_PARAMS)
val_flow = val_gen.flow_from_directory(f'{DATA_SRC}/val', **FLOW_PARAMS, shuffle=False)
test_flow = val_gen.flow_from_directory(f'{DATA_SRC}/test', **FLOW_PARAMS, shuffle=False)


# callback functions
model_name = 'vgg19_single_src_no_dup'
es_c = EarlyStopping(monitor='val_loss', patience=2, mode='min')
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

# evaluate model
model = load_model(f'serialized/{model_name}.h5')
test_flow.reset()
predictions = model.predict_generator(test_flow, len(test_flow), verbose=1)

y_true = test_flow.classes
print(classification_report(y_true, 
                            np.argmax(predictions, 1), 
                            target_names=test_flow.class_indices))
print(confusion_matrix(y_true, np.argmax(predictions, 1)))
