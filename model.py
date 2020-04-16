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

# preprocess and augment training data
train_gen = ImageDataGenerator(rescale=1./255,
                                rotation_range=15,
                                horizontal_flip=True)

# preprocess validation and test data
val_gen = ImageDataGenerator(rescale=1./255)

# common parameters for generators
FLOW_PARAMS = {'target_size':INPUT_SHAPE[:2],
        'batch_size':BATCH_SIZE,
        'class_mode':'categorical'}

train_flow = train_gen.flow_from_directory(f'{DATA_SRC}/train', **FLOW_PARAMS, shuffle=True)
val_flow = val_gen.flow_from_directory(f'{DATA_SRC}/val', **FLOW_PARAMS, shuffle=False)
test_flow = val_gen.flow_from_directory(f'{DATA_SRC}/test', **FLOW_PARAMS, shuffle=False)


# callback functions
model_name = 'inception_trainable_weighted' # for logs and serialization
# stop when validation loss does not improve for 2 consecutive epochs
es_c = EarlyStopping(monitor='val_loss', patience=2, mode='min')

# save the model that had the best validation performance so far
mc_c = ModelCheckpoint(f'serialized/{model_name}.h5',
                       monitor='val_loss',
                       save_best_only=True,
                       mode='min', verbose=1)

# setup tensorboard logs
tb_c = TensorBoard(log_dir=f'./serialized/logs/{model_name}')
model = make_model()

weights = class_weight.compute_class_weight('balanced', 
                                            [0,1,2], 
                                            train_flow.classes)
history = model.fit_generator(train_flow,
                                steps_per_epoch=len(train_flow),
                                epochs=EPOCHS,
                                class_weight=weights,
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
