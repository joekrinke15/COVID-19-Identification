import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from keras.applications.vgg19 import VGG19
from keras import layers as nn
from keras.models import Model, load_model
from keras import optimizers as optim
import os
import shutil
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix



# global training variables
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
LR_DECAY = LEARNING_RATE/EPOCHS
DATA_SRC = 'data/gradual_dec' # source folder for the data
# load base model and freeze learning

def make_model(input_shape=INPUT_SHAPE, learning_rate=LEARNING_RATE, lr_decay=LR_DECAY):
    base_model = VGG19(weights="imagenet", include_top=False,
                            input_tensor=nn.Input(shape=input_shape))
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

train_flow = train_gen.flow_from_directory(f'{DATA_SRC}/train', **FLOW_PARAMS)
val_flow = val_gen.flow_from_directory(f'{DATA_SRC}/val', **FLOW_PARAMS, shuffle=False)
test_flow = val_gen.flow_from_directory(f'{DATA_SRC}/test', **FLOW_PARAMS, shuffle=False)


# callback functions
model_name = 'vgg19_single_gradual_dec' # for logs and serialization
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

def move_subset(path_list, src_subdir='train', dst_subdir='moved'):
    print(f'removing {len(path_list)} files from {src_subdir} directory')
    for path in tqdm(path_list):
        dst_dir = f'{DATA_SRC}/{dst_subdir}/'
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)
            pass
        shutil.move(f'{DATA_SRC}/{src_subdir}/{path}', f'{dst_dir}/{path}')

cxr_data = [path for path in np.array(train_flow.filenames).flatten() if 'CXR' in path]

init_epoch = 0
epochs_per_subset = 3
cxr_subsets = np.array_split(cxr_data, epochs_per_subset)

# gradually remove cxr data from the training set
for subset in cxr_subsets:
    # reset generator to reflect change in directory
    train_flow = train_gen.flow_from_directory(f'{DATA_SRC}/train', **FLOW_PARAMS)

    history = model.fit_generator(train_flow,
                                steps_per_epoch=len(train_flow),
                                epochs=init_epoch+epochs_per_subset,
                                initial_epoch=init_epoch,
                                callbacks=[es_c, mc_c, tb_c],
                                verbose=1,
                                validation_data=val_flow,
                                validation_steps=len(val_flow))
    # remove the subset files from the training set
    move_subset(subset)
    # keep track of the current epoch
    init_epoch+=epochs_per_subset

train_flow = train_gen.flow_from_directory(f'{DATA_SRC}/train', **FLOW_PARAMS)
history = model.fit_generator(train_flow,
                                steps_per_epoch=len(train_flow),
                                epochs=init_epoch+epochs_per_subset,
                                initial_epoch=init_epoch,
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
