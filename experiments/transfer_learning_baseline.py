import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, History
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Flatten, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 16} ) 
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) 
tf.keras.backend.set_session(sess)

print(tf.__version__)

## User Param
# args = {'dataset':'./data/satellite/amazon/',
#         'model_path':'./experiments/weights/amazon_weights128_vgg16knn_eval00.best.hdf5',
#         'tag':'agri'
#         }

args = {'dataset':'./data/satellite/oilpalm/',
        'model_path':'./experiments/weights/oilpalm_weights128_transfersvm.best.hdf5',
        'tag':'palm'
        }

pickle_file = '/home/yoyo/Desktop/SimCLR/experiments/results/oilpalm_transfersvm_eval00.pkl'



img_resize = (128, 128) # The resize size of each image ex: (64, 64) or None to use the default image size

batch_size = 64

datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            horizontal_flip=True,
            vertical_flip=True) 

train_generator = datagen.flow_from_directory(
        args['dataset'] + 'transfer_predict_svm',
        target_size=(128, 128),
        batch_size=128,
        class_mode='categorical')
        
datagen = ImageDataGenerator(
            rescale=1./255)

steps = int(len(train_generator.filenames) / batch_size)

val_generator = datagen.flow_from_directory(
        args['dataset'] + 'test',
        target_size=(128, 128),
        batch_size=128,
        class_mode='categorical')

args['x_train'] = len(train_generator.filenames)
print(args['x_train'])

acc = []
result_dict = [args]
for i in range(5):
        baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(128, 128, 3)))
        headModel = Flatten(name="flatten")(baseModel.output)
        headModel = Dense(2, activation="softmax")(headModel)

        model = Model(inputs=baseModel.input, outputs=headModel)

        model.summary()

        history = History()
        callbacks = [history, 
                EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=1e-4),
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, cooldown=0, min_lr=1e-7, verbose=1),
                ModelCheckpoint(filepath=args['model_path'], verbose=1, save_best_only=True, 
                                save_weights_only=True, mode='auto')]

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics = ['accuracy'])
        history = model.fit_generator(train_generator, 
                        steps, 
                        epochs=25, 
                        verbose=1, 
                        validation_data=val_generator,
                        validation_steps=5,
                        callbacks=callbacks)


        test_datagen = ImageDataGenerator(rescale=1/255.)

        test_generator = test_datagen.flow_from_directory(
                                args['dataset'] + 'test',
                                classes=None,
                                class_mode=None,
                                shuffle=False,
                                target_size=(128, 128))
        args['x_test'] = len(test_generator.filenames)


        filenames = test_generator.filenames
        nb_samples = len(filenames)
        y_test = np.array([int(f.split('/')[-2] != next(iter(test_generator.class_indices))) for f in filenames])

        preds = model.predict_generator(test_generator)
        predIdxs = preds.argmax(axis=1)

        print("-------------------------------")
        vgg16_dict = classification_report(y_test, predIdxs, target_names=['no_'+args['tag'], args['tag']], digits=3, output_dict=True)
        print(classification_report(y_test, predIdxs, target_names=['no_'+args['tag'], args['tag']], digits=3))
        vgg16_dict['confusion_matrix'] = confusion_matrix(y_test, predIdxs)
        print(vgg16_dict['confusion_matrix'])
        vgg16_dict['history'] = history.history
        result_dict.append(vgg16_dict)

        acc.append(vgg16_dict['accuracy'])

print(f'Accuracy over 5 iter: {np.mean(acc)} +/- {np.std(acc)}')

with open(pickle_file,"wb") as f:
        pickle.dump(result_dict,f)