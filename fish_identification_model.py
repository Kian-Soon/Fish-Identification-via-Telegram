"""
The goal of this project is to identify 3 different types of fishes (grouper, pomfret, snapper) using the inception pretrained network.
Try with two different variation of transfer function:
1. Vanilla transfer function
2. Transfer function wif fine tuning

Dataset
Training Data: 105 images of grouper, pomfret, snapper in three subfolders
Validation Data: 30 images of grouper, pomfret, snapper in three subfolders
Testing Data: 5 images of grouper, pomfret, snapper in three subfolders

Tasks
1. Load the inception model
2. Chop off the inception's classifier and replace it with my own classifier (3-classes)
3. Perform training, predict and metrics computation
4. Unfreeze last few blocks of the model to enable fine tuning
5. Perform training, prediction and metrics computation
"""

import pandas as pd
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import inception_v3
from keras.models import Sequential, Model, InputLayer
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from sklearn.metrics import classification_report, confusion_matrix

img_height=299
img_width=299
channels=3

model_inception = inception_v3.InceptionV3(input_shape=(img_height, img_height, 3))

'''Download image from drive'''
from keras.preprocessing import image
import numpy as np
img_path = 'fish_image/test/grouper/grouper140.jpg'
img = image.load_img(img_path, target_size=(img_height, img_width))
x = image.img_to_array(img)
print(x.shape)
x = np.expand_dims(x, axis=0)
x = inception_v3.preprocess_input(x)

'''Test the prediction with one test sample'''
preds = model_inception.predict(x)
print('Predicted:', inception_v3.decode_predictions(preds, top=3))

'''Load training set and apply data augmentation to prevent over-fitting'''
train_datagen = ImageDataGenerator(rotation_range=45,
                horizontal_flip=True,
                zoom_range=0.5,
                preprocessing_function=inception_v3.preprocess_input)

'''No. of training samples = 315. Batch size = 15. Then the "ImageDataGenerator" will produce 15 images in each iteration of the training.
An iteration is defined as steps per epoch i.e. the total number of samples / batch_size.'''
X_train_gen = train_datagen.flow_from_directory('fish_image/train',
                                  batch_size=15,
                                  target_size=(img_height, img_width))

'''Load validation set without augmentation
No. of training samples = 315. Batch size = 15. Then the "ImageDataGenerator" will produce 15 images in each iteration of the training. 
An iteration is defined as steps per epoch i.e. the total number of samples / batch_size.'''
val_datagen = ImageDataGenerator(preprocessing_function=inception_v3.preprocess_input)
X_val_gen = val_datagen.flow_from_directory('fish_image/validation',
                                             target_size=(img_height, img_width),
                                             batch_size=15)

'''Vanilla Transfer Function (w/o any training except for the sequential)
Extract Imagenet CNN features. Load the model without the classifier. This will remove the avg_pool (GlobalAveragePooling) and the prediction (dense) layers'''
model_base = inception_v3.InceptionV3(input_shape=(img_height, img_height, 3), include_top=False)

'''Create a new model with the featurizer and a new classifier'''
clf = Sequential()

'''Shape at the top must match the final layer from the model (i.e 8, 8, 2048)'''
clf.add(InputLayer(input_shape=(8, 8, 2048)))
clf.add(Flatten())
clf.add(Dense(128, activation='relu'))
clf.add(Dense(32, activation='relu'))
'''clf.add(Dropout(0.5))'''
clf.add(Dense(3, activation='softmax'))
clf_output = clf(model_base.output)
model = Model(inputs=model_base.input, outputs=clf_output)

'''Get all layers except classifier (the last one)'''
model.layers[:-1]

'''Freeze all layers up to the classifier'''
for layer in model.layers[:-1]:
    layer.trainable = False

'''Compile the model'''
sgd = SGD(lr=1e-3)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['acc'])

'''Setup callback'''
import time
es = EarlyStopping(patience=3)
mc = ModelCheckpoint('transfer.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

'''Fit the model and save the results in modelcheckpoint'''
model.fit(
  X_train_gen,
  steps_per_epoch=21,
  epochs=10,
  validation_data=X_val_gen,
  validation_steps=6,
  callbacks=[mc])

'''Load the best model from drive'''
from keras.models import load_model
'''This file will be different for each run'''
model_path = 'transfer.06-0.13.hdf5'
best_model = load_model(model_path)

'''Metrics computation - for vanilla transfer function (w/o any training except for the sequential layer)'''
test_datagen = ImageDataGenerator(preprocessing_function=inception_v3.preprocess_input)
X_test_gen = test_datagen.flow_from_directory('fish_image/test',
                                                target_size=(img_height, img_width),
                                                batch_size=15)

X_test, y_test = X_test_gen.next()
pred = best_model.predict(X_test)
pred_classes = pred.argmax(axis=1)
pred_classes
y_test_classes = y_test.argmax(axis=1)
y_test_classes

print(classification_report(y_test_classes, pred_classes))
print(confusion_matrix(y_test_classes, pred_classes))

model.save("fish_ident_model.h5")

'''Transfer Function with Fine Tunning'''
for i, layer in enumerate(model.layers):
    print(i, layer.name)

'''Unfreeze the last few layers'''
for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

'''Compile the model'''
sgd = SGD(lr=1e-3)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['acc'])

'''Setup callback (wif a different save path)'''
import time
es = EarlyStopping(patience=3)
mc = ModelCheckpoint('transfer_finetunning.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

'''Fit the model and save the results in modelcheckpoint'''
model.fit(
  X_train_gen,
  steps_per_epoch=21,
  epochs=10,
  validation_data=X_val_gen,
  validation_steps=6,
  callbacks=[mc])

'''Load the best model from drive'''
'''This file will be different for each run'''
model_path = 'transfer_finetunning.02-0.21.hdf5'
best_model = load_model(model_path)

'''Metrics computation - for transfer function with fine tuning'''
pred = best_model.predict(X_test)
pred_classes = pred.argmax(axis=1)
print(classification_report(y_test_classes, pred_classes))
print(confusion_matrix(y_test_classes, pred_classes))