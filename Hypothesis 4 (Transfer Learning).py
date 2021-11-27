# -*- coding: utf-8 -*-
"""BT4221_Project_Transfer_Learning_(1_3_trainable).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1K1UPaMImra5mP1ba5mtdNNe54sSjF2EQ

# Import Modules and Files
"""

from google.colab import drive
drive.mount('/content/drive')

from google.colab import files

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation, BatchNormalization, SpatialDropout2D, GlobalMaxPooling2D
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall, AUC
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import imagenet_utils, vgg16, vgg19, resnet50, inception_v3
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_curve, balanced_accuracy_score, average_precision_score, roc_auc_score, accuracy_score

!unzip -q '/content/drive/MyDrive/Category and Attribute Prediction Benchmark/Img/img.zip'

# Import files

train_df = pd.read_csv('/content/train_df.csv').astype({'class': str})
val_df = pd.read_csv('/content/val_df.csv').astype({'class': str})
test_df = pd.read_csv('/content/test_df.csv').astype({'class': str})

num_categories = len(train_df['class'].unique())
num_categories

"""# Model Building

## Data Preparation
"""

train_datagen = ImageDataGenerator()

val_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()

batch_size = 64
target_size = (150, 150)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory='/content/',
    x_col="filename",
    y_col="class",
    weight_col=None,
    color_mode='rgb',
    target_size=target_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=4221
)

val_gen = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory='/content/',
    x_col="filename",
    y_col="class",
    weight_col=None,
    color_mode='rgb',
    target_size=target_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=4221
)

test_gen = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory='/content/',
    x_col="filename",
    y_col="class",
    weight_col=None,
    color_mode='rgb',
    target_size=target_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=4221
)

METRICS = [
      CategoricalAccuracy(name='accuracy'),
      Precision(name='precision'),
      Recall(name='recall'),
      AUC(name='auc'),
      AUC(name='prc', curve='PR')
]

from sklearn.metrics import roc_curve, balanced_accuracy_score, average_precision_score, roc_auc_score, accuracy_score

def compute_score(name: str, labels: np.array, pred: np.array) -> None:
    label_1d = np.array(list(map(lambda x: list(x).index(max(x)), labels)))
    pred_1d = np.array(list(map(lambda x: list(x).index(max(x)), pred)))
    print(f"({name}) ROC: {roc_auc_score(labels, pred)}, AUPRC: { average_precision_score(labels, pred)}")
    print(f"({name}) Accuracy: {accuracy_score(label_1d, pred_1d)} Balanced Accuracy: {balanced_accuracy_score(label_1d, pred_1d)}")
    


def one_hot_encode_labels(labels: list) -> np.array:
    class_no = len(np.unique(np.array(labels)))
    output = np.zeros((len(labels), class_no))
    for idx, v in enumerate(labels):
      output[idx][v] = 1.0
    return output

"""## VGG16"""

input_layer = Input(shape=(150, 150, 3))
input_layer = vgg16.preprocess_input(input_layer)

base_model = VGG16(include_top=False, input_tensor=input_layer)
base_model.trainable = False
x = base_model.output
x = GlobalMaxPooling2D()(x)
x = Dropout(rate=0.5)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(rate=0.5)(x)
output = Dense(num_categories, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics=METRICS)
model.summary()

history = model.fit(
    train_gen,
    epochs=5,
    validation_data=val_gen,
    callbacks=[
      ModelCheckpoint('/content/vgg16.h5', monitor='val_acc', verbose=1, save_best_only=True)
    ]
)

fig, axes = plt.subplots(2, 3, figsize=(20, 6))
axes[0][0].plot(history.history['loss'])
axes[0][0].plot(history.history['val_loss'])
axes[0][1].plot(history.history['accuracy'])
axes[0][1].plot(history.history['val_accuracy'])
axes[0][2].plot(history.history['precision'])
axes[0][2].plot(history.history['val_precision'])
axes[1][0].plot(history.history['recall'])
axes[1][0].plot(history.history['val_recall'])
axes[1][1].plot(history.history['auc'])
axes[1][1].plot(history.history['val_auc'])
axes[1][2].plot(history.history['prc'])
axes[1][2].plot(history.history['val_prc'])

"""## VGG16 (all trainable, run after VGG16 base model)"""

# Unfreeze weights
for layer in base_model.layers[-len(base_model.layers:)//3]:
    layer.trainable = True
    
model.compile(optimizer=Adam(1e-5), loss=CategoricalCrossentropy(), metrics=METRICS)
model.summary()

history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    callbacks=[ # This model will be saved as model.h5 and select the best accuracy rate for val_acc. 
      ModelCheckpoint('vgg16_tuned.h5', monitor='val_acc', verbose=1, save_best_only=True)
    ]
)

fig, axes = plt.subplots(2, 3, figsize=(20, 6))
axes[0][0].plot(history.history['loss'])
axes[0][0].plot(history.history['val_loss'])
axes[0][1].plot(history.history['accuracy'])
axes[0][1].plot(history.history['val_accuracy'])
axes[0][2].plot(history.history['precision'])
axes[0][2].plot(history.history['val_precision'])
axes[1][0].plot(history.history['recall'])
axes[1][0].plot(history.history['val_recall'])
axes[1][1].plot(history.history['auc'])
axes[1][1].plot(history.history['val_auc'])
axes[1][2].plot(history.history['prc'])
axes[1][2].plot(history.history['val_prc'])

"""## VGG19"""

input_layer = Input(shape=(150, 150, 3))
input_layer = vgg19.preprocess_input(input_layer)

base_model = VGG19(include_top=False, input_tensor=input_layer)
base_model.trainable = False
x = base_model.output
x = GlobalMaxPooling2D()(x)
x = Dropout(rate=0.5)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(rate=0.5)(x)
output = Dense(num_categories, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics=METRICS)
model.summary()

history = model.fit(
    train_gen,
    epochs=5,
    validation_data=val_gen,
    callbacks=[
      ModelCheckpoint('/content/vgg19.h5', monitor='val_acc', verbose=1, save_best_only=True)
    ]
)

fig, axes = plt.subplots(2, 3, figsize=(20, 6))
axes[0][0].plot(history.history['loss'])
axes[0][0].plot(history.history['val_loss'])
axes[0][1].plot(history.history['accuracy'])
axes[0][1].plot(history.history['val_accuracy'])
axes[0][2].plot(history.history['precision'])
axes[0][2].plot(history.history['val_precision'])
axes[1][0].plot(history.history['recall'])
axes[1][0].plot(history.history['val_recall'])
axes[1][1].plot(history.history['auc'])
axes[1][1].plot(history.history['val_auc'])
axes[1][2].plot(history.history['prc'])
axes[1][2].plot(history.history['val_prc'])

"""## VGG19 (all trainable, run after VGG19 base model)"""

# Unfreeze weights
for layer in base_model.layers[-len(base_model.layers:)//3]:
    layer.trainable = True
    
model.compile(optimizer=Adam(1e-5), loss=CategoricalCrossentropy(), metrics=METRICS)
model.summary()

history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    callbacks=[ # This model will be saved as model.h5 and select the best accuracy rate for val_acc. 
      ModelCheckpoint('/content/vgg19_tuned.h5', monitor='val_acc', verbose=1, save_best_only=True)
    ]
)

fig, axes = plt.subplots(2, 3, figsize=(20, 6))
axes[0][0].plot(history.history['loss'])
axes[0][0].plot(history.history['val_loss'])
axes[0][1].plot(history.history['accuracy'])
axes[0][1].plot(history.history['val_accuracy'])
axes[0][2].plot(history.history['precision'])
axes[0][2].plot(history.history['val_precision'])
axes[1][0].plot(history.history['recall'])
axes[1][0].plot(history.history['val_recall'])
axes[1][1].plot(history.history['auc'])
axes[1][1].plot(history.history['val_auc'])
axes[1][2].plot(history.history['prc'])
axes[1][2].plot(history.history['val_prc'])

model.save('/content/vgg19_tuned.h5')
files.download('/content/vgg19_tuned.h5')

"""## ResNet50"""

input_layer = Input(shape=(150, 150, 3))
input_layer = resnet50.preprocess_input(input_layer)

base_model = ResNet50(include_top=False, input_tensor=input_layer)
base_model.trainable = False
x = base_model.output
x = GlobalMaxPooling2D()(x)
x = Dropout(rate=0.5)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(rate=0.5)(x)
output = Dense(num_categories, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics=METRICS)
model.summary()

history = model.fit(
    train_gen,
    epochs=5,
    validation_data=val_gen,
    callbacks=[
      ModelCheckpoint('/content/resnet50.h5', monitor='val_acc', verbose=1, save_best_only=True)
    ]
)

fig, axes = plt.subplots(2, 3, figsize=(20, 6))
axes[0][0].plot(history.history['loss'])
axes[0][0].plot(history.history['val_loss'])
axes[0][1].plot(history.history['accuracy'])
axes[0][1].plot(history.history['val_accuracy'])
axes[0][2].plot(history.history['precision'])
axes[0][2].plot(history.history['val_precision'])
axes[1][0].plot(history.history['recall'])
axes[1][0].plot(history.history['val_recall'])
axes[1][1].plot(history.history['auc'])
axes[1][1].plot(history.history['val_auc'])
axes[1][2].plot(history.history['prc'])
axes[1][2].plot(history.history['val_prc'])

"""## ResNet50 (all trainable, run after ResNet50 base model)"""

# Unfreeze weights
for layer in base_model.layers[-len(base_model.layers:)//3]:
    layer.trainable = True
    
model.compile(optimizer=Adam(1e-5), loss=CategoricalCrossentropy(), metrics=METRICS)
model.summary()

history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    callbacks=[ # This model will be saved as model.h5 and select the best accuracy rate for val_acc. 
      ModelCheckpoint('resnet50_tuned.h5', monitor='val_acc', verbose=1, save_best_only=True)
    ]
)

fig, axes = plt.subplots(2, 3, figsize=(20, 6))
axes[0][0].plot(history.history['loss'])
axes[0][0].plot(history.history['val_loss'])
axes[0][1].plot(history.history['accuracy'])
axes[0][1].plot(history.history['val_accuracy'])
axes[0][2].plot(history.history['precision'])
axes[0][2].plot(history.history['val_precision'])
axes[1][0].plot(history.history['recall'])
axes[1][0].plot(history.history['val_recall'])
axes[1][1].plot(history.history['auc'])
axes[1][1].plot(history.history['val_auc'])
axes[1][2].plot(history.history['prc'])
axes[1][2].plot(history.history['val_prc'])

"""## InceptionV3"""

input_layer = Input(shape=(150, 150, 3))
input_layer = inception_v3.preprocess_input(input_layer)

base_model = InceptionV3(include_top=False, input_tensor=input_layer)
base_model.trainable = False
x = base_model.output
x = GlobalMaxPooling2D()(x)
x = Dropout(rate=0.5)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(rate=0.5)(x)
output = Dense(num_categories, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics=METRICS)
model.summary()

history = model.fit(
    train_gen,
    epochs=5,
    validation_data=val_gen,
    callbacks=[
      ModelCheckpoint('/content/inceptionv3.h5', monitor='val_acc', verbose=1, save_best_only=True)
    ]
)

fig, axes = plt.subplots(2, 3, figsize=(20, 6))
axes[0][0].plot(history.history['loss'])
axes[0][0].plot(history.history['val_loss'])
axes[0][1].plot(history.history['accuracy'])
axes[0][1].plot(history.history['val_accuracy'])
axes[0][2].plot(history.history['precision'])
axes[0][2].plot(history.history['val_precision'])
axes[1][0].plot(history.history['recall'])
axes[1][0].plot(history.history['val_recall'])
axes[1][1].plot(history.history['auc'])
axes[1][1].plot(history.history['val_auc'])
axes[1][2].plot(history.history['prc'])
axes[1][2].plot(history.history['val_prc'])

"""## InceptionV3 (all trainable, run after InceptionV3 base model)"""

# Unfreeze weights
for layer in base_model.layers[-len(base_model.layers:)//3]:
    layer.trainable = True
    
model.compile(optimizer=Adam(1e-5), loss=CategoricalCrossentropy(), metrics=METRICS)
model.summary()

history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    callbacks=[ # This model will be saved as model.h5 and select the best accuracy rate for val_acc. 
      ModelCheckpoint('inceptionv3_tuned.h5', monitor='val_acc', verbose=1, save_best_only=True)
    ]
)

fig, axes = plt.subplots(2, 3, figsize=(20, 6))
axes[0][0].plot(history.history['loss'])
axes[0][0].plot(history.history['val_loss'])
axes[0][1].plot(history.history['accuracy'])
axes[0][1].plot(history.history['val_accuracy'])
axes[0][2].plot(history.history['precision'])
axes[0][2].plot(history.history['val_precision'])
axes[1][0].plot(history.history['recall'])
axes[1][0].plot(history.history['val_recall'])
axes[1][1].plot(history.history['auc'])
axes[1][1].plot(history.history['val_auc'])
axes[1][2].plot(history.history['prc'])
axes[1][2].plot(history.history['val_prc'])