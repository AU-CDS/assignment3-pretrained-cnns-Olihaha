#importing packages
import os
import numpy as np
import cv2 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import random
# cifar10 data 
from tensorflow.keras.datasets import cifar10
# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
# models + optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD
# sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load model
model = VGG16()

# plotting with plt
def plot_history(H, epochs):
    plt.figure(figsize=(12,6))
    plt.style.use("seaborn")

    # plot accuracy
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_accuracy")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_accuracy", linestyle=":")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()

    # plot loss
    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()
    plt.suptitle('Training and validation metrics', fontsize=16)


# Load JSON into dataframe
def make_dataframe_from_json(json_path):
    # Load JSON data into a list of dictionaries
    data = []
    with open(json_path) as f:
        for line in f:
            data.append(json.loads(line))

    # Convert list of dictionaries to a dataframe
    return pd.DataFrame(data)

# Convert json files to df
train_df = make_dataframe_from_json("images/train_data.json")
val_df = make_dataframe_from_json("images/val_data.json")
test_df = make_dataframe_from_json("images/test_data.json")

# Assigning label names
labelnames = val_df['class_label'].unique()

# Define data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
# Data Generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="image_path",
    y_col="class_label",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    seed=42)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="image_path",
    y_col="class_label",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    seed=42)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="image_path",
    y_col="class_label",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False,  
    seed=42)

# Load VGG16 without classification layer, therefore include_top = False
model = VGG16(weights='imagenet', 
              include_top=False,
              pooling='avg',
              input_shape=(32, 32, 3))

tf.keras.backend.clear_session()

# Freeze the base model layers so only the layers we add on top will be trainable
for layer in model.layers:
    layer.trainable = False


# Add new classifier layers and define model
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(128, activation='relu')(flat1)
output = Dense(15, activation='softmax')(class1)
model = Model(inputs=model.inputs, 
              outputs=output)

# Optimization
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01, 
    decay_steps=10000,
    decay_rate=0.9)
sgd = SGD(learning_rate=lr_schedule)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training model
batch_size = 64
H = model.fit(train_generator,
              validation_data=val_generator,
              steps_per_epoch=train_generator.samples // batch_size,
              validation_steps=val_generator.samples // batch_size,
              epochs=10,
              verbose=1)

# Graphs
plot_history(H, 10)
plt.savefig('out/History.png')

# Classification report
predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)
report = classification_report(test_generator.labels,
                               predictions.argmax(axis=1),
                               target_names=labelnames)

# Save the classification report to a file
with open('out/classification_report.txt', 'w') as f:
    f.write(report)

