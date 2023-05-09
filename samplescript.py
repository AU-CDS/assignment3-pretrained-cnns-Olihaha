#importing
import tensorflow as tf
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import json
import numpy
import pandas as pd
import random
import os 
from pathlib import Path

absolute_path = os.path.dirname(os.path.abspath(__file__))


#sample
def make_dataframe_from_json(json_path, num_samples=None):
    # Load JSON data into a list of dictionaries
    data = []
    with open(json_path) as f:
        for line in f:
            data.append(json.loads(line))

    # Randomly sample num_samples if specified
    if num_samples is not None:
        data = random.sample(data, num_samples)

    # Convert list of dictionaries to a dataframe
    return pd.DataFrame(data)

train_df = make_dataframe_from_json("/work/cds-viz/vis-data/indo_fashion/train_data.json", num_samples=20000)
val_df = make_dataframe_from_json("/work/cds-viz/vis-data/indo_fashion/val_data.json", num_samples=7500)
test_df = make_dataframe_from_json("/work/cds-viz/vis-data/indo_fashion/test_data.json", num_samples=7500)

NUMBER_OF_CLASSES = train_df["class_label"].nunique()


#Load VGG16
model = VGG16(weights='imagenet', 
              include_top=False,
              pooling='avg',
              input_shape=(224, 224, 3))

#Freeze the base model layers so only the layers we add on top will be trainable
for layer in model.layers:
    layer.trainable = False

model.summary()


# add new classifier layers
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(128, activation='relu')(flat1)
output = Dense(15, activation='softmax')(class1)

# define new model
model = Model(inputs=model.inputs, 
              outputs=output)
# summarize
model.summary()

#Compile the model. Have to set a learning rate as we are working with the SGD optimizer instead of the ADAM one. 
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)
sgd = SGD(learning_rate=lr_schedule)

model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="image_path",
    y_col="class_label",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="image_path",
    y_col="class_label",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="image_path",
    y_col="class_label",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=val_generator,
    validation_steps=len(val_generator)
)


# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))

# Save the model
model.save('model.h5')


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(epochs, accuracy, 'b', label='Training accuracy')
ax1.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
ax1.set_title('Training and validation accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(epochs, loss, 'b', label='Training loss')
ax2.plot(epochs, val_loss, 'r', label='Validation loss')
ax2.set_title('Training and validation loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()

fig.suptitle('Training and validation metrics', fontsize=16)
plt.savefig('history_plots.png')
plt.show()


#prediction and classifcation report
pred = model.predict(test_generator)
pred = np.argmax(pred,axis=1)

# Map the label
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
pred = [labels[k] for k in pred]


y_test = list(test_df.class_label)
print(classification_report(y_test, pred))

report = classification_report(y_test, pred)
with open('classification_report.txt', 'w') as f:
    f.write(report)
