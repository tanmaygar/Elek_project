from typing import final
import tensorflow
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

# path = r"image_dataset\mask\01001_Mask.jpg"
# img_array = cv2.imread(path)
# plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
# plt.show()

datadirectory = r"image_dataset/"
classes = ["mask", "no_mask"]
# for category in classes:
#     path = os.path.join(datadirectory, category)
#     for img in os.listdir(path):
#         img_array = cv2.imread(os.path.join(path, img))
#         plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
#         break
#     break

train_data = []
for category in classes:
    path = os.path.join(datadirectory, category)
    class_num = classes.index(category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        img_array = cv2.resize(img_array, (224,224))
        train_data.append([img_array, class_num])

X = []
Y = []

random.shuffle(train_data)

for features, label in train_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, 224,224,3)
X = X/256.0
Y = np.array(Y)

model = tf.keras.applications.mobilenet_v2.MobileNetV2()

base_input = model.layers[0].input
base_output = model.layers[-4].output

Flat_layer = layers.Flatten()(base_output)
final_output = layers.Dense(1)(Flat_layer)
final_output = layers.Activation('sigmoid')(final_output)

new_model = keras.Model(inputs = base_input, outputs = final_output)

new_model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
new_model.fit(X,Y, epochs = 1, validation_split = 0.1)

print("[INFO] saving mask detector model...")
new_model.save("mask_detector2.model", save_format="h5")
