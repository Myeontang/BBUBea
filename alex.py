import tensorflow as tf
import numpy as np
import ssl
import pandas as pd


inputs = tf.keras.Input(shape=(32,32,3))
inputconv = tf.keras.layers.Conv2D(filters=96,kernel_size=(11,11),strides=4,activation=tf.nn.relu)(inputs)
inputpool = tf.keras.layers.MaxPool2D(pool_size=(3,3),padding='SAME')(inputconv)
norm1 = tf.nn.local_response_normalization(inputpool,depth_radius=4,bias=1.0,alpha=0.001/90,beta=0.75)

conv1 = tf.keras.layers.Conv2D(filters=256,kernel_size=(5,5),strides=1,padding='SAME',activation=tf.nn.relu)(inputpool)
pool1 = tf.keras.layers.MaxPool2D(pool_size=(3,3),padding='SAME')(conv1)
norm2 = tf.nn.local_response_normalization(pool1,depth_radius=4,bias=1.0,alpha=0.001/90,beta=0.75)

conv2 = tf.keras.layers.Conv2D(filters=384,kernel_size=(3,3),strides=1,padding='SAME',activation=tf.nn.relu)(pool1)
conv3 = tf.keras.layers.Conv2D(filters=384,kernel_size=(3,3),strides=1,padding='SAME',activation=tf.nn.relu)(conv2)
conv4 = tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),strides=1,padding='SAME',activation=tf.nn.relu)(conv3)
pool2 = tf.keras.layers.MaxPool2D(pool_size=(3,3),padding='SAME')(conv4)
norm3 = tf.nn.local_response_normalization(pool2,depth_radius=4,bias=1.0,alpha=0.001/90,beta=0.75)

##OUTPUT
flat = tf.keras.layers.Flatten()(norm3)
dense1= tf.keras.layers.Dense(units=4096,activation=tf.nn.relu)(flat)
dense2= tf.keras.layers.Dense(units=4096,activation=tf.nn.relu)(dense1)
logits= tf.keras.layers.Dense(units=10,activation = tf.nn.softmax)(dense2)

Alexnet = tf.keras.models.Model(inputs,logits)
Alexnet.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

Alexnet.summary()

ssl._create_default_https_context = ssl._create_unverified_context

##데이터셋 설정
from keras.datasets import cifar10
(x_train, y_train),(x_test,y_test) =cifar10.load_data()
y_train = pd.get_dummies(y_train.reshape(50000))
y_test = pd.get_dummies(y_test.reshape(10000))
print(Alexnet.fit(x_train, y_train, epochs=30,verbose=0))
print(Alexnet.fit(x_train, y_train, epochs=3))
