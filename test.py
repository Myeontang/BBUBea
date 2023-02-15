import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

##업로드 할 이미지 설정
img = cv2.imread('./num.png',cv2.IMREAD_GRAYSCALE)
ret , binary = cv2.threshold(img,170,255,cv2.THRESH_BINARY_INV)
myNum = np.asarray(cv2.resize(binary,dsize=(28,28),interpolation=cv2.INTER_AREA))/255

print(myNum.shape)

test_model = tf.keras.models.load_model('./mnist_train.h5')
myNum = myNum.reshape(-1,28,28,1)

pred = test_model.predict(myNum)
print(np.argmax(pred))