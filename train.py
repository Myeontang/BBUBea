import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

#데이터셋 설정
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test,y_test) = mnist.load_data()
x_train , x_test = x_train/255 ,x_test/255
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

## 데이터 셋 확인
# plt.figure(figsize=(10,10))
# c = 0
# for x in range(5):
#     for y in range(3):
#         plt.subplot(5,3,c+1)
#         plt.imshow(x_train[c],cmap='gray')
#         c+=1

# plt.show()

# print(y_train[:15])

#모델구축
model = tf.keras.models.Sequential([
   tf.keras.layers.Conv2D(input_shape=(28, 28, 1), kernel_size=3, filters=64),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(1,1)),
    tf.keras.layers.Conv2D(input_shape=(28, 28, 1), kernel_size=3, filters=32),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(1,1)),
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax'),
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#모델학습 확인
print(model.fit(x_train, y_train, epochs=5))


#모델학습 후 결과 확인
pred = model.predict(x_train[0:5])
fin = pd.DataFrame(pred).round(2)

print(fin)
print(y_train[0:5])

##모델 h5파일로 저장
model.save('mnist_train.h5')