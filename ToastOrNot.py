import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time
NAME="ToastOrNot{}".format(int(time.time()))
TensorBoard=TensorBoard(log_dir="logs/{}".format(NAME))

DATADIR="C:\\Users\\Sameer\\Desktop\\Toast"
CATEGORIES=["Toast","White"]
IMG_SIZE=100
def train():
	training_data=[]

	for category in CATEGORIES:
		path=os.path.join(DATADIR,category)
		class_num=CATEGORIES.index(category)
		for img in os.listdir(path):
			try:
				img_array = cv2.imread(os.path.join(path, img))
				new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
				training_data.append([new_array,class_num])
			except Exception as e:
				pass

	random.shuffle(training_data)
	X=[]
	y=[]
	for features,label in training_data:
		X.append(features)
		y.append(label)
	X=np.array(X).reshape(-1 , IMG_SIZE, IMG_SIZE,3)

	X=X/255

	model=Sequential()
	model.add(Conv2D(64, (3,3) ,input_shape=X.shape[1:]))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Conv2D(64, (3,3) ))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Flatten())
	
	model.add(Dense(1))
	model.add(Activation("sigmoid"))
	model.compile(loss='binary_crossentropy', optimizer='adam',metrics=["accuracy"])

	model.fit(X,y,batch_size=2, epochs=5,validation_split=.1,callbacks=[TensorBoard])

	model.save('Yeet')

def test(filepath):
	model=tf.keras.models.load_model('Yeet')
	img_array=cv2.imread(filepath)
	new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
	new_array=new_array.reshape(-1,IMG_SIZE,IMG_SIZE,3)
	predict=model.predict(new_array)
	print(CATEGORIES[int(predict[0][0])])
train()     
test('test//White_Test.jpg')