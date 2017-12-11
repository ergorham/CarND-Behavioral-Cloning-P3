import argparse
import numpy as np
import tensorflow as tf
import csv
import cv2

parser = argparse.ArgumentParser(description='Trains the model for behavioural learning')
parser.add_argument('data_path',metavar='P', nargs='+',help='The subfolder within ../data/<sub>/IMG to the recorded data')
args = parser.parse_args()
if args.data_path == "":
	print('no path specified')
	exit()
sub_paths = args.data_path
tf.python.control_flow_ops = tf

lines = []
images = []
measurements = []

bias = +0.0
offset = 0.5

def getImage (inStr, folder):
	source_path = inStr
	filename = source_path.split('\\')[-1]
	current_path = '../data/'+folder+'/IMG/'+filename
	bgr_image = cv2.imread(current_path)
	image = cv2.cvtColor(bgr_image,cv2.COLOR_BGR2RGB)
	return image
	
def flipAndAppend(image,angle):
	rev_image = cv2.flip(image,1)
	images.append(image)
	images.append(rev_image)
	measurements.append(angle)
	measurements.append(angle*-1.0)
	
def append(image, angle):
	images.append(image)
	measurements.append(angle)
	
for sub_path in sub_paths:
	with open('../data/'+sub_path+'/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)
	
	for line in lines:
		#center
		image = getImage(line[0],sub_path)
		#flipAndAppend(image,float(line[3]))
		append(image,float(line[3]))
		#left image; ensure offset max = 1
		if float(line[3])+offset < 1:
			left_meas = float(line[3])+offset 
		else:
			left_meas = 1
		image = getImage(line[1],sub_path)
		append(image,float(left_meas))
		#right image; ensure offset max = -1
		if float(line[3])-offset>-1:
			right_meas = float(line[3])-offset
		else:
			right_meas = -1
		image = getImage(line[2],sub_path)
		append(image,float(right_meas))
		
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.utils.visualize_util import plot

model = Sequential()
model.add(Cropping2D(cropping=((75,10),(0,0)), input_shape=(160,320,3))) #nvidia image dims 66x200; my dims are 60x320
model.add(Lambda(lambda x: x/255.0 -0.5))
model.add(Convolution2D(36,5,5,subsample=(2,2))) #nvidia convo layers 24,36,48,64,64
model.add(Convolution2D(48,5,5,subsample=(2,2))) #I changed to 36, 48, 64, 80, 80
model.add(Convolution2D(64,5,5,subsample=(2,2))) 
model.add(Convolution2D(108,3,3)) 
model.add(Convolution2D(108,3,3))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1)) 

model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True, nb_epoch=3)

plot(model,to_file='nvidiaModel.png',show_shapes='true')
model.save('model.h5')
exit()