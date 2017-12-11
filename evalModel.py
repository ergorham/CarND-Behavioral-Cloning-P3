import argparse
import numpy as np
import tensorflow as tf
import csv
import cv2

parser = argparse.ArgumentParser(description='Trains the model for behavioural learning')
parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
parser.add_argument('data_path',metavar='P', nargs=1,help='The subfolder within ../data/<sub>/IMG to the recorded data')
args = parser.parse_args()
if args.data_path == "":
	print('no path specified')
	exit()
sub_path = args.data_path[0]

lines = []
with open('../data/'+sub_path+'/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
		


images = []
measurements = []

def getImage (inStr, folder):
	source_path = inStr
	filename = source_path.split('\\')[-1]
	current_path = '../data/'+folder+'/IMG/'+filename
	image = cv2.imread(current_path)
	return image
	
def flipAndAppend(image,angle):
	rev_image = cv2.flip(image,0)
	images.append(image)
	images.append(rev_image)
	measurements.append(angle)
	measurements.append(-angle)
	
for line in lines:
	#center
	if(float(line[4])>0.05):
		image = getImage(line[0],sub_path)
		images.append(image)
		measurements.append(float(line[3]))
	
X_test = np.array(images)
y_test = np.array(measurements)

from keras.models import load_model
from keras.models import Model

model = load_model(args.model)
history = model.evaluate(x=X_test,y=y_test)

print(history)
	
exit()