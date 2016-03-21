#!/usr/local/bin/python2.7

'''
ENSURE NO HIDDEN FILES IN FOLDERS!
CLASSIFIER MAY FAIL IN OSX EVEN WITH
ADDED CODE TO IGNORE HIDDEN FILES
'''

# to run: /classifier/location getClass.py -t /dataset

import argparse as ap
import cv2
import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *

# Load the classifier, class names, scaler, number of clusters and vocabulary
clf, classes_names, stdSlr, k, voc = joblib.load("bof.pkl")

# Get the path of the testing set
parser = ap.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-t", "--testingSet", help="Path to testing Set")
group.add_argument("-i", "--image", help="Path to image")
args = vars(parser.parse_args())

# Get the path of the testing image(s) and store them in a list
image_paths = []
if args["testingSet"]:
    test_path = args["testingSet"]
    try:
        testing_names = os.listdir(test_path)
    except OSError:
        print "No such directory {}\nCheck if the file exists".format(test_path)
        exit()
    for testing_name in testing_names:
        if not testing_name.startswith('.'):
            dir = os.path.join(test_path, testing_name)
            class_path = imutils.imlist(dir)
            image_paths+=class_path
else:
    image_paths = [args["image"]]
    
# Create feature extraction and keypoint detector objects
fea_det = cv2.FeatureDetector_create("SIFT")
des_ext = cv2.DescriptorExtractor_create("SIFT")

# Extract features, combine with image storage location
des_list = []
print('extracting features')

for image_path in image_paths:
    if not image_path.startswith('.'):
        im = cv2.imread(image_path)
        print(im.shape)
        if im == None:
            print "No such file {}\nCheck if the file exists".format(image_path)
            exit()
        kpts = fea_det.detect(im)
        kpts, des = des_ext.compute(im, kpts)
        des_list.append((image_path, des))
    
# Stack all the descriptors vertically in a numpy array
print('Stacking descriptors')

descriptors = des_list[0][1]
for image_path, descriptor in des_list[0:]:
    if not image_path.startswith('.') or descriptor.startswith('.'):
        descriptors = np.vstack((descriptors, descriptor))

# Calculate frequency vector
test_features = np.zeros((len(image_paths), k), "float32")
for i in xrange(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        test_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scale the features
test_features = stdSlr.transform(test_features)

# Perform the predictions
print('Performing prediction')

predictions =  [classes_names[i] for i in clf.predict(test_features)]

#print predictions
for image_path, prediction in zip(image_paths, predictions):
    if not image_path.startswith('.'): #skip hidden files
        print(image_path,prediction,'\n')


