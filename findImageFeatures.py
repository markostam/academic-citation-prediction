#!/usr/local/bin/python2.7

'''
ENSURE NO HIDDEN FILES IN FOLDERS!
CLASSIFIER SOMETIMES FAILS EVEN WITH
ADDED CODE TO IGNORE HIDDEN FILES
'''

# to run: /classifier/location findFeatures.py -t /dataset

import argparse as ap
import cv2
import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
args = vars(parser.parse_args())

# Get the training classes names and store them in a list
train_path = args["trainingSet"]

#make sure to ignore hidden files
def nodot(item): return item[0] != '.'
training_names = filter(nodot, os.listdir(train_path))


# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
print('getting class names')

image_paths = []
image_classes = []
class_id = 0
for training_name in training_names:
    if not training_name.startswith('.'): #ignore hidden files
        dir = os.path.join(train_path, training_name)
        class_path = imutils.imlist(dir)
        image_paths+=class_path
        image_classes+=[class_id]*len(class_path) 
        class_id+=1

# Create feature extraction and keypoint detector objects
fea_det = cv2.FeatureDetector_create("SIFT")
des_ext = cv2.DescriptorExtractor_create("SIFT")

# Extract features, combine with image storage location
des_list = []
print('extracting features')

for image_path in image_paths:
    if not image_path.startswith('.'):
        im = cv2.imread(image_path, cv2.IMREAD_COLOR)
        print(im.shape)
        kpts = fea_det.detect(im)
        kpts, des = des_ext.compute(im, kpts)
        des_list.append((image_path, des))

print(des_list)

# Stack all the descriptors vertically in a numpy array
print('Stacking descriptors')

descriptors = des_list[0][1]
print((descriptors))
print(descriptors.shape)

for image_path, descriptor in des_list[1:]:
    if not image_path.startswith('.') or descriptor.startswith('.'):
    #descriptor = np.rot90(descriptor)
        descriptors = np.vstack((descriptors, descriptor))

print(descriptors)
print(descriptors.shape)

# build vocabulary with k-means clustering
k = 100
print('Performing clustering K=', k)
voc, variance = kmeans(descriptors, k, 1) #voc = visual vocabulary

# Calculate frequency vector
print('creating frequency vector')
im_features = np.zeros((len(image_paths), k), "float32")
for i in xrange(len(image_paths)):
    if not image_path.startswith('.') or descriptor.startswith('.'):
        words, distance = vq(des_list[i][1],voc)
        for w in words:
            im_features[i][w] += 1

# Perform Tf-Idf vectorization
print('performing TF-IDF')
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Standardization for input ot linear classifier
print('stanardizing input for classification')
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

# Train the Linear SVM
print('training classifier')
clf = LinearSVC()
clf.fit(im_features, np.array(image_classes))

# Save the SVM
joblib.dump((clf, training_names, stdSlr, k, voc), "bof.pkl", compress=3)    
print('saving classifier as "bof.pkl"')

