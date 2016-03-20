import numpy as np
import os
import re
import sys
from random import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from matplotlib import pyplot
#for Marko
import cv2
import imutils
from sklearn.externals import joblib
from sklearn.cross_validation import *
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
import random
from sklearn.metrics import roc_curve,f1_score

#usage; python2 combine.py /path/to/text /path/to/images
#pulls only files that have both an image and text for sanitization


def main(txtPath, imgPath):
    
    #nFolds=5
    #txtPaths is now the list of shared files(REMOVED FOR TESTING). Divide into nFolds sublists for cross validation.

    # [1:] due to hidden files in OSX
    names = [os.path.splitext(i)[0] for i in os.listdir(imgPath)][1:]
    shuffle(names)    
    
    kf = KFold(len(names), n_folds=5, shuffle=True)        
    kf = [i for i in kf]
    

    #regex to find class, divided into the same sublistings
    pattern=re.compile(r"([0-9]+)-")
    img,txt,cls = [],[],[]
    cite_high = 10 #divider for high vs low citations

    
    for f in names:
        
        cites=pattern.search(f)
        if cites:
            if int(cites.group(1)) >= cite_high: 
                cls.append(True) 
            else: 
                cls.append(False)
        else: 
            print(f)
            print("WARNING: file name not formatted correctly. giving up.")
            exit()
        txt.append(os.path.join(txtPath,f)+".pdf.txt")
        img.append(os.path.join(imgPath,f)+".jpg")
    
    #True is good papers, False is bad
    #we have sublists of complete file paths
    txtRocs,imgRocs,tiRocs,txtF1,imgF1,tiF1 = [],[],[],[],[],[]
    
    '''main k-fold cross validation loop'''
    for train_index, test_index in kf:
        print("TRAIN:", train_index, "TEST:", test_index)
        IMG_train, IMG_test = [img[i] for i in train_index], [img[i] for i in test_index]
        TXT_train, TXT_test = [txt[i] for i in train_index], [txt[i] for i in test_index]
        cls_train, cls_test = [cls[i] for i in train_index], [cls[i] for i in test_index]    
    
        '''text classifier'''
        #tfidf extractor
        txtExtract=TfidfVectorizer(input='filename',stop_words='english')
        #extract train and test features
        TXT_feat=txtExtract.fit_transform(TXT_train+TXT_test)
        TXT_train_feat=TXT_feat[:len(TXT_train)]
        TXT_test_feat=TXT_feat[-len(TXT_test):]
        #train the classfier
        txtClf=SVC(kernel='linear', probability=True, random_state = random.randint(0,10000))
        txtClf.fit(TXT_train_feat,cls_train)
        
        #get confidence and build roc curve
        txtConfs = txtClf.decision_function(TXT_test_feat)
        txtPredictions = txtClf.predict(TXT_test_feat)
        fpr, tpr, thresholds = roc_curve(cls_test,txtConfs)
        fMeasure = f1_score(cls_test, txtPredictions)
        #append to overall list        
        txtRocs.append([fpr, tpr, thresholds])
        txtF1.append(fMeasure)
        
        '''image classifier'''
        imgClf=SVC(kernel='linear', probability=True, random_state = random.randint(0,10000))
        #extract features
        IMG_feat=imgFeatExtract(IMG_train+IMG_test,None)
        IMG_train_feat=[IMG_feat[0][:len(IMG_train)],IMG_feat[1]]
        IMG_test_feat=[IMG_feat[0][-len(IMG_test):],IMG_feat[1]]
        imgClf.fit(IMG_train_feat[0],cls_train)

        #get confidence and build roc curve
        imgConfs = imgClf.decision_function(IMG_test_feat[0])
        imgPredictions = imgClf.predict(IMG_test_feat[0])
        fpr, tpr, thresholds = roc_curve(cls_test,imgConfs)
        fMeasure = f1_score(cls_test, imgPredictions)
        #append to overall list        
        imgRocs.append([fpr, tpr, thresholds])
        imgF1.append(fMeasure)
        
        '''combine classifications'''
#        #tiClf = RandomForestClassifier(n_estimators=2)
#        ensemble_input = [[i,j] for i,j in zip(txtPredictions,imgPredictions)]
#        tiClf.fit(ensemble_input, cls_test)
#                
#        
        #chooses classifier based on confidence level
        tiPredictions,tiConfs = [],[]
        for j in xrange(len(imgConfs)):
            tiConf=max(abs(txtConfs[j]),abs(imgConfs[j]))
            tiConfs.append(tiConf)
            if abs(txtConfs[j]) > abs(imgConfs[j]):
                tiPrediction = txtPredictions[j]
            else:
                tiPrediction = imgPredictions[j]
            tiPredictions.append(tiPrediction)
        
        fpr, tpr, thresholds = roc_curve(cls_test,tiConfs)
        fMeasure = f1_score(cls_test, tiPredictions)

        #append to overall list        
        tiRocs.append([fpr, tpr, thresholds])
        tiF1.append(fMeasure)

#    avTiRoc=avRoc(tiRocs)
#    pyplot.plot(fpr,tpr,color='green', marker='o', linestyle='solid')
#    pyplot.title("Average combined ROC")
#    pyplot.xlabel("False Positive Rate")
#    pyplot.ylabel("True Positive Rate")
#    pyplot.show()
    
#take a list of image file names and transform them into a feature matrix. 
#Returns tuple with the matrix first, vocab second.
def imgFeatExtract(image_paths, inVoc):
    # Create feature extraction and keypoint detector objects
    fea_det = cv2.FeatureDetector_create("SIFT")
    des_ext = cv2.DescriptorExtractor_create("SIFT")

    # Extract features, combine with image storage location
    print 'extracting features'
    des_list = []
    for image_path in image_paths:
        if ".jpg" in image_path:
            im = cv2.imread(image_path, cv2.IMREAD_COLOR)
            kpts = fea_det.detect(im)
            kpts, des = des_ext.compute(im, kpts)
            des_list.append((image_path, des))

    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        if ".jpg" in image_path:
        #descriptor = np.rot90(descriptor)
            descriptors = np.vstack((descriptors, descriptor))

    k=10

    if inVoc is None: #so that we can build vocab or not
        # build vocabulary with k-means clustering
        print('Performing clustering K=', k)
        voc, variance = kmeans(descriptors, k, 1) #voc = visual vocabulary
    else: voc=inVoc

    # Calculate frequency vector
    print('creating frequency vector')
    im_features = np.zeros((len(image_paths), k), "float32")
    for i in xrange(len(image_paths)):
        if ".jpg" in image_path:
            words, distance = vq(des_list[i][1],voc)
            for w in words:
                im_features[i][w] += 1

    # Perform Tf-Idf vectorization
#    print('performing TF-IDF')
#    nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
#    idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

    # Standardization for input ot linear classifier
    print('stanardizing input for classification')
    stdSlr = StandardScaler().fit(im_features)
    return((stdSlr.transform(im_features),voc))
    
main(sys.argv[1], sys.argv[2])