import numpy as np
import os
import re
import sys
import csv
from time import strftime
from random import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
import matplotlib.pyplot as plt
import cv2
import imutils
from sklearn.externals import joblib
from sklearn.cross_validation import *
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
import random
from sklearn.metrics import roc_curve,f1_score,auc
from scipy import interp


#usage; python2 combine.py /path/to/text /path/to/images
#pulls only files that have both an image and text for sanitization


def main(txtPath, imgPath):
    
    nFolds = 10
    names = [os.path.splitext(i)[0] for i in os.listdir(imgPath) if '.jpg' in i]
    shuffle(names)    
    
    kf = KFold(len(names), n_folds=nFolds, shuffle=True)        
    kf = [i for i in kf]
    

    #regex to find class, divided into the same sublistings
    pattern=re.compile(r"([0-9]+)-")
    img,txt,cls = [],[],[]
    cite_high = 10 #divider for high vs low citations

    #split papers into high vs low cited True = high, False = low
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


    '''main k-fold cross validation loop'''
    print 'Performing %s fold cross validation' %nFolds
    txtF1,imgF1,tiF1 = [],[],[]
    txt_mean_tpr,img_mean_tpr,ti_mean_tpr = 0,0,0
    fpr_space = np.linspace(0, 1, 100)
    
    count = 1
    
    for train_index, test_index in kf:
        print '\n*******Fold %s********' %count
        count += 1        
        
        print "TRAIN: %s" %train_index, "\nTEST: %s" %test_index 
        IMG_train, IMG_test = [img[i] for i in train_index], [img[i] for i in test_index]
        TXT_train, TXT_test = [txt[i] for i in train_index], [txt[i] for i in test_index]
        cls_train, cls_test = [cls[i] for i in train_index], [cls[i] for i in test_index]    
    
        '''text classifier'''
        #tfidf extractor
        txtExtract=TfidfVectorizer(input='filename',stop_words='english')
        #extract train and test features
        print 'extracting text features'
        TXT_feat=txtExtract.fit_transform(TXT_train+TXT_test)
        TXT_train_feat=TXT_feat[:len(TXT_train)]
        TXT_test_feat=TXT_feat[-len(TXT_test):]
        #train the classfier
        print 'training text classifier'
        txtClf=SVC(kernel='linear', probability=True, random_state = random.randint(0,10000))
        txtClf.fit(TXT_train_feat,cls_train)
        
        #get confidence and build roc curve
        txtConfs = txtClf.decision_function(TXT_test_feat)
        txtPredictions = txtClf.predict(TXT_test_feat)
        fpr, tpr, thresholds = roc_curve(cls_test,txtConfs)
        txt_mean_tpr += interp(fpr_space, fpr, tpr)
        txt_mean_tpr[0] = 0
        txt_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='Text ROC fold %d (area = %0.2f)' % (count, txt_auc))

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
        img_mean_tpr += interp(fpr_space, fpr, tpr)
        img_mean_tpr[0] = 0
        img_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='Image ROC fold %d (area = %0.2f)' % (count, img_auc))
        
        fMeasure = f1_score(cls_test, imgPredictions)
        #append to overall list        
        imgRocs.append([fpr, tpr, thresholds])
        imgF1.append(fMeasure)
        
        '''combine classifiers'''
#        eclf1 = VotingClassifier(estimators=[('txt', txtClf), ('img', imgClf)],voting='hard')
#        ENSEMBLE input is for possibility of choosing voting classifier in sklearn 
#        ensemble_input = [[i,j] for i,j in zip(txtPredictions,imgPredictions)]
#        eclf1.fit([ensemble_input[i][1] for i in range(0,16)], cls_train)
                
        
        #chooses classifier based on confidence level
        tiPredictions,tiConfs = [],[]
        for j in xrange(len(imgConfs)):
            #chooses confidence of clf furthest from the hyperplane ie most in a class
            tiConf=max(abs(txtConfs[j]),abs(imgConfs[j])) 
            tiConfs.append(tiConf)
            if abs(txtConfs[j]) > abs(imgConfs[j]):
                tiPrediction = txtPredictions[j]
            else:
                tiPrediction = imgPredictions[j]
            tiPredictions.append(tiPrediction)
        
        fpr, tpr, thresholds = roc_curve(cls_test,tiConfs)
        ti_mean_tpr += interp(fpr_space, fpr, tpr)
        ti_mean_tpr[0] = 0
        ti_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='Combined ROC fold %d (area = %0.2f)' % (count, ti_auc))
        fMeasure = f1_score(cls_test, tiPredictions)

        #append to overall list        
        tiRocs.append([fpr, tpr, thresholds])
        tiF1.append(fMeasure)
    
    '''calculate results'''
    txt_mean_F1 = sum(txtF1)/len(txtF1)
    txt_mean_tpr /= nFolds
    txt_mean_tpr[-1] = 1.0
    txt_mean_auc = auc(fpr_space, txt_mean_tpr)
    
    img_mean_F1 = sum(imgF1)/len(imgF1)
    img_mean_tpr /= nFolds
    img_mean_tpr[-1] = 1.0
    img_mean_auc = auc(fpr_space, img_mean_tpr)

    ti_mean_F1 = sum(tiF1)/len(tiF1)
    ti_mean_tpr /= nFolds
    ti_mean_tpr[-1] = 1.0
    ti_mean_auc = auc(fpr_space, img_mean_tpr)
    
    print '*******Output*******'
    
    print '\nText:'
    print 'F1 Score: %s' %txt_mean_F1
    print 'AUC: %s' %txt_mean_auc
    plotROC(fpr_space,txt_mean_tpr,txt_mean_auc,'Text')
    
    print '\nImages:'
    print 'F1 Score: %s' %img_mean_F1
    print 'AUC: %s' %img_mean_auc
    plotROC(fpr_space,img_mean_tpr,img_mean_auc,'Image')

    print '\nCombined:'
    print 'F1 Score: %s' %ti_mean_F1
    print 'AUC: %s' %ti_mean_auc
    plotROC(fpr_space,ti_mean_tpr,ti_mean_auc,'Combined')
    
    #save TPR's to CSV
    time = strftime("%Y-%m-%d_%H:%M:%S")
    txt = csv.writer(open("txt_tpr_%s.csv" %time, "wb"))
    txt.writerow(txt_mean_tpr)
    img = csv.writer(open("img_tpr_%s.csv" %time, "wb"))
    img.writerow(img_mean_tpr)
    ti = csv.writer(open("ti_tpr_%s.csv" %time, "wb"))
    ti.writerow(ti_mean_tpr)


#    avTiRoc=avRoc(tiRocs)
#    plt.plot(fpr,tpr,color='green', marker='o', linestyle='solid')
#    plt.title("Average combined ROC")
#    plt.xlabel("False Positive Rate")
#    plt.ylabel("True Positive Rate")
#    plt.show()
    
#take a list of image file names and transform them into a feature matrix. 
#Returns tuple with the matrix first, vocab second.
def imgFeatExtract(image_paths, inVoc):
    # Create feature extraction and keypoint detector objects
    fea_det = cv2.FeatureDetector_create("SIFT")
    des_ext = cv2.DescriptorExtractor_create("SIFT")

    # Extract features, combine with image storage location
    print 'Extracting img features'
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

    k=100

    if inVoc is None: #so that we can build vocab or not
        # build vocabulary with k-means clustering
        print('Performing img feature clustering K=%s' %k)
        voc, variance = kmeans(descriptors, k, 1) #voc = visual vocabulary
    else: voc=inVoc

    # Calculate frequency vector
    print('Creating img frequency vector')
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
    print('Stanardizing img input for classification')
    stdSlr = StandardScaler().fit(im_features)
    return((stdSlr.transform(im_features),voc))

def plotROC(mean_fpr, mean_tpr, mean_auc, feature_type):
    
    #function to plot ROC curve
    plt.plot(mean_fpr, mean_tpr, 'k--',
    label='Mean ROC (area = %0.2f)' %mean_auc, lw=2)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('%s Receiver operating characteristic plot' %feature_type)
    plt.legend(loc="lower right")
    plt.show()
    
#main(sys.argv[1], sys.argv[2])
main(txtPath, imgPath)