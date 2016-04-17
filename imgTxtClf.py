import time
start = time.time()
import numpy as np
import os
import re
import sys
import csv
import math
import pickle
import itertools
from time import strftime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC #LinearSVC
import matplotlib.pyplot as plt
import cv2
from sklearn.cross_validation import *
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import random
from sklearn.metrics import roc_curve,f1_score,auc
from scipy import interp
import sift_pyocl as sift
from matplotlib.backends.backend_pdf import PdfPages

'''
USAGE: python2 imgTxtClf.py /path/to/text /path/to/images
pulls only files that have both an image and text for sanitization
'''

'''GLOBALS'''
#precision of ROC plots
fpr_space = np.linspace(0, 1, 500)
#image clusters
imgVoc = 100
#test size #SET TO NONE FOR FULL SET
testSize = 2000
#number of folds for cv
nFolds = 2
#granularity of roc curve
fpr_space = np.linspace(0, 1, 500)
''''''

def main(txtPath, imgPath):
    
    #get txt img and class values. also set domain name for file output.
    txt,img,cls = getPathsCitations(imgPath, txtPath)
    #check if we are usin more than one domain
    if imgPath2:
        txt2,img2,cls2 = getPathsCitations(imgPath2, txtPath2)
        txt+=txt2
        img+=img2
        cls+=cls2
        domain = 'all'
    else:
        domain = txtPath[-4:-1]
        

    
    #extract text features
    txtExtract=TfidfVectorizer(input='filename',stop_words='english')
    print 'extracting text features'
    TXT_feat=txtExtract.fit_transform(txt)
    TXT_feat=TXT_feat.toarray()
    print 'done'
    
    #extract image features
    print 'extracting image features'
    IMG_feat = imgFeatExtract(img)
    print 'done'
    
    '''save extracted image and text features for later
    #not worth it about 15gb of data
    try:
        with open("img_txt_feat_n%s_cv%s_%s.pkl" %(testSize, nFolds, domain), 'wb') as handle:
            pickle.dump((IMG_feat, TXT_feat), handle)    
    except:
        print 'error saving the data. possibly too big. look at log.'''
    
    #initiate test report pdf
    pp = PdfPages('img_txt_feat_n%s_cv%s_%s.pdf' %(testSize, nFolds, domain))
    
    '''main k-fold cross validation loop'''
    print '\nPerforming %s fold cross validation' %nFolds
    #define stratified crosss validation scheme
    skf = StratifiedKFold(cls, n_folds=nFolds, shuffle=True)
    
    #initiate clf/stats variables    
    txtF1, txtRocs = [],[]
    imgF1, imgRocs = [],[]
    clsShuffled, namesShuffled= [],[]
    txt_mean_tpr,img_mean_tpr = 0,0

    count = 0
    for train_index, test_index in skf:
        count += 1
        plt.figure()        
        
        #split data into test and train sets for this fold        
        print '\n*******Fold %s********' %count
        #print "TRAIN: %s" %train_index, "\nTEST: %s" %test_index 
        IMG_train_feat, IMG_test_feat = ([IMG_feat[0][i] for i in train_index],IMG_feat[1]), ([IMG_feat[0][i] for i in test_index],IMG_feat[1])
        TXT_train_feat, TXT_test_feat = [TXT_feat[i] for i in train_index], [TXT_feat[i] for i in test_index]
        cls_train, cls_test = [cls[i] for i in train_index], [cls[i] for i in test_index]    
        names_train, names_test = [txt[i] for i in train_index], [txt[i] for i in test_index]    
        
        #keep track of order of filenames and class for metaclassifier
        clsShuffled.append(cls_test)
        namesShuffled.append(names_test)

        '''text classifier'''
        #train classifier and get values
        kernel='linear'
        txtConfs, txtProbas, txtPredict, txtClf, fpr, tpr, thresholds, fMeasure, txt_auc = trainSVM(TXT_train_feat,cls_train,TXT_test_feat,cls_test,count,kernel)        
        #append to overall text values
        txt_mean_tpr += interp(fpr_space, fpr, tpr)
        txt_mean_tpr[0] = 0
        txtRocs.append([fpr, tpr, thresholds])
        txtF1.append(fMeasure)
        #plot text roc
        plotROC(fpr,tpr,txt_auc,'Text fold %d' %count)
        #print and save most informative txt features by coefficient weight
        show_most_informative_features(txtExtract, txtClf, testSize, nFolds, domain, count, n=20)        
        
        '''image classifier'''
        #train classifier and get values
        kernel='rbf'
        imgConfs, imgProbas, imgPredict, imgClf, fpr, tpr, thresholds, fMeasure, img_auc = trainSVM(IMG_train_feat,cls_train,IMG_test_feat,cls_test,count,kernel)        
        #append to overall text values
        img_mean_tpr += interp(fpr_space, fpr, tpr)
        img_mean_tpr[0] = 0
        imgRocs.append([fpr, tpr, thresholds])
        imgF1.append(fMeasure)
        #plot image roc
        plotROC(fpr,tpr,img_auc,'Image fold %d' %count)
        #save and show figure
        pp.savefig()
        plt.show() 
        
    '''calculate results'''
    txt_mean_tpr /= nFolds
    txt_mean_tpr[-1] = 1.0
    txt_mean_auc = auc(fpr_space, txt_mean_tpr)
    
    img_mean_tpr /= nFolds
    img_mean_tpr[-1] = 1.0
    img_mean_auc = auc(fpr_space, img_mean_tpr)
    
    '''save outputs of text and image svm's'''
    try:
        joblib.dump((txtConfs, imgConfs, clsShuffled, namesShuffled, txt_mean_auc, img_mean_auc, imgProbas, txtProbas, txtPredict, imgPredict, imgClf, txtClf), "meta_input_n%s_cv%s_%s.pkl" %(testSize, nFolds, domain), compress=3)    
    except:
        print 'error saving metaclf input.'
    
    '''meta classifier'''
    metaF1, meta_mean_tpr, meta_mean_auc = metaClf(txtConfs, imgConfs, clsShuffled, namesShuffled, txt_mean_auc, img_mean_auc)
    
    '''nonlinear classifier'''
    tiNL_F1, tiNL_tpr, tiNL_auc = nonlinClf(txtProbas, imgProbas, txtConfs, imgConfs, clsShuffled, namesShuffled, txt_mean_auc, img_mean_auc, txtPredict, imgPredict)    
    
    txtF1 = np.asarray(txtF1)
    imgF1 = np.asarray(imgF1)
    metaF1 = np.asarray(metaF1)
    tiNLF1 = np.asarray(tiNL_F1)
    
    print '*******Output*******'
    #f1 score, 95% conf interval and AUC
    
    print '\nText:'
    print "F1: %0.2f (+/- %0.2f)" % (txtF1.mean(), txtF1.std() * 2)
    print 'AUC: %0.2f' %txt_mean_auc
    
    print '\nImages:'
    print "F1: %0.2f (+/- %0.2f)" % (imgF1.mean(), imgF1.std() * 2)
    print 'AUC: %0.2f' %img_mean_auc

    print '\nCombined:'
    print "F1: %0.2f (+/- %0.2f)" % (metaF1.mean(), metaF1.std() * 2)
    print 'AUC: %0.2f' %meta_mean_auc
    
    print '\nNonlinear:'
    print "F1: %0.2f (+/- %0.2f)" % (metaF1.mean(), metaF1.std() * 2)
    print 'AUC: %0.2f' %meta_mean_auc
    
    outtime = strftime("%Y-%m-%d_%H:%M:%S")
    with open('Fscores_n%s_cv%s_%s.csv' %(testSize, nFolds, domain), 'w') as fp:
        a = csv.writer(fp, delimiter=',')
        data = [[outtime, 'F1', 'AUC'],
                ['Text', "%0.2f (+/- %0.2f)" % (txtF1.mean(), txtF1.std() * 2),'%0.2f' %txt_mean_auc],
                ['Images', "%0.2f (+/- %0.2f)" % (imgF1.mean(), imgF1.std() * 2),'%0.2f' %img_mean_auc],
                ['Combined', "%0.2f (+/- %0.2f)" % (metaF1.mean(), metaF1.std() * 2),'%0.2f' %meta_mean_auc],
                ['Nonlinear', "%0.2f (+/- %0.2f)" % (tiNLF1.mean(), tiNLF1.std() * 2),'%0.2f' %tiNL_auc]]
        a.writerows(data)    
    
    plt.figure()
    plotROC(fpr_space,txt_mean_tpr,txt_mean_auc,'Text')
    plotROC(fpr_space,img_mean_tpr,img_mean_auc,'Image')
    #plotROC(fpr_space,tiNL_tpr,tiNL_auc,'Nonlinear')
    plotROC(fpr_space,meta_mean_tpr,meta_mean_auc,'Combined')
    pp.savefig()
    plt.show()
    
    #save TPR's to CSV for plotting ROC elsewhere
    ROCS = csv.writer(open('ROCS_n%s_cv%s_%s.csv' %(testSize, nFolds, domain), "ab"))
    ROCS.writerow([["domain: %s" %domain],["n = %s" %testSize],["nFolds = %s" %domain],["%s" %outtime]])   
    ROCS.writerow(txt_mean_tpr)
    ROCS.writerow(img_mean_tpr)
    ROCS.writerow(meta_mean_tpr)
    ROCS.writerow(tiNL_tpr)

#    txt = csv.writer(open("txt_tpr_%s_%s.csv" %(outtime, domain), "wb"))
#    txt.writerow(txt_mean_tpr)
#    img = csv.writer(open("img_tpr_%s_%s.csv" %(outtime, domain), "wb"))
#    img.writerow(img_mean_tpr)
#    ti = csv.writer(open("ti_tpr_%s_%s.csv" %(outtime, domain), "wb"))
#    ti.writerow(meta_mean_tpr)

    pp.close()
    print 'Function time:', time.time()-start, 'seconds.'
    
#take a list of image file names and transform them into a feature matrix. 
#returns tuple with the matrix first, vocab second.
def imgFeatExtract(image_paths):
    # Create feature extraction and keypoint detector objects
    #surf = cv2.SURF()

    # Extract features, combine with image storage location
    des_list = []
    count = 1
    for image_path in image_paths:
        if ".jpg" in image_path:
            print 'processing image %s: \n%s' %(count, image_path)
            im = cv2.imread(image_path, 1) #read in image
            im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #convert to grayscale
            im = cv2.resize(im, (im.shape[1],300)) #normalize shape
            sift_ocl = sift.SiftPlan(template=im, devicetype='GPU2')
            des = sift_ocl.keypoints(im)
            des = np.asarray([des[i][4] for i in xrange(len(des))])
            des = np.float32(des)
            ###deleted because of memory leak in cv2###
            #_, des = surf.detectAndCompute(im, None)
            des_list.append((image_path, des))
            count+=1

    # Stack all the descriptors vertically in a numpy array
    print 'stacking descriptor features in numpy array'
    count=1    
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        try:        
            if ".jpg" in image_path:
                print 'stacking image %s: \n%s' %(count, image_path)
                descriptors = np.vstack((descriptors, descriptor))
                count+=1
        except:
            print 'error! image %s: wrong size \n%s' %(count, image_path)
            pass
    
    #vocabulary = cluster centroids
    k=imgVoc #number of clusters
    print('performing image feature clustering K=%s' %k)
    voc, variance = kmeans(descriptors, k, 1) #voc = visual vocabulary

    # Calculate frequency vector
    print('creating img frequency vector')
    im_features = np.zeros((len(image_paths), k), "float32")
    for i in xrange(len(image_paths)):
        if ".jpg" in image_path:
            words, distance = vq(des_list[i][1],voc)
            for w in words:
                im_features[i][w] += 1

    # Standardization for input ot linear classifier
    print('standardizing img input for classification')
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)
    
    #save image classifier
    #joblib.dump((clf, training_names, stdSlr, k, voc), "imgclf.pkl", compress=3)    

    return(im_features,voc)

def nonlinClf(txtProbas, imgProbas, txtConfs, imgConfs, clsShuffled, namesShuffled, txt_mean_auc, img_mean_auc, txtPredict, imgPredict):

#    for fold in xrange(len(txtConfs)):
#        txtConfs_wtd = [i*txt_auc_sqrt for i in txtConfs[fold]]
#        imgConfs_wtd = [i*img_auc_sqrt for i in imgConfs[fold]]
    #flatten lists, maintain correct order
    txtConfs = list(itertools.chain(*txtConfs))
    imgConfs = list(itertools.chain(*imgConfs))
    txtPredict = list(itertools.chain(*txtPredict))
    imgPredict = list(itertools.chain(*imgPredict))
    txtProbas = list([max(i) for i in list(itertools.chain(*txtProbas))])
    imgProbas = list([max(i) for i in list(itertools.chain(*imgProbas))])
    clsShuffled = list(itertools.chain(*clsShuffled))
    namesShuffled = list(itertools.chain(*namesShuffled))    
       
    txt_auc_sqrt = math.pow(txt_mean_auc, 1/2)
    img_auc_sqrt = math.pow(img_mean_auc, 1/2)
    
    #txtConfs_wtd = [i*txt_auc_sqrt for i in txtConfs]
    #imgConfs_wtd = [i*img_auc_sqrt for i in imgConfs]
    txtProbas_wtd = [i*txt_auc_sqrt for i in txtProbas]
    imgProbas_wtd = [i*img_auc_sqrt for i in imgProbas]

    #nonlinear selector
    tiNLPredict = [txtPredict[f] if txtProbas_wtd[f]+0.25 >= imgProbas_wtd[f] else imgPredict[f] for f in xrange(len(txtPredict))]  
    tiNLConfs = [txtConfs[f] if txtProbas_wtd[f]+0.25 >= imgProbas_wtd[f] else imgConfs[f] for f in xrange(len(txtPredict))]     
    
    #get confidence and build roc curve
    fpr, tpr, thresholds = roc_curve(clsShuffled,tiNLConfs)
    tiNL_tpr = interp(fpr_space, fpr, tpr)
    tiNL_tpr[0] = 0
    tiNL_tpr[-1] = 1.0
    tiNL_auc = auc(fpr_space, tiNL_tpr)       
    tiNL_F1 = f1_score(clsShuffled, tiNLPredict)
    
    return tiNL_F1, tiNL_tpr, tiNL_auc     

#metaclassifier trained on probability of the txt and img classifier's inputs
def metaClf(txtConfs, imgConfs, clsShuffled, namesShuffled, txt_mean_auc, img_mean_auc):
    
    #flatten lists, maintain correct order
    txtConfs = list(itertools.chain(*txtConfs))
    imgConfs = list(itertools.chain(*imgConfs))
    clsShuffled = list(itertools.chain(*clsShuffled))
    namesShuffled = list(itertools.chain(*namesShuffled))
    #txtProbas = [max(i) for i in list(itertools.chain(*txtProbas))]
    #imgProbas = [max(i) for i in list(itertools.chain(*imgProbas))]

    
    txt_auc_sqrt = math.pow(txt_mean_auc, 1/2)
    img_auc_sqrt = math.pow(img_mean_auc, 1/2)
    
    txtConfs_wtd = [i*txt_auc_sqrt for i in txtConfs]
    imgConfs_wtd = [i*img_auc_sqrt for i in imgConfs]

    #arrange txt and img confidences as features for the meta clf
    #and weight the confidences by the auc
    tiConfs = np.vstack((txtConfs_wtd,imgConfs_wtd))
    tiConfs = np.transpose(tiConfs)

    skf = StratifiedKFold(clsShuffled, n_folds=nFolds, shuffle=True)

    count = 0
    namesReshuffled = []
    metaRocs,metaF1,metaConfs = [],[],[]
    meta_mean_tpr = 0
    
    for train_index, test_index in skf:
        count += 1
        plt.figure()        

        #print some stuff about data split then split it        
        print '\nTraining Meta Classifier'        
        print '*******Fold %s********' %count
        META_train, META_test = [tiConfs[i] for i in train_index], [tiConfs[i] for i in test_index]
        cls_train, cls_test = [clsShuffled[i] for i in train_index], [clsShuffled[i] for i in test_index]
        namesReshuffled.append([namesShuffled[i] for i in test_index]) #keep track of reshuffled filenames

        #train the metaclassifier
        metaClf=SVC(kernel='rbf', probability=False, decision_function_shape='ovr', gamma = 0.01, random_state = random.randint(0,10000))
        metaClf.fit(META_train,cls_train)
        
        #get confidence and build roc curve
        metaPredictions = metaClf.predict(META_test)
        metaConfs.append(metaClf.decision_function(META_test))
        fpr, tpr, thresholds = roc_curve(cls_test,metaConfs[count-1])
        meta_mean_tpr += interp(fpr_space, fpr, tpr)
        meta_mean_tpr[0] = 0
        meta_auc = auc(fpr, tpr)
        plotROC(fpr,tpr,meta_auc,'Meta Clf fold %d' %count)

        fMeasure = f1_score(cls_test, metaPredictions)
        #append to overall list        
        metaRocs.append([fpr, tpr, thresholds])
        metaF1.append(fMeasure)
    
    plt.show()
    meta_mean_tpr /= nFolds
    meta_mean_tpr[-1] = 1.0
    meta_mean_auc = auc(fpr_space, meta_mean_tpr)

    return metaF1, meta_mean_tpr, meta_mean_auc
        
    
#function to plot ROC curve
def plotROC(mean_fpr, mean_tpr, mean_auc, feature_type):
    
    #plt.figure()
    plt.plot(mean_fpr, mean_tpr,label='%s ROC (area = %0.2f)' %(feature_type, mean_auc), lw=2)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))#, label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('%s Receiver Operating Characteristic Plot' %feature_type)
    plt.legend(loc="lower right")
    #plt.show()

#function to print and save most informative text features in classifier
def show_most_informative_features(vectorizer, clf, testSize, nFolds, domain, count, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    with open('infoGain_n%s_cv%s_%s_foldNum%s.csv' %(testSize, nFolds, domain, count), 'w') as fp:   
        for (coef_1, fn_1), (coef_2, fn_2) in top:
            print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (round(coef_1,2), fn_1, round(coef_2,2), fn_2)
            a = csv.writer(fp, delimiter=',')
            row = [[round(coef_1,2), fn_1.encode('ascii', 'ignore'), round(coef_2,2), fn_2.encode('ascii', 'ignore')]]
            a.writerows(row)
            
# get image and text paths and citation counts
def getPathsCitations(imgPath, txtPath):
    names = [os.path.splitext(i)[0] for i in os.listdir(imgPath) if '.jpg' in i]
  
    random.shuffle(names)
    
    ###TEST FOR SMALER SUBSETS###
    if testSize is not None:
        names = [random.choice(names) for i in range(0,testSize)]

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
    
    return txt, img, cls

def trainSVM(train_feat,train_values,test_feat,test_values,cv_fold,kernel):
    
    confs,predicts,probas = [],[],[]
    
    print 'training text classifier'
    clf=SVC(kernel=kernel, probability=True, random_state = random.randint(0,10000))
    clf.fit(train_feat,train_values)
   
    #get confidence and build roc curve
    confs = clf.decision_function(test_feat)
    predicts = clf.predict(test_feat)
    probas = clf.predict_proba(test_feat)
    fpr, tpr, thresholds = roc_curve(test_values,confs)
    AUC = auc(fpr, tpr)
    fMeasure = f1_score(test_values, predicts)

    return confs, probas, predicts, clf, fpr, tpr, thresholds, fMeasure, AUC
    
#imgPath = 
#txtPath = 
##main(txtPath, imgPath) #for running from IDE
#main(sys.argv[1], sys.argv[2]) #for running from command line
##main(sys.argv[1], sys.argv[2]) #for running from command line
main(txtPath, imgPath) #for running from IDE
#