import numpy as np
import os
import re
import sys
from random import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from matplotlib import pyplot
#for Marko
import cv2
import imutils
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
def main(txtName, imgName):
    #usage; python2 combine.py /path/to/text /path/to/images
    #pulls only files that have both an image and text for sanitization
    imgNames=os.listdir(imgName)
    #txtNames is now the list of shared files(REMOVED FOR TESTING). Divide into ten sublists for cross validation.
    shuffle(imgNames)
    names=list()
    for section in range(0,10):
        current=list()
        names.append(current)
    counter=0
    while imgNames:

        names[counter].append(os.path.splitext(imgNames.pop())[0])
        counter=(counter+1)%10
    #regex to find class, divided into the same sublistings
    pattern=re.compile(r"([0-9]+)-")
    classes=list()
    txtFiles=list()
    imgFiles=list()
    for sublist in names:
        txt=list()
        img=list()
        cls=list()
        txtFiles.append(txt)
        imgFiles.append(img)
        classes.append(cls)
        for f in sublist:
            cites=pattern.search(f)
            if cites:
                if int(cites.group(1))>10: 
                    cls.append(True) 
                else: 
                    cls.append(False)
            else: 
                print(f)
                print("WARNING: file name not formatted correctly. giving up.")
                exit()
            txt.append(os.path.join(txtName,f)+".pdf.txt")
            img.append(os.path.join(imgName,f)+".jpg")
    #True is now good papers, False is bad; we have sublists of complete file paths
    txtRocs=list()
    imgRocs=list()
    tiRocs=list()
    fMeasures=list()
    #i will now be the held out subsection.
    for i in range(10):
        print("constructing fold...")
        curTxtFiles=list()
        curImgFiles=list()
        curClasses=list()
        for j in range(0,9):
            if j!=i:
                curTxtFiles+=txtFiles[j]
                curImgFiles+=imgFiles[j]
                curClasses+=classes[j]
        txtExtract=TfidfVectorizer(input='filename',stop_words='english')
        txtData=txtExtract.fit_transform(curTxtFiles)
        txtClf=LinearSVC()
        txtClf.fit(txtData,curClasses)
        txtTune=txtExtract.fit_transform(txtFiles[i])
        imgClf=LinearSVC()
        markoPrepped=markoPrep(curImgFiles,None)
        imgClf.fit(markoPrepped[0])
        imgTune=markoPrep(imgFiles[i],markoPrepped[1])[0]
        txtConfs=txtClf.decision_function(txtTune)
        txtRocs.append(buildRoc(txtConf,classes[i],100))
        imgConfs=imgClf.decision_function(imgTune)
        imgRocs.append(buildRoc(imgConf,classes[i],100))
        tiConfs=list()
        tp=0
        fp=0
        tn=0
        fn=0
        for i in range(len(0,classes)):
            #combine classifications
            tiConf=(txtConfs[i]+imgConfs[i])/2
            tiConfs.append(tiConf)
            prediction=confidence>=0
            if classes[i]:
                if prediction:
                    tp+=1
                else:
                    fn+=1
            else:
                if prediction:
                    fp+=1
                else:
                    tn+=1
        tiRocs.append(buildRoc(tiConfs,classes[i],100))
        precision=tp/(tp+fp)
        recall=tp/(tp+fn)
        Fmeasure=2*(precision*recall)/(precision+recall)
        fMeasures.append(Fmeasure)
    avTiRoc=avRoc(tiRocs)
    pyplot.plot(avTiRoc[0],avTiRoc[1],color='green', marker='o', linestyle='solid')
    pyplot.title("Average combined ROC")
    pyplot.xlabel("False Positive Rate")
    pyplot.ylabel("True Positive Rate")
    pyplot.show()
#markoPrep is marko's code for taking a list of file names and transforming them into a feature matrix. Returns tuple with the matrix first, vocab second.
def markoPrep(image_paths, inVoc):
    # Create feature extraction and keypoint detector objects
    fea_det = cv2.FeatureDetector_create("SIFT")
    des_ext = cv2.DescriptorExtractor_create("SIFT")

    # Extract features, combine with image storage location
    des_list = []
    print('extracting features')

    for image_path in image_paths:
        if ".jpg" in image_path:
            im = cv2.imread(image_path, cv2.IMREAD_COLOR)
            #print(im.shape)
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
        if ".jpg" in image_path:
        #descriptor = np.rot90(descriptor)
            descriptors = np.vstack((descriptors, descriptor))

    print(descriptors)
    print(descriptors.shape)

    if inVoc is None:#so that we can build vocab or not
        # build vocabulary with k-means clustering
        k = 100
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
    print('performing TF-IDF')
    nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
    idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

    # Standardization for input ot linear classifier
    print('stanardizing input for classification')
    stdSlr = StandardScaler().fit(im_features)
    return((stdSlr.transform(im_features),voc))
#returns a pair of lists, x values and y values.
def buildRoc(confidences, classes, res):
    threshold=max(confidences)
    step=threshold/res
    x=list()
    y=list()
    while threshold>0:
        tp=0
        fp=0
        for i in len(confidences):
            if classes[i]:
                if confidences[i]>threshold:
                    tp+=1
                else:
                    fp+=1
        x.append(fp/(tp+fp))
        y.append(tp/(tp+fp))
        threshold-=step
    pair=(x,y)
    return(pair)
def avRoc(pairs):
    x=list()
    y=list()
    n=len(pairs[0])
    for i in range(n):
        curx=0
        cury=0
        for pair in pairs:
            curx+=pair[i][0]
            cury+=pair[i][1]
        x.append(curx/n)
        y.append(cury/n)
    av=(x,y)
    return(av)

main(sys.argv[1], sys.argv[2])