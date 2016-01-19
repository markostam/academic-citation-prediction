import numpy as np
import os
import re
import sys
from random import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from matplotlib import pyplot
def main(txtNames, imgNames):
    #usage; python2 combine.py /path/to/text /path/to/images
    #pulls only files that have both an image and text for sanitization
    removes=list()
    for txt in txtNames:
    	if txt not in imgNames:
    		removes.append(txt)
    for name in removes:
    	txtNames.remove(name)
    #txtNames is now the list of shared files. Divide into ten sublists for cross validation.
    txtNames=shuffle(txtNames)
    names=list()
    for section in range(1,10):
        current=list()
        names.append(current)
    counter=0
    while txtNames:
        names[counter].append(txtNames.pop())
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
        imgfiles.append(img)
        classes.append(cls)
        for f in sublist:
            cites=pattern.search(f)
            if cites:
                if int(cites.group(1))>10: 
                    cls.append(True) 
                else: 
                    cls.append(False)
            else: 
                print("WARNING: file name not formatted correctly. giving up.")
                exit()
            txt.append(os.path.join(argv[1],f))
            img.append(os.path.join(argv[2],f))
    #True is now good papers, False is bad; we have sublists of complete file paths
    txtRocs=list()
    imgRocs=list()
    tiRocs=list()
    fMeasures=list()
    #i will now be the held out subsection.
    for i in range(10):
        print("constructing fold",i,"...")
        curTxtFiles=list()
        curImgFiles=list()
        curClasses=list()
        for j in range(10):
            if j!=i:
                curTxtFiles+=txtFiles[j]
                curImgFiles+=imgFiles[j]
                curClasses+=classes[j]
        txtExtract=TfidfVectorizer(input='filename',stop_words='english')
        txtData=txtExtract.fit_transform(curTxtFiles)
        txtClf=LinearSVC()
        textClf.fit(txtData,curClasses)
        txtTune=txtExtract.fit_transform(txtFiles[i])
        imgClf=LinearSVC()
        imgClf.fit(markoPrep(curImgFiles))
        imgTune=markoPrep(imgFiles[i])
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
#markoPrep is marko's code for taking a list of file names and transforming them into a feature matrix.
def markoPrep(img_files):
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
    return(stdSlr.transform(im_features))
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

main(argv[1], argv[2])