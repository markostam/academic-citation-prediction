#!/usr/local/bin/python2

import sys
import numpy as np
import os
import re
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

#to use; python2 buildTextClassifier /path/to/text

folder=sys.argv[1]
files=os.listdir(folder)
classes=list()
#regex to find class
pattern=re.compile(r"([0-9]+)-")
for f in files:
	cites=pattern.search(f)
	if cites:
		if int(cites.group(1))>10: 
			classes.append(True) 
		else: 
			classes.append(False)
	else: 
		print("WARNING: file name not formatted correctly. giving up.")
		break

	f=folder+"/"+f
classes=np.array(classes)
#feature extraction
extract=TfidfVectorizer(input='filename', stop_words='english')
train_data=extract.fit_transform(files)
vocab=dict(zip(extract.get_feature_names(), idf))
#train classifier
classify=LinearSVC()
classify.fit(data,classes)
attributes=extract.get_feature_names()
#serialize
joblib.dump((classify, vocab, train_data, attributes),"text.pkl", compress=3)