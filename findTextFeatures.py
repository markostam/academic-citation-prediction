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
names=os.listdir(folder)
files=list()
classes=list()
print(os.getcwd())
#regex to find class
pattern=re.compile(r"([0-9]+)-")
for f in names:
	cites=pattern.search(f)
	if cites:
		if int(cites.group(1))>10: 
			classes.append(True) 
		else: 
			classes.append(False)
	else: 
		print("WARNING: file name not formatted correctly.")
		print(f)
		answer=input("would you like to ignore this file? y/N")
		if answer=="y":
			pass
		else:
			print("givin up")
			break

	files.append(os.path.join(folder,f))
	print(f)
classes=np.array(classes)
#feature extraction
extract=TfidfVectorizer(input='filename', stop_words='english')
train_data=extract.fit_transform(files)
vocab=extract.get_feature_names()
#train classifier
classify=LinearSVC()
classify.fit(train_data,classes)
attributes=extract.get_feature_names()
#serialize
joblib.dump((classify, vocab, train_data, attributes),os.getcwd()+"/text.pkl", compress=3)