import numpy as np
import os
import re
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
#usage; python2 getTextClass.py /path/to/textFolder /path/to/classifier
# Load classifier. 
txt_clf, vocab = joblib.load(os.path.join(sys.argv[2],"text.pkl"))
folder=sys.argv[1]
names=os.listdir(folder)
files=list()
classes=list()
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
        print("WARNING: file name not formatted correctly. giving up.")
        break

    files.append(os.path.join(folder,f))
classes=np.array(classes)
#feature extraction
extract=TfidfVectorizer(input='filename', vocabulary=vocab)
data=extract.fit_transform(files)
#classify
predicted_classes=txt_clf.predict(data)
tp=0
fp=0
tn=0
fn=0
for paper in range(0,len(classes)):
    if classes[paper]:
        if predicted_classes[paper]:
            tp+=1
        else:
            fn+=1
    else:
        if predicted_classes[paper]:
            fp+=1
        else:
            tn+=1
precision=tp/(tp+fp)
recall=tp/(tp+fn)
Fmeasure=2*(precision*recall)/(precision+recall)
print('True positives;',tp)
print('False positives;',fp)
print('True negatives;',tn)
print('False negatives;',fn)
print('F-measure;',Fmeasure)
