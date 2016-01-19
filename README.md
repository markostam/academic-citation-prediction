# classic-paper-prediction
Python implementation of OpenCV bag of words computer vision analysis &amp; Weka bag of words textual analysis combined to identify valuable academic research.

## File list

findFeatures.py - extract features and train cv classifier

getClass.py - extract features and classify unknown images

imutils.py - secondary functions used by findfeatures and getclass 

imutils.pyc - secondary functions used by findfeatures and getclass

classic-paper-prediction - dataset of 300 pixel height academic papers in two domains: data mining + biomedical

## Training the classifier
```
python findFeatures.py -t dataset/train/
```

## Testing the classifier
* Testing a number of images
```
python getClass.py -t dataset/test
```

## Troubleshooting

Make sure dataset folders are clear of hidden files, as these will cause the classifier to fail.

If you get 

```python
AttributeError: 'LinearSVC' object has no attribute 'classes_'
```

error, then simply retrain the model. 
