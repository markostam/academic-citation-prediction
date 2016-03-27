
# image-text-bow-clf

python computer vision + textual bag of words classifier. 
used to automatically identify highly cited academic papers by combining visual and image bag of words svm classifiers.

## File list

imgTxtClf.py - extract features and train cv classifier

## Usage
```
python imgTxtClf.py /path/to/text /path/to/images
```

## Troubleshooting

Make sure dataset folders are clear of hidden files, as these will cause the classifier to fail.

If you get 

```python
AttributeError: 'LinearSVC' object has no attribute 'classes_'
```

error, then simply retrain the model. 
