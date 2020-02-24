# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import confusion_matrix
import re

train = pd.read_csv("../input/nlp-getting-started/train.csv")
test = pd.read_csv("../input/nlp-getting-started/test.csv")
#print (train.info()) 
#print (train.target.value_counts()) #train set has approx 60-40 distribution (0, 1)
#print (train.head(3))
#print (train.keyword.value_counts())

# Keywords #%20 is a spacing
keywords_train = train[train.keyword.notna()]
keywords_train.loc["keyword"] = keywords_train.keyword.apply(lambda x: re.sub("%20", " ", x))

t = keywords_train.groupby("keyword").mean().target
#print (t[t <= 0.1])
#print (t[t >= 0.9])

# Using sklearn package
# CountVectorizer counts the words in each tweet
count_vectorizer = feature_extraction.text.CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train["text"])
# Apply the same transformations to test set
test_vectors = count_vectorizer.transform(test["text"])

# Vector is sparse: .todense() keeps only non-zero elements to save space
#print(train_vectors[0].todense().shape)
#print(train_vectors[0].todense())

# Ridge regression differentiates between more important features and 
# with R2 regularisation, prevents overfitting
clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, train["target"], cv = 3, scoring = "f1")

#scores
clf.fit(train_vectors, train["target"])

y_test = train.target
y_pred = clf.predict(train_vectors)

#y_pred[train.keyword.apply(lambda x: x in t[t <= 0.1])] = 0
#y_pred[train.keyword.apply(lambda x: x in t[t >= 0.9])] = 1

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print (confusion_matrix(y_test, y_pred))
print ("accuracy", (tp + tn) / (tn + fp + fn + tp))
print ("specificity", tn / (tn + fp))
print ("sensitivity", tp / (tp + fn))
# Train set
#[[4351    2]
# [   7 3253]]
#accuracy 0.9988178116379877
#specificity 0.9995405467493682
#sensitivity 0.9978527607361963

#train_vectors = count_vectorizer.fit_transform(train["text"])
sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = clf.predict(test_vectors)
sample_submission.to_csv("submission.csv", index = False)

