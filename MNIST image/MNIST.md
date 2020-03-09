## MNIST Kaggle
The MNIST is a public data set comprising of handwritten digits (0 to 9).
The Kaggle data competition is available [here](https://www.kaggle.com/c/digit-recognizer).

The training (and test) set is made up of 784 pixels to represent a 28 by 28 image.
'label' is the target class (0 to 9):
<br> 

### Initial Set-up
#### Loading training file and import required libraries

```Python
# Imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
import tensorflow as tf
import operator # Accessing indices quickly

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
```

A sample of the data:
```Python
train.head(4) 
```

label |	pixel0 |	pixel1 |	pixel2 |	... | pixel779 | pixel780 | pixel781 | pixel782 | pixel783  
------|--------|---------|---------|------|--------|----------|----------|----------|-----------  
0 |	1 | 0 |	0  |	... | 0    |    1 	 |     0 |	0    |    0  
1 |	0 |	0 |	0  |	... | 1    |    0 	 |     0 |	0    |    0  
2 |	1 |	0 |	0  |	... | 2    |    1 	 |     0 |	0    |    0
<br>

By rearranging the pixels to 28x28, we can plot the image:
<br>

```Python
def print_img(sample_image):    
    plt.axis("off")
    plt.imshow(sample_image, cmap = cm.binary)
    plt.show()

X = train.iloc[:, 1:]
y = train.iloc[:, 0]

print_img(np.array(X.iloc[3]).reshape(28, 28))
print ("Image label: ", y[3])
```
<img height = "200" src = "https://bit.ly/3c4Mfdt" />

>Image label: 4
<br><br><br>

### Neural Network
#### Preprocessing
Splitting the initial data set into train and validation data sets. A validation set is used, to measure the accuracy of the model on a test set. 
We set the seed for train_test_split = 1, so that subsequent runs will generate the same results.

``` Python
# input & output of NN
image_size = 784
labels_count = 10
image_width = image_height = 28

X_1 = train.iloc[:, 1:]/255
X = np.array([np.array(X_1.iloc[x]).reshape(image_width, image_height, 1) for x in range(len(X_1))])

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.3, random_state = 1)
```
<br>

#### Modelling
So the summary of the model is first a 3x3 convolution layer, then a 2x2 pooling (obtain the max of each pool), then a hidden layer with 256 nodes and dropout value of 0.2. Finally, the output layer with a softmax function to give the probability distribution of each row (image) to each label class (0 to 9).  
The speed to train the model varies, but each epoch should take ~39s.

```Python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation = 'relu', input_shape = (image_width, image_height, 1)),
    tf.keras.layers.MaxPool2D(pool_size = 2, strides = 1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

history = model.fit(train_X, np.array(train_y), epochs = 5) 
```
<br>

#### Checking with validation set
Now we will apply the model to the validation set that we separated earlier.

```Python
# Probability class for each class
pred = model.predict(val_X)

pred_prob = [max(enumerate(x), key = operator.itemgetter(1))[1] for x in pred]
pred_int = [max(enumerate(x), key = operator.itemgetter(1))[0] for x in pred]

# Find the ones where the predictions are wrong
errors_idx = [x for x, (e1, e2) in enumerate(zip(pred_int, val_y)) if e1 != e2]
print ("Number of misclassifications: ", len(errors_idx)) 
```

Just a sample of what was misclassified with the validation set:
```Python
sample = errors_idx[19]
print_img(val_X[sample].reshape(28, 28))
print ("Sample misclassification")
print ("Prediction: ", pred_int[sample])
print ("Actual: ", val_y.iloc[sample])
```
>[0.06198824127624181, 0.985]  
Number of misclassifications:  189

<img height = "200" src = "https://bit.ly/2VhFcZi" />

>Sample misclassification  
Prediction:  7  
Actual:  4

<br><br><br>

### Test Set & Submission
Now that the first CNN model has been trained, we can apply it the test set and submit on the leaderboard.

```Python
# Test set
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
test = test/255
test = np.array([np.array(test.iloc[x]).reshape(image_width, image_height, 1) 
                 for x in range(len(test))])
      
# Submission
test_pred = model.predict(test)
test_int = [max(enumerate(x), key = operator.itemgetter(1))[0] for x in test_pred]
submission = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
submission["Label"] = test_int
submission.to_csv("submission.csv", index = False)
```
The test file gives 0.98400 on the leaderboard.
Results are similar to validation set accuracy, so both train and test sets have a similar distribution.
This ensures that the model trained can be used for test set.

