A repository to store my data science projects and some code snippets. 
Data is often obtained from Kaggle competitions 
(links will be provided to the respective competitions if that is the case).  



## PROJECTS:  
  
**[MNIST](https://github.com/mango-muffin/Projects/blob/master/MNIST%20image/MNIST.md)**  


MNIST is a famous data set of numerical images, 28x28 pixels.  
Currently, the folder contains a CNN method using a training set to predict the numerals of the **test** data.

A brief data exploration (more details in the project folder):  
The images are in grayscale (with black being 255).  
The values will be scaled (i.e. value/255) for better performance.  

A sample of the train set:  
![](https://user-images.githubusercontent.com/40700585/76387146-30554300-63a1-11ea-8ed0-dc1e329e70ad.png)

In general, the distribution of the train set is quite even, and assuming the test set has a similar distribution and features (it does), the train set is a good indicator for the test set.  
Distribution of labels of the train set:  

![MNIST_train_distribution](https://user-images.githubusercontent.com/40700585/76390522-87aae180-63a8-11ea-92c1-288edfdc5600.png)



<br><br><br>
__________________________________________________________________________________________________________________________________  
**[Disaster Tweets](https://github.com/mango-muffin/Projects/tree/master/NLP%20Disaster%20Tweets)**


The train set for this is a collection of >7500 tweets, labelled disaster tweet (1) or non-disaster tweet (0).  
Currently the folder contains a ridge regression analysis to predict whether a tweet is about a disaster or not.


