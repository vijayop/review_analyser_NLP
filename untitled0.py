# Natural language processing 

# importing the libraries  
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset 
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t' , quoting = 3)  # delimiter = '\t' specifies that the file is tsv , and quoting = 3 is used to ignore the double quotes

# cleaning the texts  
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = [] # corpus is a collection of texts
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'] [i])     # dataset['review'] will fetch the review column from the dataset. [i] is used for iteration purpose.
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]  # stem converts the word into its root word , eg. loved will become love
    review = ' '.join(review)    # join the words in the review list to mwake it a string , and it separated by a space
    corpus.append(review)
    
# Creating the bag of words model 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)   # this will include only 1500 most frequent words in the bag of words
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)