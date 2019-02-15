#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np  
import re  
import nltk  
from sklearn.datasets import load_files  
# nltk.download('stopwords')  
import pickle  
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[2]:


movie_data = load_files("txt_sentoken/")  
X, y = movie_data.data, movie_data.target


# In[3]:


print(X[0])


# In[4]:


print(y,len(y))


# In[5]:


documents = []
stemmer = WordNetLemmatizer()
for sen in range(0, len(X)):  
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)


# In[6]:


vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
X = vectorizer.fit_transform(documents).toarray()
tfidfconverter = TfidfTransformer()  
X = tfidfconverter.fit_transform(X).toarray()

# OR the below script can be used instead
# from sklearn.feature_extraction.text import TfidfVectorizer  
# tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
# X = tfidfconverter.fit_transform(documents).toarray()  


# In[7]:


print(X[0],len(X))


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)  
classifier.fit(X_train, y_train)


# In[9]:


y_pred = classifier.predict(X_test)


# In[10]:


print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))

