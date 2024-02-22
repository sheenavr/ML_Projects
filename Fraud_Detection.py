#!/usr/bin/env python
# coding: utf-8

# # Online Payments Fraud Detection with Machine Learning
Online payment frauds can happen with anyone using any payment system, especially while making payments using a credit card. That is why detecting online payment fraud is very important for credit card companies to ensure that the customers are not getting charged for the products and services they never paid. If you want to learn how to detect online payment frauds, this article is for you. In this article, I will take you through the task of online payments fraud detection with machine learning using Python.To identify online payment fraud with machine learning, we need to train a machine learning model for classifying fraudulent and non-fraudulent payments. For this, we need a dataset containing information about online payment fraud, so that we can understand what type of transactions lead to fraud. For this task, I collected a dataset from Kaggle, which contains historical information about fraudulent transactions which can be used to detect fraud in online payments. Below are all the columns from the dataset I’m using here:

step: represents a unit of time where 1 step equals 1 hour
type: type of online transaction
amount: the amount of the transaction
nameOrig: customer starting the transaction
oldbalanceOrg: balance before the transaction
newbalanceOrig: balance after the transaction
nameDest: recipient of the transaction
oldbalanceDest: initial balance of recipient before the transaction
newbalanceDest: the new balance of recipient after the transaction
isFraud: fraud transaction
I hope you now know about the data I am using for the online payment fraud detection task. Now in the section below, I’ll explain how we can use machine learning to detect online payment fraud using Python.

Online Payments Fraud Detection using Python
I will start this task by importing the necessary Python libraries and the dataset we need for this task:
# In[3]:


import pandas as pd
import numpy as np
data = pd.read_csv("C:/Users/vrshe/OneDrive/Desktop/Machine_Learning_code/creditcard.csv")
print(data.head())

Now, let’s have a look at whether this dataset has any null values or not:
# In[4]:


print(data.isnull().sum())

So this dataset does not have any null values. Before moving forward, now, let’s have a look at the type of transaction mentioned in the dataset:
# In[5]:


# Exploring transaction type
print(data.type.value_counts())


# In[8]:


type = data["type"].value_counts()
transactions = type.index
quantity = type.values

import plotly.express as px
figure = px.pie(data, 
             values=quantity, 
             names=transactions,hole = 0.5, 
             title="Distribution of Transaction Type")
figure.show()

Now let’s have a look at the correlation between the features of the data with the isFraud column:
# In[10]:


# Checking correlation for numeric columns
numeric_data = data.select_dtypes(include=['number'])

try:
    correlation = numeric_data.corr()
    print(correlation["isFraud"].sort_values(ascending=False))
except KeyError:
    print("The column 'isFraud' does not exist in the dataset.")
except Exception as e:
    print("An error occurred:", e)

Now let’s transform the categorical features into numerical. Here I will also transform the values of the isFraud column into No Fraud and Fraud labels to have a better understanding of the output:
# In[11]:


data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, 
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
print(data.head())

Online Payments Fraud Detection Model
Now let’s train a classification model to classify fraud and non-fraud transactions. Before training the model, I will split the data into training and test sets:
# In[12]:


# splitting the data
from sklearn.model_selection import train_test_split
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])

Now let’s train the online payments fraud detection model:
# In[13]:


# training a machine learning model
from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))


# In[14]:


# prediction
#features = [type, amount, oldbalanceOrg, newbalanceOrig]
features = np.array([[4, 9000.60, 9000.60, 0.0]])
print(model.predict(features))


# In[ ]:




