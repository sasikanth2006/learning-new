import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pylab
import seaborn as sns
import os
#os.chdir("D:\Sasi\shell work\Data")
os.chdir("/tmp")

#Loading adult census data file into dataframe.
adult_cns_income = pd.read_csv("adult.csv")
adult_cns_income.head(5)

#Finding the size of the dataset
adult_cns_income.shape

#Finding the data types of variables
adult_cns_income.info()

#Checking whether there is a class imbalance
adult_cns_income.groupby("income").size()

#Plotting the proportion of <=50K and >50K
adult_cns_income.groupby('income').size().plot(kind='pie', legend=True,autopct='%1.2f%%')

#Finding if there are any missing values
adult_cns_income.isnull().sum().sum()

import string
special_chars=string.punctuation
bools = list(map(lambda spch: spch in special_chars,adult_cns_income["workclass"]))
if any(bools):
    print("No Special Chars exist")
else :
    print("No Special Chars exist")

#Checking Distinct values and the count for the attribute workclass
adult_cns_income['workclass'].value_counts()

#Checking Distinct marital.status and the count for the attribute marital.status
adult_cns_income['marital.status'].value_counts()

#Checking Distinct values and the count for the attribute occupation
adult_cns_income['occupation'].value_counts()

#Checking Distinct values and the count for the attribute occupation
adult_cns_income['relationship'].value_counts()

#Checking Distinct values and the count for the attribute race
adult_cns_income['race'].value_counts()

#Checking Distinct values and the count for the attribute race
adult_cns_income['sex'].value_counts()

#Checking Distinct values and the count for the attribute native.country
adult_cns_income['native.country'].value_counts()

#Removing special characters if any with Null
adult_cns_income= adult_cns_income.replace(r'\?',np.nan,regex = True)

adult_cns_income.head(5)

adult_cns_income.isnull().any()

#Replacing special character in 'workclass' variable with its mode value
workclass_mode = adult_cns_income['workclass'].mode()[0]
adult_cns_income['workclass'].replace('\?', workclass_mode, regex=True, inplace=True)

#Replacing special character in 'occupation' variable with its mode value
workclass_mode = adult_cns_income['occupation'].mode()[0]
adult_cns_income['occupation'].replace('\?', workclass_mode, regex=True, inplace=True)

#Replacing special character in 'native.country' variable with its mode value
workclass_mode = adult_cns_income['native.country'].mode()[0]
adult_cns_income['native.country'].replace('\?', workclass_mode, regex=True, inplace=True)

adult_cns_income.head(5)

#Using Label encoder for the categorical variables to convert categorical data into numbers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
adult_cns_income['workclass'] = le.fit_transform(adult_cns_income['workclass'])
adult_cns_income['marital.status'] = le.fit_transform(adult_cns_income['marital.status'])
adult_cns_income['occupation'] = le.fit_transform(adult_cns_income['occupation'])
adult_cns_income['relationship'] = le.fit_transform(adult_cns_income['relationship'])
adult_cns_income['race'] = le.fit_transform(adult_cns_income['race'])
adult_cns_income['sex'] = le.fit_transform(adult_cns_income['sex'])
adult_cns_income['native.country'] = le.fit_transform(adult_cns_income['native.country'])
adult_cns_income['income'] = le.fit_transform(adult_cns_income['income'])

#Checking how categorical data is convereted into numbers after Label Encoder
adult_cns_income.head(5)

adult_cns_income = adult_cns_income.drop(["education","fnlwgt"],axis = 1)

adult_cns_income.head(5)

corr= adult_cns_income.corr()
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr, annot=True, vmax=.8, square=True,cmap = 'YlOrBr');

adult_cns_income
X = adult_cns_income.iloc[:,0:len(adult_cns_income.columns)-1]  #independent columns
Y = adult_cns_income.iloc[:,-1] # Dependent Colum or Target Varaible

X.count()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

Y_test.count()

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
gNB = GaussianNB()
Y_gNB_pred = gNB.fit(X_train,Y_train).predict(X_test)
print("Number of misclassified income out of total %d classes: %d"%(X_test.shape[0],(Y_test!=Y_gNB_pred).sum()))

print("Classsification Report:\n",classification_report(Y_test,Y_gNB_pred))
print("Confusion Matrix:\n",confusion_matrix(Y_test,Y_gNB_pred))
print("Accuracy Percentage is",accuracy_score(Y_test,Y_gNB_pred)*100)

Y_act_pred = pd.DataFrame()
Y_act_pred["Actual Income"]= pd.Series(Y_test)
Y_act_pred["gNB Predicted Income"] = Y_gNB_pred

from sklearn.naive_bayes import BernoulliNB
bNB = BernoulliNB()
Y_bNB_pred = bNB.fit(X_train,Y_train).predict(X_test)
print("Number of misclassified income out of total %d classes: %d"%(X_test.shape[0],(Y_test!=Y_bNB_pred).sum()))

print("Classsification Report:\n",classification_report(Y_test,Y_bNB_pred))
print("Confusion Matrix:\n",confusion_matrix(Y_test,Y_bNB_pred))
print("Accuracy Percentage is:",accuracy_score(Y_test,Y_bNB_pred)*100)

Y_act_pred["bNB Predicted Income"] = Y_bNB_pred

from sklearn.naive_bayes import MultinomialNB
mNB = MultinomialNB()
Y_mNB_pred = mNB.fit(X_train,Y_train).predict(X_test)
print("Number of misclassified income out of total %d classes: %d"%(X_test.shape[0],(Y_test!=Y_mNB_pred).sum()))

print("Classsification Report:\n",classification_report(Y_test,Y_mNB_pred))
print("Confusion Matrix:\n",confusion_matrix(Y_test,Y_mNB_pred))
print("Accuracy Percentage is:",accuracy_score(Y_test,Y_mNB_pred)*100)

Y_act_pred["mNB Predicted Income"] = Y_mNB_pred

Y_act_pred.isnull().any()

#Loading the Actual and predicted values of Y_test into csv file
Y_act_pred.to_csv("Q2_output.csv")
