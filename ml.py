import numpy as np
import pandas as pd

train = pd.read_csv('Training.csv')
test = pd.read_csv('Testing.csv')

#EDA    
train.info()
train.describe()
train.head()
train.tail()
train.isnull().sum()
train.dtypes

# make a list of symptoms from the dataset
symptoms = list(train.columns[1:132])
symptoms

# make a list of diseases from the dataset
diseases = list(train['prognosis'].unique())
diseases

#make a model using Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = train[symptoms]
y = train['prognosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

# test the model using the test dataset
X_test = test[symptoms]
y_test = test['prognosis']
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

# save the model using pickle
import pickle
pickle.dump(clf, open('model.pkl','wb'))


#also make a model using Random Forest Classifier and save it and compare the accuracy
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

pickle.dump(clf, open('model2.pkl','wb'))

# compare the accuracy of both the models using the test dataset
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

#save all the symptoms and diseases in a json file
import json
data = {'symptoms': symptoms, 'diseases': diseases}
with open('data.json', 'w') as f:
    json.dump(data, f)

