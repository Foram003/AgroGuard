# -*- coding: utf-8 -*-
"""Untitled17.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZLDvvbKERvzn9wiYedq_dLd-hm6hlcnw
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import model_selection

crop = pd.read_csv('Data/crop_recommendation_new.csv')
X = crop.iloc[:,:-1].values
Y = crop.iloc[:,-1].values

import numpy as np

np.unique(Y)

pd.Series(Y).value_counts()

crop = crop[crop['label'] != 'Lentil']

zero_counts = (crop[['N', 'P', 'K']] == 0).sum()
print("Zero values in N, P, K:")
print(zero_counts)

crop['N'] = crop['N'].mask(crop['N'] == 0, crop.groupby(crop.columns[-1])['N'].transform('median'))


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
crop['label_encoded'] = le.fit_transform(crop['label'])  # Encoding

# Create a mapping of encoded labels to original crop names
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(label_mapping)

"""3. Feature Scaling (Standardization)
SVM & KNN are sensitive to feature scales, so we must standardize numeric features:
"""

from sklearn.preprocessing import StandardScaler

# Drop the categorical "label" column, keep only numerical features
X = crop.drop(columns=['label'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Standardize numerical features
  # Standardize features

print(crop.dtypes)

print(crop['label'].value_counts())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.15, random_state=42, stratify=Y)

models = []
models.append(('SVC', SVC(gamma ='scale', probability = True)))
models.append(('svm1', SVC(probability=True, kernel='poly', degree=1)))
models.append(('svm2', SVC(probability=True, kernel='poly', degree=2)))
models.append(('svm3', SVC(probability=True, kernel='poly', degree=3)))
models.append(('rf',RandomForestClassifier(n_estimators = 21,max_depth=100)))
models.append(('gnb',GaussianNB()))
models.append(('knn5', KNeighborsClassifier(n_neighbors=5)))
models.append(('knn7', KNeighborsClassifier(n_neighbors=7)))
models.append(('knn9', KNeighborsClassifier(n_neighbors=9)))

vot_soft = VotingClassifier(estimators=models, voting='soft')
vot_soft.fit(X_train, y_train)
y_pred = vot_soft.predict(X_test)

scores = model_selection.cross_val_score(vot_soft, X_test, y_test,cv=5,scoring='accuracy')
print("Accuracy: ",scores.mean())

import pickle

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('Crop_Recommendation.pkl', 'wb') as f:
    pickle.dump(vot_soft, f)