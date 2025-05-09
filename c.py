from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import model_selection

# Load dataset
crop = pd.read_csv('Data/crop_recommendation.csv')
X = crop.iloc[:, :-1].values
Y = crop.iloc[:, -1].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

# Define models
models = [
    ('SVC', SVC(gamma='scale', probability=True, C=0.1)),
    ('svm1', SVC(probability=True, kernel='poly', degree=1)),
    ('svm2', SVC(probability=True, kernel='poly', degree=2)),
    ('svm3', SVC(probability=True, kernel='poly', degree=3)),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
    ('gnb',GaussianNB(var_smoothing=1e-9)),
    ('knn7', KNeighborsClassifier(n_neighbors=7)),
    ('knn9', KNeighborsClassifier(n_neighbors=9))
]

# Train and evaluate each model individually
for name, model in models:
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name} - Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

# Voting Classifier
vot_soft = VotingClassifier(estimators=models, voting='soft')
vot_soft.fit(X_train, y_train)
y_pred = vot_soft.predict(X_test)

# Evaluate Voting Classifier
train_acc_vot = accuracy_score(y_train, vot_soft.predict(X_train))
test_acc_vot = accuracy_score(y_test, y_pred)
print(f"Voting Classifier - Train Accuracy: {train_acc_vot:.4f}, Test Accuracy: {test_acc_vot:.4f}")

# Cross-validation score
scores = model_selection.cross_val_score(vot_soft, X_test, y_test, cv=5, scoring='accuracy')
print("Voting Classifier Cross-Validation Accuracy: ", scores.mean())
from collections import Counter
print(Counter(y_train))
print(Counter(y_test))

