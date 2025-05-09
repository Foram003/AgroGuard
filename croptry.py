import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 🔹 Load the dataset
crop = pd.read_csv('Data/crop_recommendation.csv')

# 🔹 Split features (X) and labels (Y)
X = crop.iloc[:, :-1].values  # Extract features
Y = crop.iloc[:, -1].values   # Extract crop labels (names)

# 🔹 Encode labels (Convert crop names to numbers)
le = LabelEncoder()
Y = le.fit_transform(Y)  # Now Y contains numbers instead of crop names

# 🔹 Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

# 🔹 Choose one test sample to check predictions
sample_index = 17  # Change this to see different test samples
test_sample = X_test[sample_index].reshape(1, -1)  # Reshape for prediction

# 🔹 Print feature values at sample index 0
print("\n🔹 Test Sample (Index 18) Feature Values:")
print(X_test[17])

# 🔹 Print actual crop label (before encoding)
actual_crop = le.inverse_transform([y_test[17]])  # Convert numeric label back to crop name
print("\n🔹 Actual Crop at Sample Index 0:", actual_crop[0])

# 🔹 Define models
models = [
    ('SVC', SVC(gamma='scale', probability=True)),
    ('svm1', SVC(probability=True, kernel='poly', degree=1)),
    ('svm2', SVC(probability=True, kernel='poly', degree=2)),
    ('svm3', SVC(probability=True, kernel='poly', degree=3)),
    ('rf', RandomForestClassifier(n_estimators=21,max_depth=100)),
    ('gnb', GaussianNB()),
    ('knn5', KNeighborsClassifier(n_neighbors=5)),
    ('knn7', KNeighborsClassifier(n_neighbors=7)),
    ('knn9', KNeighborsClassifier(n_neighbors=9))
]

# 🔹 Make predictions and show probabilities for each model
print("\n🔹 Model Predictions and Probabilities for Test Sample 0:")

for name, model in models:
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(test_sample)  # Predict on the single test sample
    crop_name = le.inverse_transform(y_pred)  # Convert back to crop name
    
    # Print predicted class and the probability distribution
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(test_sample)[0]  # Get the probability distribution for the sample
        prob_dict = {le.inverse_transform([i])[0]: prob for i, prob in enumerate(probs)}  # Convert to readable format
        print(f"{name}: Predicted: {crop_name[0]} | Probabilities: {prob_dict}")
        print()
    else:
        print(f"{name}: Predicted: {crop_name[0]} (No probability output)")

# 🔹 Voting Classifier (Soft Voting)
vot_soft = VotingClassifier(estimators=models, voting='soft')
vot_soft.fit(X_train, y_train)

# 🔹 Final Prediction by Voting Classifier
voting_pred = vot_soft.predict(test_sample)
final_prediction = le.inverse_transform(voting_pred)
print("\n🔹 Final Prediction by Voting Classifier:", final_prediction[0])

# 🔹 Cross-validation score
scores = cross_val_score(vot_soft, X_test, y_test, cv=5, scoring='accuracy')
print("\n🔹 Voting Classifier Accuracy:", scores.mean())

# 🔹 Save the model using Pickle
#with open('Crop_Recommendation.pkl', 'wb') as Model_pkl:
#    pickle.dump(vot_soft, Model_pkl)
