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
X = crop.iloc[:, :-1].values  # Extract features (all columns except last)
Y = crop.iloc[:, -1].values  # Extract labels (crop names)

# 🔹 Encode labels (Convert crop names to numbers)
le = LabelEncoder()
Y = le.fit_transform(Y)  # Now Y contains numbers instead of crop names

# 🔹 Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

# 🔹 Take manual input from the user
print("\n🔹 Enter values for the following features:")
N = float(input("Nitrogen (N): "))
P = float(input("Phosphorus (P): "))
K = float(input("Potassium (K): "))
temperature = float(input("Temperature (°C): "))
humidity = float(input("Humidity (%): "))
ph = float(input("pH Level: "))
rainfall = float(input("Rainfall (mm): "))

# 🔹 Create the test sample with manual input
test_sample = np.array([[N, P, K, temperature, humidity, ph, rainfall]]).reshape(1, -1)

# 🔹 Define models
models = [
    ('SVC', SVC(gamma='scale', probability=True)),
    ('svm1', SVC(probability=True, kernel='poly', degree=1)),
    ('svm2', SVC(probability=True, kernel='poly', degree=2)),
    ('svm3', SVC(probability=True, kernel='poly', degree=3)),
    ('rf', RandomForestClassifier(n_estimators=21, max_depth=100)),
    ('gnb', GaussianNB()),
    ('knn5', KNeighborsClassifier(n_neighbors=5)),
    ('knn7', KNeighborsClassifier(n_neighbors=7)),
    ('knn9', KNeighborsClassifier(n_neighbors=9))
]

# 🔹 Make predictions and show probabilities for each model
print("\n🔹 Model Predictions and Probabilities for Custom Input:")

for name, model in models:
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(test_sample)  # Predict on the manually entered sample
    crop_name = le.inverse_transform(y_pred)  # Convert back to crop name

    # Print predicted class and probability distribution
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(test_sample)[0]  # Get probability distribution
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
# with open('Crop_Recommendation.pkl', 'wb') as Model_pkl:
#     pickle.dump(vot_soft, Model_pkl)