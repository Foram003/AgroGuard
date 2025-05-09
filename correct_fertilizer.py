# Importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Ignore warnings
warnings.filterwarnings('ignore')

# Importing the dataset
data = pd.read_csv(r'D:/AgroGuard_updated/Data/train_data.csv')
print("Dataset head:")
print(data.head())

# Data preprocessing
print("\nData information:")
data.info()

# Drop moisture column if it exists
if 'Moisture' in data.columns:
    data.drop(columns=['Moisture'], inplace=True)

# Changing the column names
data.rename(columns={
    'Humidity ': 'Humidity',
    'Soil Type': 'Soil_Type',
    'Crop Type': 'Crop_Type',
    'Fertilizer Name': 'Fertilizer'
}, inplace=True)

print("\nAfter renaming columns:")
print(data.columns)

# Encoding categorical variables
# Encoding Soil Type
encode_soil = LabelEncoder()
data.Soil_Type = encode_soil.fit_transform(data.Soil_Type)

# Encoding Crop Type
encode_crop = LabelEncoder()
data.Crop_Type = encode_crop.fit_transform(data.Crop_Type)

# Encoding Fertilizer
encode_ferti = LabelEncoder()
data.Fertilizer = encode_ferti.fit_transform(data.Fertilizer)

# Print encoded classes for reference
print("\nSoil Type classes:", encode_soil.classes_)
print("Crop Type classes:", encode_crop.classes_)
print("Fertilizer classes:", encode_ferti.classes_)

# Saving encoders
print("\nSaving encoders...")
with open('soil_encoder.pkl', 'wb') as f:
    pickle.dump(encode_soil, f)

with open('crop_encoder.pkl', 'wb') as f:
    pickle.dump(encode_crop, f)

with open('fertilizer_encoder.pkl', 'wb') as f:
    pickle.dump(encode_ferti, f)

# Splitting the data into train and test
X = data.drop('Fertilizer', axis=1)
y = data.Fertilizer

print("\nFeatures used for training:")
print(X.columns)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print('\nShape of Splitting:')
print('x_train = {}, y_train = {}, x_test = {}, y_test = {}'.format(
    x_train.shape, y_train.shape, x_test.shape, y_test.shape))

# Random Forest Classifier
print("\nTraining Random Forest model...")
RF1 = RandomForestClassifier(n_estimators=100, max_depth=10,
                            min_samples_split=5, min_samples_leaf=2,
                            random_state=0)
RF1.fit(x_train, y_train)

# Predictions and accuracy
predicted_values = RF1.predict(x_test)
accuracy = metrics.accuracy_score(y_test, predicted_values)
print("Random Forest's Accuracy is: {:.2f}%".format(accuracy * 100))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': RF1.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Save the Random Forest model
print("\nSaving model...")
pickle_out = open('classifier.pkl', 'wb')
pickle.dump(RF1, pickle_out)
pickle_out.close()

# Load the model and make a sample prediction
print("\nLoading model for a test prediction...")
model = pickle.load(open('classifier.pkl', 'rb'))

# Get a sample row from the test set to ensure correct format
sample_input = x_test.iloc[0].values.reshape(1, -1)
print(f"Sample input shape: {sample_input.shape}")
print(f"Sample input values: {sample_input}")

# Make prediction
prediction = model.predict(sample_input)
print(f"Predicted fertilizer (encoded): {prediction[0]}")
print(f"Predicted fertilizer (decoded): {encode_ferti.inverse_transform([prediction[0]])[0]}")

# Example of how to use the model with new data
print("\nExample of prediction with new data:")
# Note: This example assumes the same feature order as in X.columns
# Replace these values with appropriate values for your use case
example_data = [[34, 65, 0, 1, 7, 9, 30]]  # Adjust based on your actual features
print(f"Example input: {example_data}")

# Ensure the example has the correct number of features
if len(example_data[0]) == len(X.columns):
    example_prediction = model.predict(example_data)
    print(f"Example prediction (encoded): {example_prediction[0]}")
    print(f"Example prediction (decoded): {encode_ferti.inverse_transform([example_prediction[0]])[0]}")
else:
    print(f"Error: Example data has {len(example_data[0])} features, but model requires {len(X.columns)} features")
    print(f"Required features in order: {X.columns.tolist()}")