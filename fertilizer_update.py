# Importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn import metrics  # Import metrics module
from sklearn.metrics import classification_report  # Import classification_report
warnings.filterwarnings('ignore')

# Importing the dataset
data = pd.read_csv('C:/Users/Vaidehi/Desktop/7 sem vaidehi/AgroGuard_updated/Data/train_data.csv')
data.head()

# Data preprocessing
data.info()
data.drop(columns=['Moisture'], inplace=True)

# Changing the column names
data.rename(columns={'Humidity ':'Humidity','Soil Type':'Soil_Type','Crop Type':'Crop_Type','Fertilizer Name':'Fertilizer'}, inplace=True)

# Encoding categorical variables
from sklearn.preprocessing import LabelEncoder

# Encoding Soil Type
encode_soil = LabelEncoder()
data.Soil_Type = encode_soil.fit_transform(data.Soil_Type)

# Creating the DataFrame for Soil Type
Soil_Type = pd.DataFrame(zip(encode_soil.classes_, encode_soil.transform(encode_soil.classes_)), columns=['Original', 'Encoded'])
Soil_Type = Soil_Type.set_index('Original')
print(Soil_Type)

# Encoding Crop Type
encode_crop = LabelEncoder()
data.Crop_Type = encode_crop.fit_transform(data.Crop_Type)

# Creating the DataFrame for Crop Type
Crop_Type = pd.DataFrame(zip(encode_crop.classes_, encode_crop.transform(encode_crop.classes_)), columns=['Original', 'Encoded'])
Crop_Type = Crop_Type.set_index('Original')
print(Crop_Type)

# Encoding Fertilizer
encode_ferti = LabelEncoder()
data.Fertilizer = encode_ferti.fit_transform(data.Fertilizer)

# Creating the DataFrame for Fertilizer
Fertilizer = pd.DataFrame(zip(encode_ferti.classes_, encode_ferti.transform(encode_ferti.classes_)), columns=['Original', 'Encoded'])
Fertilizer = Fertilizer.set_index('Original')
print(Fertilizer)

# Saving encoders
import pickle
with open('soil_encoder.pkl', 'wb') as f:
    pickle.dump(encode_soil, f)

with open('crop_encoder.pkl', 'wb') as f:
    pickle.dump(encode_crop, f)

# Splitting the data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data.drop('Fertilizer', axis=1), data.Fertilizer, test_size=0.2, random_state=1)
print('Shape of Splitting:')
print('x_train = {}, y_train = {}, x_test = {}, y_test = {}'.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

# Initialize variables for accuracy and model names
acc = []  # Test accuracy
acc1 = []  # Train accuracy
model = []  # Model names

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
RF1 = RandomForestClassifier(n_estimators=5, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=0)
RF1.fit(x_train, y_train)

# Predictions and accuracy
predicted_values = RF1.predict(x_test)
x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)

predicted_values1 = RF1.predict(x_train)
y = metrics.accuracy_score(y_train, predicted_values1)
acc1.append(y)

model.append('RF1')
print("RF's Accuracy is: ", x, y)

# Classification report
print(classification_report(y_test, predicted_values))

# Save the Random Forest model
import pickle
pickle_out = open('classifier.pkl', 'wb')
pickle.dump(RF1, pickle_out)
pickle_out.close()

# Load the model and make predictions
model = pickle.load(open('classifier.pkl', 'rb'))
print(model.predict([[34, 65, 0, 1, 7, 9, 30]]))  # Example prediction
print(model.predict([[28, 61, 2, 14, 31, 25, 38]]))  # Example prediction
print(model.predict([[26, 84, 2, 16, 92, 8, 54]]))  # Example prediction

# Save the Fertilizer encoder
pickle_out = open('fertilizer.pkl', 'wb')
pickle.dump(encode_ferti, pickle_out)
pickle_out.close()

# Load the Fertilizer encoder
ferti = pickle.load(open('fertilizer.pkl', 'rb'))
print(ferti.classes_[1])  # Example: Print the second class name