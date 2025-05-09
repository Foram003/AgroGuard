from keras.models import load_model
from keras.layers import Input
from keras.models import Model

# Load the .h5 model file
model = load_model('Trained_model.h5', custom_objects={"Input": Input})

# Ensure the model's first layer matches the input shape
print(model.input_shape)
model.summary()
