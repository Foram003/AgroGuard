import os
import shutil
import random
import matplotlib.pyplot as plt
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# -------------------- STEP 1: Define Paths --------------------
dataset_path = r"Data/quality_data"  # ✅ Change this to your dataset folder path
train_val_path = r"Data/train_val"
test_path = r"Data/test_val"

# Create train/val and test directories
os.makedirs(train_val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# -------------------- STEP 2: Split Dataset (90% Train/Validation, 10% Test) --------------------
if not os.listdir(train_val_path) and not os.listdir(test_path):  # Only split if not already done
    for grade in ["Grade A", "Grade B", "Grade C", "Grade D"]:
        grade_path = os.path.join(dataset_path, grade)
        images = os.listdir(grade_path)
        random.shuffle(images)

        # Split 90% for training/validation, 10% for testing
        train_val_images, test_images = train_test_split(images, test_size=0.1, random_state=42)

        os.makedirs(os.path.join(train_val_path, grade), exist_ok=True)
        os.makedirs(os.path.join(test_path, grade), exist_ok=True)

        # Move images to respective folders
        for img in train_val_images:
            shutil.copy(os.path.join(grade_path, img), os.path.join(train_val_path, grade, img))

        for img in test_images:
            shutil.copy(os.path.join(grade_path, img), os.path.join(test_path, grade, img))

    print("✅ Dataset split successfully!")
else:
    print("✅ Dataset already split!")

# -------------------- STEP 3: Data Augmentation --------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% training, 20% validation
)

test_datagen = ImageDataGenerator(rescale=1./255)  # No augmentation for test set

# -------------------- STEP 4: Load Train & Validation Data --------------------
train_set = train_datagen.flow_from_directory(
    train_val_path,
    target_size=(299, 299),  # InceptionV3 requires 299x299
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_set = train_datagen.flow_from_directory(
    train_val_path,
    target_size=(299, 299),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

test_set = test_datagen.flow_from_directory(
    test_path,
    target_size=(299, 299),
    batch_size=32,
    class_mode="categorical",
    shuffle=False  # No need to shuffle test set
)

# -------------------- STEP 5: Build InceptionV3 Model --------------------
base_model = InceptionV3(input_shape=(299, 299, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # Freeze base model layers

# Add Custom Layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)  # Increased neurons for deeper learning
x = Dropout(0.5)(x)
output_layer = Dense(4, activation="softmax")(x)  # 4 Classes (A, B, C, D)

# Create Final Model
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

print("✅ Model built successfully!")

# -------------------- STEP 6: Train the Model --------------------
history = model.fit(
    train_set,
    epochs=10,
    validation_data=val_set
)

# Save Model
model.save("fruit_quality_inception_model.h5")
print("✅ Model saved successfully!")

# -------------------- STEP 7: Plot Accuracy & Loss --------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()

# -------------------- STEP 8: Evaluate on Test Set --------------------
test_loss, test_acc = model.evaluate(test_set)
print(f"✅ Test Accuracy: {test_acc:.2%}")
