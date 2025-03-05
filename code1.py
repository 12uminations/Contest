import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import cv2
import pandas as pd
import os
from sklearn.metrics import accuracy_score

# Enable eager execution for compatibility with tf.data.Dataset
tf.compat.v1.enable_eager_execution()

# Load dataset
dataset_csv_path = "D:/CMU/computer vision/contest/data_from_questionaire.csv"
df = pd.read_csv(dataset_csv_path)

# Define image folder paths
image_folder_train = "D:/CMU/computer vision/contest/Questionair Images"
image_folder_test = "D:/CMU/computer vision/contest/Test Images"

# Function to preprocess images
def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = cv2.resize(image, (224, 224))  # Resize to match EfficientNet input size
        image = image / 255.0  # Normalize pixel values
        return image
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return np.zeros((224, 224, 3), dtype=np.float32)  # Return zero image to avoid breaking the pipeline

# Combine images and randomly swap order
def combine_images(img1_path, img2_path, label):
    img1 = preprocess_image(img1_path.numpy().decode('utf-8'))
    img2 = preprocess_image(img2_path.numpy().decode('utf-8'))
    
    # Randomly swap the order of images
    if np.random.rand() > 0.5:
        img1, img2 = img2, img1
        label = 1 - label  # Flip the label if the order is swapped
    
    return (img1, img2, tf.cast(label, tf.int32))  # Cast label to int32

# Load and preprocess data
def load_and_preprocess_data(image_paths1, image_paths2, labels):
    # Create pairs for the dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths1, image_paths2, labels))
    
    # Combine both images of each pair
    dataset = dataset.map(lambda img1, img2, label: tf.py_function(
        combine_images, [img1, img2, label], Tout=(tf.float32, tf.float32, tf.int32)
    ))

    # Unpack the tuple and set shapes
    def set_shapes(img1, img2, label):
        img1.set_shape((224, 224, 3))  # Image 1
        img2.set_shape((224, 224, 3))  # Image 2
        label.set_shape(())
        return (img1, img2), label

    dataset = dataset.map(set_shapes)
    dataset = dataset.shuffle(buffer_size=len(image_paths1))  # Shuffle the entire dataset
    dataset = dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

# Build CNN Model
def build_cnn_model():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True  # Fine-Tuning

    # Freeze the first 50 layers
    for layer in base_model.layers[:50]:
        layer.trainable = False

    # Inputs for two images
    input1 = layers.Input(shape=(224, 224, 3))
    input2 = layers.Input(shape=(224, 224, 3))

    # Process each image through the same base model
    output1 = base_model(input1)
    output2 = base_model(input2)

    # Global Average Pooling
    output1 = layers.GlobalAveragePooling2D()(output1)
    output2 = layers.GlobalAveragePooling2D()(output2)

    # Concatenate the outputs
    merged = layers.concatenate([output1, output2])

    # Add dense layers
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=[input1, input2], outputs=output)
    return model

# Prepare training data
positive_images = []
negative_images = []
labels = []
for _, row in df.iterrows():
    img1_path = os.path.join(image_folder_train, row['Image 1'])
    img2_path = os.path.join(image_folder_train, row['Image 2'])
    if os.path.exists(img1_path) and os.path.exists(img2_path):
        positive_images.append(img1_path)
        negative_images.append(img2_path)
        labels.append(1 if row['Winner'] == 1 else 0)

# Convert to numpy arrays
positive_images = np.array(positive_images)
negative_images = np.array(negative_images)
labels = np.array(labels)

# Load and preprocess training data
train_dataset = load_and_preprocess_data(positive_images, negative_images, labels)

# Build the CNN model
cnn_model = build_cnn_model()
cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-6)

# Train the model
history = cnn_model.fit(train_dataset, epochs=25, callbacks=[early_stopping, lr_scheduler], class_weight={0: 1.0, 1: 1.0})

# Save the model
cnn_model.save("cnn2.h5")

# Evaluation & Prediction
# Load and preprocess test data (similar to training data)
test_positive_images = [...]  # Add paths to test images
test_negative_images = [...]  # Add paths to test images
test_labels = [...]  # Add test labels

test_dataset = load_and_preprocess_data(test_positive_images, test_negative_images, test_labels)

# Evaluate the model
test_loss, test_accuracy = cnn_model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predict on the test set
y_pred = cnn_model.predict(test_dataset)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary labels

# Calculate accuracy
y_true = np.concatenate([y for _, y in test_dataset], axis=0)
accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")