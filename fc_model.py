import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn.metrics as metrics

# --- 1. Configuration ---
DATASET_DIR = 'dataset_fc'
IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128
BATCH_SIZE = 32
EPOCHS = 15 # You might need to tune this (e.g., 10-20)

# --- 1b. Check for GPU ---
print("--- Checking for GPU ---")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs found")
        print("--- GPU will be used! ---")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("--- No GPU found, will use CPU ---")


# --- 2. Load Datasets ---
print(f"Loading datasets from '{DATASET_DIR}'...")

# Note: We use 'label_mode='binary'' for 2-class problems.
# 'color_mode='grayscale'' matches your data generator.
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, 'train'),
    label_mode='binary',
    color_mode='grayscale',
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, 'validation'),
    label_mode='binary',
    color_mode='grayscale',
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False # No need to shuffle validation data
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, 'test'),
    label_mode='binary',
    color_mode='grayscale',
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False # No need to shuffle test data
)

# Get class names - Keras will infer them from the folders.
# E.g., ['fearful', 'neutral']. This is CRITICAL for interpretation.
class_names = train_ds.class_names
print(f"Class names found: {class_names}")
print(f"  -> '{class_names[0]}' will be class 0")
print(f"  -> '{class_names[1]}' will be class 1")

# --- 3. Configure Dataset for Performance ---
# Use caching and prefetching for faster training
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 4. Build the CNN Model ---
# We'll build a simple Convolutional Neural Network (CNN),
# which is excellent for image shape recognition.
# 
model = keras.Sequential([
    # Input layer: Rescale pixel values from [0, 255] to [0, 1]
    layers.Rescaling(1./255, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)),
    
    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flatten the 3D feature maps to a 1D vector
    layers.Flatten(),
    
    # Fully-connected dense layer
    layers.Dense(64, activation='relu'),
    
    # Output layer: 1 node with a sigmoid activation
    # 'sigmoid' squashes the output to be between 0 and 1,
    # perfect for a binary probability.
    layers.Dense(1, activation='sigmoid')
])

# --- 5. Compile the Model ---
model.compile(
    optimizer='adam',
    loss='binary_crossentropy', # The standard loss for binary (0/1) classification
    metrics=['accuracy']
)

# Show the model's architecture
model.summary()

# --- 6. Train the Model ---
print("\nStarting model training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)
print("Training complete.")

# --- 7. Evaluate the Model ---
print("\n--- Model Evaluation ---")

# Evaluate on all three splits to see the effect of the bias
train_loss, train_acc = model.evaluate(train_ds, verbose=0)
print(f"Training Accuracy (Biased):   {train_acc:.4f}")

val_loss, val_acc = model.evaluate(val_ds, verbose=0)
print(f"Validation Accuracy (Balanced): {val_acc:.4f}")

test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"Test Accuracy (Balanced):     {test_acc:.4f}")


# --- 8. Plot Training History ---
# This helps us see overfitting
print("\nGenerating training history plots...")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.savefig('training_history.png')
print("Saved training plot to 'training_history.png'")
plt.show()

# --- 9. Detailed Test Set Analysis (Confusion Matrix) ---
# This is the most important part for your "fear conditioning" test
print("\n--- Detailed Test Set Analysis ---")

# Get true labels from the test dataset
y_true = []
for images, labels in test_ds:
    y_true.extend(labels.numpy())
y_true = np.array(y_true).astype(int) # Ensure labels are integers (0 or 1)

# Get model predictions
y_pred_probs = model.predict(test_ds)
y_pred = (y_pred_probs > 0.5).astype(int).flatten() # Convert probabilities to 0 or 1

# 
# Generate and print the classification report
print("Classification Report:")
# 'target_names' must match the order Keras found: 0 and 1
print(metrics.classification_report(y_true, y_pred, target_names=class_names))

# Generate and print the confusion matrix
print("Confusion Matrix:")
cm = metrics.confusion_matrix(y_true, y_pred)
print(cm)

# A more visual display of the confusion matrix
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot()
plt.title('Confusion Matrix for Test Set')
plt.savefig('confusion_matrix.png')
print("Saved confusion matrix plot to 'confusion_matrix.png'")
plt.show()
