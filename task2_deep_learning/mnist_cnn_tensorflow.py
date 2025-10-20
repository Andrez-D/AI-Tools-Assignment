"""
Task 2: Deep Learning with TensorFlow
Dataset: MNIST Handwritten Digits
Goal: Build a CNN model to classify handwritten digits (>95% accuracy)
Author: [Kipruto Andrew Kipngetich]
Date: October 2025
"""

# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("TASK 2: MNIST DIGIT CLASSIFICATION WITH CONVOLUTIONAL NEURAL NETWORK")
print("="*80)

# Check TensorFlow version and GPU availability
print(f"\nTensorFlow Version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
if tf.config.list_physical_devices('GPU'):
    print("✓ GPU acceleration enabled!")
else:
    print("⚠ Running on CPU (slower but functional)")

# ============================================================================
# STEP 1: LOAD AND EXPLORE THE MNIST DATASET
# ============================================================================
print("\n[STEP 1] Loading MNIST Dataset...")

# Load MNIST dataset (automatically downloads if not present)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"\n✓ Dataset loaded successfully!")
print(f"  - Training samples: {X_train.shape[0]}")
print(f"  - Test samples: {X_test.shape[0]}")
print(f"  - Image shape: {X_train.shape[1:]} (28x28 pixels)")
print(f"  - Number of classes: {len(np.unique(y_train))} (digits 0-9)")

# Display class distribution
print(f"\nTraining set class distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for digit, count in zip(unique, counts):
    print(f"  Digit {digit}: {count} samples ({count/len(y_train)*100:.1f}%)")

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================
print("\n[STEP 2] Preprocessing Data...")

# Reshape data to include channel dimension (required for CNN)
# From (28, 28) to (28, 28, 1) for grayscale
X_train_reshaped = X_train.reshape(-1, 28, 28, 1)
X_test_reshaped = X_test.reshape(-1, 28, 28, 1)

# Normalize pixel values from [0, 255] to [0, 1]
X_train_normalized = X_train_reshaped.astype('float32') / 255.0
X_test_normalized = X_test_reshaped.astype('float32') / 255.0

print(f"\n✓ Data preprocessing completed!")
print(f"  - Original pixel range: [0, 255]")
print(f"  - Normalized pixel range: [{X_train_normalized.min():.1f}, {X_train_normalized.max():.1f}]")
print(f"  - Training data shape: {X_train_normalized.shape}")
print(f"  - Test data shape: {X_test_normalized.shape}")

# One-hot encode labels for categorical crossentropy
y_train_categorical = to_categorical(y_train, 10)
y_test_categorical = to_categorical(y_test, 10)

print(f"\n✓ Labels one-hot encoded!")
print(f"  - Example: Label {y_train[0]} → {y_train_categorical[0]}")

# ============================================================================
# STEP 3: VISUALIZE SAMPLE IMAGES
# ============================================================================
print("\n[STEP 3] Visualizing Sample Images...")

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Sample MNIST Digits', fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(f'Label: {y_train[i]}', fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.savefig('mnist_sample_images.png', dpi=300, bbox_inches='tight')
print("✓ Sample images saved as 'mnist_sample_images.png'")
plt.show()

# ============================================================================
# STEP 4: BUILD CNN MODEL ARCHITECTURE
# ============================================================================
print("\n[STEP 4] Building CNN Model Architecture...")

# Define the CNN model
model = models.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv1'),
    layers.BatchNormalization(name='bn1'),
    layers.MaxPooling2D((2, 2), name='pool1'),
    layers.Dropout(0.25, name='dropout1'),
    
    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
    layers.BatchNormalization(name='bn2'),
    layers.MaxPooling2D((2, 2), name='pool2'),
    layers.Dropout(0.25, name='dropout2'),
    
    # Third Convolutional Block
    layers.Conv2D(128, (3, 3), activation='relu', name='conv3'),
    layers.BatchNormalization(name='bn3'),
    layers.Dropout(0.4, name='dropout3'),
    
    # Flatten and Dense Layers
    layers.Flatten(name='flatten'),
    layers.Dense(128, activation='relu', name='dense1'),
    layers.BatchNormalization(name='bn4'),
    layers.Dropout(0.5, name='dropout4'),
    layers.Dense(10, activation='softmax', name='output')
])

print("\n✓ Model architecture created!")

# Display model summary
print("\nModel Architecture Summary:")
print("="*80)
model.summary()
print("="*80)

# Count total parameters
total_params = model.count_params()
print(f"\nTotal Parameters: {total_params:,}")

# ============================================================================
# STEP 5: COMPILE THE MODEL
# ============================================================================
print("\n[STEP 5] Compiling Model...")

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model compiled successfully!")
print(f"  - Optimizer: Adam")
print(f"  - Loss Function: Categorical Crossentropy")
print(f"  - Metrics: Accuracy")

# ============================================================================
# STEP 6: TRAIN THE MODEL
# ============================================================================
print("\n[STEP 6] Training Model...")

# Define callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

# Train the model
print("\nStarting training process...")
print("-" * 80)

history = model.fit(
    X_train_normalized,
    y_train_categorical,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("\n✓ Training completed!")

# ============================================================================
# STEP 7: EVALUATE MODEL PERFORMANCE
# ============================================================================
print("\n[STEP 7] Evaluating Model Performance...")

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test_normalized, y_test_categorical, verbose=0)

print("\n" + "="*80)
print("MODEL PERFORMANCE METRICS")
print("="*80)

print(f"\nTest Set Results:")
print(f"  - Loss: {test_loss:.4f}")
print(f"  - Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Check if we achieved the target accuracy
if test_accuracy > 0.95:
    print(f"\n✓ SUCCESS! Achieved target accuracy (>95%)")
else:
    print(f"\n⚠ Target accuracy not reached. Current: {test_accuracy*100:.2f}%")

# Make predictions on test set
y_pred_prob = model.predict(X_test_normalized, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)

# Classification report
print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))

# ============================================================================
# STEP 8: VISUALIZE TRAINING HISTORY
# ============================================================================
print("\n[STEP 8] Creating Training Visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Plot training & validation accuracy
axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='Target (95%)')

# Plot training & validation loss
axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mnist_training_history.png', dpi=300, bbox_inches='tight')
print("✓ Training history saved as 'mnist_training_history.png'")
plt.show()

# ============================================================================
# STEP 9: VISUALIZE CONFUSION MATRIX
# ============================================================================
print("\n[STEP 9] Creating Confusion Matrix...")

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, 
            xticklabels=range(10), yticklabels=range(10), cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - MNIST Digit Classification', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
plt.tight_layout()
plt.savefig('mnist_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrix saved as 'mnist_confusion_matrix.png'")
plt.show()

# ============================================================================
# STEP 10: VISUALIZE MODEL PREDICTIONS ON SAMPLE IMAGES
# ============================================================================
print("\n[STEP 10] Visualizing Model Predictions on Sample Images...")

# Select 5 random test images
sample_indices = np.random.choice(len(X_test), 5, replace=False)

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle('Model Predictions on Sample Test Images', fontsize=16, fontweight='bold')

for idx, ax in zip(sample_indices, axes):
    # Get image and prediction
    image = X_test[idx]
    true_label = y_test[idx]
    pred_probs = y_pred_prob[idx]
    pred_label = np.argmax(pred_probs)
    confidence = pred_probs[pred_label] * 100
    
    # Display image
    ax.imshow(image, cmap='gray')
    
    # Color code: green if correct, red if incorrect
    color = 'green' if pred_label == true_label else 'red'
    title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%'
    ax.set_title(title, fontsize=12, color=color, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('mnist_sample_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Sample predictions saved as 'mnist_sample_predictions.png'")
plt.show()

# Print detailed predictions
print("\nDetailed Sample Predictions:")
print("="*80)
for i, idx in enumerate(sample_indices):
    true_label = y_test[idx]
    pred_label = y_pred[idx]
    confidence = y_pred_prob[idx][pred_label] * 100
    status = "✓ CORRECT" if pred_label == true_label else "✗ INCORRECT"
    print(f"\nSample {i+1}:")
    print(f"  True Label: {true_label}")
    print(f"  Predicted: {pred_label}")
    print(f"  Confidence: {confidence:.2f}%")
    print(f"  Status: {status}")

# ============================================================================
# STEP 11: SAVE THE MODEL
# ============================================================================
print("\n[STEP 11] Saving Model...")

# Save model in TensorFlow format
model.save('mnist_cnn_model.h5')
print("✓ Model saved as 'mnist_cnn_model.h5'")

# Save model in SavedModel format (recommended for production)
model.save('mnist_cnn_model_savedmodel')
print("✓ Model saved in SavedModel format to 'mnist_cnn_model_savedmodel/'")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TASK 2 COMPLETION SUMMARY")
print("="*80)

print("\n✓ All steps completed successfully!")
print(f"\nKey Results:")
print(f"  - Model Type: Convolutional Neural Network (CNN)")
print(f"  - Dataset: MNIST (60,000 train + 10,000 test)")
print(f"  - Final Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  - Target Achieved: {'YES ✓' if test_accuracy > 0.95 else 'NO ✗'}")
print(f"  - Total Parameters: {total_params:,}")
print(f"  - Training Epochs: {len(history.history['accuracy'])}")

print(f"\nModel Architecture:")
print(f"  - Conv Layers: 3 (32, 64, 128 filters)")
print(f"  - Pooling Layers: 2 (MaxPooling2D)")
print(f"  - Dense Layers: 2 (128 units + 10 output)")
print(f"  - Dropout Layers: 4 (25%, 25%, 40%, 50%)")
print(f"  - Batch Normalization: 4 layers")

print(f"\nFiles Generated:")
print(f"  1. mnist_sample_images.png")
print(f"  2. mnist_training_history.png")
print(f"  3. mnist_confusion_matrix.png")
print(f"  4. mnist_sample_predictions.png")
print(f"  5. mnist_cnn_model.h5")
print(f"  6. mnist_cnn_model_savedmodel/")

print("\n" + "="*80)
print("Task 2 execution complete! 🎉")
print("="*80)

# ============================================================================
# BONUS: HOW TO LOAD AND USE THE MODEL
# ============================================================================
print("\n" + "="*80)
print("BONUS: HOW TO USE THE SAVED MODEL")
print("="*80)

print("\nExample code to load and use the model:")
print("""
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# Load and preprocess a custom image
img = Image.open('your_digit.png').convert('L')  # Convert to grayscale
img = img.resize((28, 28))
img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0

# Make prediction
prediction = model.predict(img_array)
predicted_digit = np.argmax(prediction)
confidence = prediction[0][predicted_digit] * 100

print(f"Predicted digit: {predicted_digit}")
print(f"Confidence: {confidence:.2f}%")
""")