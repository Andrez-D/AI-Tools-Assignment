"""
Task 1: Classical ML with Scikit-learn
Dataset: Iris Species Dataset
Goal: Train a decision tree classifier to predict iris species
Author: [Kipruto Andrew Kipngetich]
Date: October 2025
"""

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report, 
    confusion_matrix,
    ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("TASK 1: IRIS SPECIES CLASSIFICATION WITH SCIKIT-LEARN")
print("="*70)

# ============================================================================
# STEP 1: LOAD AND EXPLORE THE DATASET
# ============================================================================
print("\n[STEP 1] Loading Iris Dataset...")

# Load the iris dataset from scikit-learn
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Target: species (0=setosa, 1=versicolor, 2=virginica)

# Create a pandas DataFrame for better visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(y, iris.target_names)

print(f"\n✓ Dataset loaded successfully!")
print(f"  - Samples: {len(df)}")
print(f"  - Features: {len(iris.feature_names)}")
print(f"  - Classes: {len(iris.target_names)}")

print(f"\nDataset Preview:")
print(df.head(10))

print(f"\nDataset Statistics:")
print(df.describe())

print(f"\nClass Distribution:")
print(df['species'].value_counts())

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================
print("\n[STEP 2] Preprocessing Data...")

# Check for missing values
missing_values = df.isnull().sum()
print(f"\nMissing Values:")
print(missing_values)

if missing_values.sum() == 0:
    print("✓ No missing values found!")
else:
    # Handle missing values (if any) - not needed for Iris dataset
    print("⚠ Missing values detected. Handling them...")
    df = df.fillna(df.mean())

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"✓ Removed {duplicates} duplicate rows")

# Encode labels (already done in Iris dataset, but showing the process)
# For Iris: 0=setosa, 1=versicolor, 2=virginica
print(f"\nLabel Encoding:")
for i, name in enumerate(iris.target_names):
    print(f"  {name}: {i}")

# Feature scaling (optional for decision trees, but good practice)
# Decision trees don't require scaling, but we'll show it for completeness
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n✓ Data preprocessing completed!")

# ============================================================================
# STEP 3: SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================================
print("\n[STEP 3] Splitting Data...")

# Split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,  # Using unscaled data (decision trees don't need scaling)
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Ensures proportional class distribution
)

print(f"\n✓ Data split completed!")
print(f"  - Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"  - Testing samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
print(f"\nTraining set class distribution:")
print(pd.Series(y_train).value_counts().sort_index())
print(f"\nTesting set class distribution:")
print(pd.Series(y_test).value_counts().sort_index())

# ============================================================================
# STEP 4: TRAIN DECISION TREE CLASSIFIER
# ============================================================================
print("\n[STEP 4] Training Decision Tree Classifier...")

# Initialize the Decision Tree Classifier
# max_depth=3 prevents overfitting on this small dataset
# random_state=42 ensures reproducibility
dt_classifier = DecisionTreeClassifier(
    max_depth=3,
    random_state=42,
    criterion='gini'  # Can also use 'entropy'
)

# Train the model
dt_classifier.fit(X_train, y_train)

print("\n✓ Model training completed!")
print(f"\nModel Parameters:")
print(f"  - Max Depth: {dt_classifier.max_depth}")
print(f"  - Criterion: {dt_classifier.criterion}")
print(f"  - Number of Leaves: {dt_classifier.get_n_leaves()}")
print(f"  - Tree Depth: {dt_classifier.get_depth()}")

# ============================================================================
# STEP 5: MAKE PREDICTIONS
# ============================================================================
print("\n[STEP 5] Making Predictions...")

# Predict on training set
y_train_pred = dt_classifier.predict(X_train)

# Predict on testing set
y_test_pred = dt_classifier.predict(X_test)

print("\n✓ Predictions completed!")

# ============================================================================
# STEP 6: EVALUATE MODEL PERFORMANCE
# ============================================================================
print("\n[STEP 6] Evaluating Model Performance...")

# Calculate metrics for training set
train_accuracy = accuracy_score(y_train, y_train_pred)

# Calculate metrics for testing set
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

print("\n" + "="*70)
print("MODEL PERFORMANCE METRICS")
print("="*70)

print(f"\nTraining Set:")
print(f"  - Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")

print(f"\nTesting Set:")
print(f"  - Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  - Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
print(f"  - Recall:    {test_recall:.4f} ({test_recall*100:.2f}%)")
print(f"  - F1-Score:  {test_f1:.4f} ({test_f1*100:.2f}%)")

# Detailed classification report
print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=iris.target_names))

# Cross-validation score
cv_scores = cross_val_score(dt_classifier, X, y, cv=5)
print(f"\n5-Fold Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================================
# STEP 7: VISUALIZE RESULTS
# ============================================================================
print("\n[STEP 7] Creating Visualizations...")

# Create a figure with multiple subplots
fig = plt.figure(figsize=(20, 12))

# 1. Confusion Matrix
ax1 = plt.subplot(2, 3, 1)
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(ax=ax1, cmap='Blues', values_format='d')
ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

# 2. Feature Importance
ax2 = plt.subplot(2, 3, 2)
feature_importance = dt_classifier.feature_importances_
features = iris.feature_names
indices = np.argsort(feature_importance)[::-1]
ax2.barh(range(len(indices)), feature_importance[indices], color='skyblue')
ax2.set_yticks(range(len(indices)))
ax2.set_yticklabels([features[i] for i in indices])
ax2.set_xlabel('Importance Score', fontsize=12)
ax2.set_title('Feature Importance', fontsize=14, fontweight='bold')
ax2.invert_yaxis()

# 3. Decision Tree Visualization
ax3 = plt.subplot(2, 3, 3)
plot_tree(
    dt_classifier, 
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    ax=ax3,
    fontsize=8
)
ax3.set_title('Decision Tree Structure', fontsize=14, fontweight='bold')

# 4. Class Distribution
ax4 = plt.subplot(2, 3, 4)
species_counts = df['species'].value_counts()
ax4.pie(species_counts, labels=species_counts.index, autopct='%1.1f%%', 
        startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
ax4.set_title('Class Distribution', fontsize=14, fontweight='bold')

# 5. Pairplot-style scatter (most important features)
ax5 = plt.subplot(2, 3, 5)
for i, species in enumerate(iris.target_names):
    mask = y == i
    ax5.scatter(X[mask, 2], X[mask, 3], label=species, alpha=0.7, s=100)
ax5.set_xlabel('Petal Length (cm)', fontsize=12)
ax5.set_ylabel('Petal Width (cm)', fontsize=12)
ax5.set_title('Feature Space: Petal Dimensions', fontsize=14, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Model Performance Comparison
ax6 = plt.subplot(2, 3, 6)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
scores = [test_accuracy, test_precision, test_recall, test_f1]
bars = ax6.bar(metrics, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
ax6.set_ylim([0, 1.1])
ax6.set_ylabel('Score', fontsize=12)
ax6.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
ax6.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect Score')
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax6.legend()

plt.tight_layout()
plt.savefig('task1_iris_results.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved as 'task1_iris_results.png'")
plt.show()

# ============================================================================
# STEP 8: SAMPLE PREDICTIONS
# ============================================================================
print("\n[STEP 8] Sample Predictions on Test Set...")

print("\n" + "="*70)
print("SAMPLE PREDICTIONS")
print("="*70)

# Show first 10 predictions
sample_size = min(10, len(X_test))
print(f"\nShowing {sample_size} sample predictions:\n")
print(f"{'Index':<8} {'True Label':<15} {'Predicted':<15} {'Correct?':<10}")
print("-" * 70)

for i in range(sample_size):
    true_label = iris.target_names[y_test[i]]
    pred_label = iris.target_names[y_test_pred[i]]
    is_correct = "✓ Yes" if y_test[i] == y_test_pred[i] else "✗ No"
    print(f"{i:<8} {true_label:<15} {pred_label:<15} {is_correct:<10}")

# ============================================================================
# STEP 9: SAVE MODEL (OPTIONAL)
# ============================================================================
print("\n[STEP 9] Saving Model...")

import joblib

# Save the trained model
model_filename = 'iris_decision_tree_model.pkl'
joblib.dump(dt_classifier, model_filename)
print(f"✓ Model saved as '{model_filename}'")

# Save the scaler (if used)
scaler_filename = 'iris_scaler.pkl'
joblib.dump(scaler, scaler_filename)
print(f"✓ Scaler saved as '{scaler_filename}'")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("TASK 1 COMPLETION SUMMARY")
print("="*70)

print("\n✓ All steps completed successfully!")
print(f"\nKey Results:")
print(f"  - Model Type: Decision Tree Classifier")
print(f"  - Dataset: Iris (150 samples, 4 features, 3 classes)")
print(f"  - Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  - Test Precision: {test_precision*100:.2f}%")
print(f"  - Test Recall: {test_recall*100:.2f}%")
print(f"  - Most Important Feature: {features[np.argmax(feature_importance)]}")

print(f"\nFiles Generated:")
print(f"  1. task1_iris_results.png (visualizations)")
print(f"  2. {model_filename} (trained model)")
print(f"  3. {scaler_filename} (data scaler)")

print("\n" + "="*70)
print("Task 1 execution complete! 🎉")
print("="*70)

# ============================================================================
# ADDITIONAL: HOW TO USE THE SAVED MODEL
# ============================================================================
print("\n" + "="*70)
print("BONUS: HOW TO USE THE SAVED MODEL FOR NEW PREDICTIONS")
print("="*70)

print("\nExample code to load and use the model:")
print("""
import joblib
import numpy as np

# Load the saved model
model = joblib.load('iris_decision_tree_model.pkl')

# Example: New iris flower measurements
# [sepal_length, sepal_width, petal_length, petal_width]
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])  # Likely setosa

# Make prediction
prediction = model.predict(new_flower)
species_names = ['setosa', 'versicolor', 'virginica']
predicted_species = species_names[prediction[0]]

print(f"Predicted species: {predicted_species}")

# Get prediction probabilities
probabilities = model.predict_proba(new_flower)
for species, prob in zip(species_names, probabilities[0]):
    print(f"{species}: {prob*100:.2f}%")
""")

