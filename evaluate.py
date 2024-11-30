import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# Load the saved model
model = load_model("drowsiness_detection_model1.keras")

# Use the test set to evaluate the model
# X_test and y_test should be the test data from the training process
# (Ensure they're still available in the current session)

# Make predictions
y_pred = model.predict(X_test)

# Convert probabilities to binary predictions (threshold = 0.5)
y_pred_binary = (y_pred > 0.5).astype(int).flatten()

# Evaluate the model's performance
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_binary))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_binary))
