import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
data = pd.read_csv("C:\\Users\\Tarun\\OneDrive\\Desktop\\New folder\\drowsiness_data_.csv")

# Split the dataset into features (X) and labels (y)
X = data[["EAR", "MAR"]].values
y = data["Label"].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (optional, but recommended)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the Keras model
model = Sequential()
model.add(Dense(32, input_dim=2, activation="relu"))  # 2 inputs: EAR and MAR
model.add(Dense(16, activation="relu"))
model.add(Dense(1, activation="sigmoid"))  # Binary classification: Drowsy (1) or Not Drowsy (0)

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
model.save("drowsiness_detection_model1.keras")