import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load California Housing dataset
data = fetch_california_housing(as_frame=True)
features = data.data
target = data.target

# Normalize the features
features = (features - features.mean()) / features.std()

# Add a bias term to the features
features.insert(0, 'Bias', 1)

# Convert to numpy arrays
X = features.values
y = target.values.reshape(-1, 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize parameters
def initialize_weights(n_features):
    return np.zeros((n_features, 1))


# Define the cost function
def compute_cost(X, y, weights):
    m = len(y)
    predictions = np.dot(X, weights)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost


# Define the gradient descent function
def gradient_descent(X, y, weights, learning_rate, epochs):
    m = len(y)
    cost_history = []

    for i in range(epochs):
        predictions = np.dot(X, weights)
        gradients = (1 / m) * np.dot(X.T, (predictions - y))
        weights -= learning_rate * gradients
        cost = compute_cost(X, y, weights)
        cost_history.append(cost)

        if i % 100 == 0:  # Print cost every 100 iterations
            print(f"Epoch {i}, Cost: {cost:.4f}")

    return weights, cost_history


# Train the model
learning_rate = 0.01
epochs = 1000
weights = initialize_weights(X_train.shape[1])

print("Training the model...")
weights, cost_history = gradient_descent(X_train, y_train, weights, learning_rate, epochs)

# Plot the cost function over epochs
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost Function Over Epochs')
plt.show()


# Evaluate the model
def predict(X, weights):
    return np.dot(X, weights)


predictions = predict(X_test, weights)
mse = np.mean((predictions - y_test) ** 2)
print(f"Mean Squared Error on Test Set: {mse:.4f}")
