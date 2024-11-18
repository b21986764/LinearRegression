import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import matplotlib.pyplot as plt

# Load California Housing dataset
data = fetch_california_housing(as_frame=True)
features = data.data
target = data.target

# Add polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
features_poly = poly.fit_transform(features)

# Normalize the polynomial features
scaler = StandardScaler()
features_poly = scaler.fit_transform(features_poly)

# Add a bias term to the features
features_poly = np.hstack((np.ones((features_poly.shape[0], 1)), features_poly))

# Convert to numpy arrays
X = features_poly
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


# Define the Adam optimizer
def adam_optimizer(X, y, weights, learning_rate, beta1, beta2, epsilon, epochs):
    m = len(y)
    cost_history = []

    # Initialize Adam parameters
    m_t = np.zeros_like(weights)  # First moment vector
    v_t = np.zeros_like(weights)  # Second moment vector
    t = 0  # Time step

    for i in range(epochs):
        t += 1
        predictions = np.dot(X, weights)
        gradients = (1 / m) * np.dot(X.T, (predictions - y))

        # Update biased first and second moment estimates
        m_t = beta1 * m_t + (1 - beta1) * gradients
        v_t = beta2 * v_t + (1 - beta2) * (gradients ** 2)

        # Correct bias in moments
        m_t_hat = m_t / (1 - beta1 ** t)
        v_t_hat = v_t / (1 - beta2 ** t)

        # Update weights
        weights -= learning_rate * m_t_hat / (np.sqrt(v_t_hat) + epsilon)

        # Compute cost
        cost = compute_cost(X, y, weights)
        cost_history.append(cost)

        if i % 100 == 0:  # Print cost every 100 iterations
            print(f"Epoch {i}, Cost: {cost:.4f}")

    return weights, cost_history


# Adam optimizer hyperparameters
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
epochs = 50000

weights = initialize_weights(X_train.shape[1])

print("Training the polynomial model with Adam optimizer...")
weights, cost_history = adam_optimizer(X_train, y_train, weights, learning_rate, beta1, beta2, epsilon, epochs)

# Plot the cost function over epochs
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost Function Over Epochs (Polynomial Regression with Adam)')
plt.show()


# Evaluate the model
def predict(X, weights):
    return np.dot(X, weights)


predictions = predict(X_test, weights)
mse = np.mean((predictions - y_test) ** 2)
print(f"Mean Squared Error on Test Set: {mse:.4f}")
