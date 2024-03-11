import numpy as np
import matplotlib.pyplot as plt

def generate_data(num_points=20, seed=0):
    np.random.seed(seed)
    X = np.random.randint(1, 11, size=(num_points, 2))
    return X

def plot_training_data(class0, class1):
    plt.figure(figsize=(8, 6))
    plt.scatter(class0[:, 0], class0[:, 1], color='blue', label='Class 0')
    plt.scatter(class1[:, 0], class1[:, 1], color='red', label='Class 1')
    plt.title('Scatter Plot of Training Data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

# Generating 20 data points with random values between 1 and 10
X = generate_data()

# Assign these points to two different classes
class0 = X[:10]
class1 = X[10:]

# Plot the training data
plot_training_data(class0, class1)
