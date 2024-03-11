import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def generate_training_data(num_points=20, seed=0):
    np.random.seed(seed)
    training_data = np.random.randint(1, 11, size=(num_points, 2))
    labels = np.random.randint(2, size=num_points)
    return training_data, labels

def generate_test_data(start=0, stop=10.1, step=0.1):
    x_values = np.arange(start, stop, step)
    y_values = np.arange(start, stop, step)
    test_data = np.array([[x, y] for x in x_values for y in y_values])
    return test_data

def fit_predict_and_plot(training_data, labels, test_data, k_values):
    for k in k_values:
        # Initialize the kNN classifier
        knn = KNeighborsClassifier(n_neighbors=k)

        # Fit the classifier to the training data
        knn.fit(training_data, labels)

        # Predict the classes for the test data
        predicted_classes = knn.predict(test_data)

        # Separate test data based on predicted classes
        class0 = test_data[predicted_classes == 0]
        class1 = test_data[predicted_classes == 1]

        # Plot the test data with predicted classes
        plt.figure()
        plt.scatter(class0[:, 0], class0[:, 1], color='blue', label='Class 0')
        plt.scatter(class1[:, 0], class1[:, 1], color='red', label='Class 1')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Scatter Plot of Test Data with Predicted Classes (k={k})')
        plt.legend()
        plt.show()

# Generate random training data
training_data, labels = generate_training_data()

# Generate test set data
test_data = generate_test_data()

# Different values of k to try
k_values = [1, 3, 5, 7, 9]

# Fit, predict, and plot for each value of k
fit_predict_and_plot(training_data, labels, test_data, k_values)
