import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\ashis\\Downloads\\archive (6)\\Data\\features_30_sec.csv")


def minkowski_distance(x, y, r):
    distance = np.sum(np.abs(x - y) ** r) ** (1 / r)
    return distance

def calculate_distances(X1, X2, r_values):
    distances = [minkowski_distance(X1, X2, r) for r in r_values]
    return distances

def plot_distances(r_values, distances):
    plt.plot(r_values, distances, marker='*')
    plt.xlabel('Value of r')
    plt.ylabel('Minkowski Distance')
    plt.title('Minkowski Distance vs. Value of r')
    plt.grid(True)
    plt.show()

# two feature vectors from dataset
X1 = data["chroma_stft_mean"]
X2 = data["rms_mean"]

# Define range of r values
r_values = range(1, 11)

# Calculate Minkowski distance for each r value
distances = calculate_distances(X1, X2, r_values)

# Plot the distances
plot_distances(r_values, distances)