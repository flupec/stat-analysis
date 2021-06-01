from typing import List
import matplotlib.pyplot as plt
import statistics as stats
import numpy as np
from scipy.stats import t as student
import scipy.stats

from sklearn import cluster

# Variant 21

n1 = 100
n2 = 50
a1 = [0, 1]
a2 = [3, -2]
R1 = [
    [2, 1],
    [1, 1]
]
R2 = [
    [2, -0.5],
    [-0.5, 1]
]

def generate2DGaussianVector(samplesAmount: int, mean: float, R: List[List]):
    x1 = stats.NormalDist(0, 1).samples(samplesAmount)
    x2 = stats.NormalDist(0, 1).samples(samplesAmount)
    y1 = [np.sqrt(R[0][0]) * x for x in x1]
    y2 = [R[0][1] * x1[i] / np.sqrt(R[0][0]) + np.sqrt(R[1][1] - R[0][1] ** 2 / R[0][0]) * x2[i] for i in range(len(x1))]
    
    return [y + mean[0] for y in y1], [y + mean[1] for y in y2]

def readDataset(filepath: str)->list:
    qWave, rWave = [], []
    with open(filepath, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            splitted = line.split(",")
            qWave.append(np.float(splitted[161]))
            rWave.append(np.float(splitted[162]))
    
    return qWave, rWave

if __name__ == "__main__":
    xs1, ys1 = generate2DGaussianVector(n1, a1, R1)
    xs2, ys2 = generate2DGaussianVector(n2, a2, R2)

    labels = {0: "o", 1: "^"}
    colors = {0: "r", 1: "g"}

    plt.scatter(xs1, ys1, marker=labels[0], s=10, c=colors[0])
    plt.scatter(xs2, ys2, marker=labels[1], s=10, c=colors[1])
    plt.show()
    
    X1 = [[xs1[i], ys1[i]] for i in range(n1)]
    X2 = [[xs2[i], ys2[i]] for i in range(n2)]
    X = [*X1, *X2]
    
    verbose = False
    kmeans = cluster.KMeans(n_clusters=2, verbose=1 if verbose else 0).fit(X)
    predicted = kmeans.labels_
    for i in range(len(predicted)):
        vec = X[i]
        label = predicted[i]
        plt.scatter(vec[0], vec[1], marker=labels[label], s=10, c=colors[label])
    
    plt.scatter(kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[0][1], marker="*", s=40, c="y")
    plt.scatter(kmeans.cluster_centers_[1][0], kmeans.cluster_centers_[1][1], marker="*", s=40, c="y")
    plt.show()

    print("Sum of squared distances is", kmeans.inertia_)
