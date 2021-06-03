import matplotlib.pyplot as plt
import statistics as stats
import numpy as np
import numpy.linalg
from scipy.stats import t as student
import scipy.stats

import sklearn

from sklearn.neighbors import KNeighborsClassifier as KNN

# Variant 21
mean1 = np.array([5, 3, 5])
mean2 = np.array([1, 7, 1])

R1 = np.array([
    [2, 1, 0.2],
    [1, 4, 1],
    [0.2, 1, 2]
])
R2 = np.array([
    [2, 1, 1],
    [1, 2, 1.4],
    [1, 1.4, 4]
])

n1, n2 = 100, 50

colors = {0: "r", 1: "g"}
markers = {"train": "^", "test": "o"}

def generate3DGaussianVectors(mean, R, amount):
    A = np.linalg.cholesky(R)
    x1 = stats.NormalDist(0, 1).samples(amount)
    x2 = stats.NormalDist(0, 1).samples(amount)
    x3 = stats.NormalDist(0, 1).samples(amount)
    X = np.zeros((amount, 3))
    X[:, 0] = x1
    X[:, 1] = x2
    X[:, 2] = x3
    print(X.shape, A.shape)
    return np.dot(X, A) + mean

def readDataset(filepath: str):
    qWave, rWave, sWave, sex = [], [], [], []
    with open(filepath, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            splitted = line.split(",")
            qWave.append(float(splitted[160]))
            rWave.append(float(splitted[161]))
            sWave.append(float(splitted[162]))
            sex.append(float(splitted[1]))
    return qWave, rWave, sWave, sex

def draw(X, Y, markKey, show, ax):
    for i in range(len(X)):
        x = X[i]
        c = colors[Y[i]]
        ax.scatter(x[0], x[1], x[2], color=c, s=5, marker=markers[markKey])
    if show:
        plt.show()

def getError(Ytest, predicted):
    return 1 - sklearn.metrics.accuracy_score(Ytest, predicted)

def setLabels(ax):
    ax.set_xlabel("qwave")
    ax.set_ylabel("rwave")
    ax.set_zlabel("swave")

if __name__ == "__main__":
    print("==========SYNTHETIC DATASET==========")
    x1 = generate3DGaussianVectors(mean1, R1, n1)
    y1 = np.array([0 for _ in range(len(x1))])
    x2 = generate3DGaussianVectors(mean2, R2, n2)
    y2 = np.array([1 for _ in range(len(x2))])
    
    X = np.concatenate((x1, x2), axis=0)
    Y = np.concatenate((y1, y2), axis=0)
    
    Xtrain, Xtest, Ytrain, Ytest = sklearn.model_selection.train_test_split(X, Y, test_size=0.5)

    knn = KNN(3)
    knn.fit(Xtrain, Ytrain)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    print("Just watch the distribution")
    ax.set_title("Distribution of train and test")
    setLabels(ax)
    draw(Xtrain, Ytrain, "train", False, ax)
    draw(Xtest, Ytest, "test", True, ax)
    
    predicted = knn.predict(Xtest)
    print("Error", getError(Ytest, predicted))
    print("Not lets compare predicted and real")
    
    # На графиках сравнения мы не видим обучающую выборку
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection="3d") # Количество строк тзображений, кол-во столбцов изображений, индекс в этом порядке
    ax.title.set_text("Test data")
    setLabels(ax)
    draw(Xtest, Ytest, "test", False, ax)
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.title.set_text("Predicted data")
    setLabels(ax)
    draw(Xtest, predicted, "test", True, ax)

    print("==========REAL DATASET==========")
    realX, realY, realZ, sex = readDataset("arrhythmia.data")

    X = np.zeros((len(realX), 3))
    X[:, 0] = np.array(realX)
    X[:, 1] = np.array(realY)
    X[:, 2] = np.array(realZ)
    Y = np.array(sex)

    Xtrain, Xtest, Ytrain, Ytest = sklearn.model_selection.train_test_split(X, Y, test_size=0.5)
    knn = KNN(3)
    knn.fit(Xtrain, Ytrain)
    
    print("Just watch the distribution")
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_title("Distribution of train and test")
    setLabels(ax)
    draw(Xtrain, Ytrain, "train", False, ax)
    draw(Xtest, Ytest, "test", True, ax)
    
    predicted = knn.predict(Xtest)
    print("Error", getError(Ytest, predicted))
    print("Not lets compare predicted and real")
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.set_title("Test data")
    setLabels(ax)
    draw(Xtest, Ytest, "test", False, ax)
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.set_title("Predicted")
    setLabels(ax)
    draw(Xtest, predicted, "test", True, ax)