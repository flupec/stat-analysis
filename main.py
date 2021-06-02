import matplotlib.pyplot as plt
import statistics as stats
import numpy as np
from scipy.stats import t as student
import scipy.stats

# Variant 21
n = int(10 ** 4)
a = 10
b = 0.1
stdSqr = 1

def readDataset(filepath: str):
    qWave, rWave = [], []
    with open(filepath, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            splitted = line.split(",")
            qWave.append(float(splitted[160]))
            rWave.append(float(splitted[161]))
    
    return qWave, rWave

def predict(ys, xs):
    meanXs, meanYs = stats.fmean(xs), stats.fmean(ys)
    aNumerator = np.sum([(x - meanXs) * (y - meanYs) for x, y in zip(xs, ys)])
    aDenumerator = np.sum([(x - meanXs) ** 2 for x in xs])
    a = aNumerator / aDenumerator
    b = meanYs - a * meanXs
    
    return a, b

def getDetermFactor(ys, predictedYs)-> float:
    return 1 - np.sum((pry - y) ** 2 for pry, y in zip(predictedYs, ys)) / np.sum([(y - stats.fmean(ys)) ** 2 for y in ys])

if __name__ == "__main__":
    print("==========SYNTHETIC DATASET==========")
    dist = stats.NormalDist(mu=0, sigma=np.sqrt(stdSqr))
    eps = dist.samples(n)
    xs = [x for x in np.arange(0, 1, 1 / n)]
    ys  = [a * x + b + e for x, e in zip(xs, eps)]
    
    predictedA, predictedB = predict(ys, xs)
    predictedYs = [predictedA * x + predictedB for x in xs]
    determFactor = getDetermFactor(ys, predictedYs)
    print("Predicted: a={}, b={}, R^2={}".format(predictedA, predictedB, determFactor))

    plt.plot(xs, ys, "o", markersize=0.5)
    plt.plot(xs, predictedYs, "-")
    plt.show()

    print("==========REAL DATASET==========")
    realX, realY = readDataset("arrhythmia.data")
    predictedA, predictedB = predict(realY, realX)
    predictedYs = [predictedA * x + predictedB for x in realX]
    determFactor = getDetermFactor(realY, predictedYs)
    print("Predicted: a={}, b={}, R^2={}".format(predictedA, predictedB, determFactor))
    plt.plot(realX, realY, "o", markersize=0.5)
    plt.plot(realX, predictedYs, "-")
    plt.show()

    print("There are 162 entries with zero values at R WAVE series (y axis)")