import matplotlib.pyplot as plt
import statistics as stats
import numpy as np
from scipy.stats import t as student
import scipy.stats

# Variant 21
n = int(10 ** 4)
a = (0, 1)
R = ((4, 3), (3, 9))

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

def getCorrelation(xs, ys)->float:
    xys = [x * y for x, y in zip(xs, ys)]
    meanXs, meanYs = stats.fmean(xs), stats.fmean(ys)
    
    return stats.fmean(xys) - meanXs * meanYs

def getPearson(xs, ys) -> float:
    numerator = getCorrelation(xs, ys)
    xsSquare = [x ** 2 for x in xs]
    ysSquare = [y ** 2 for y in ys]
    meanXs, meanYs = stats.fmean(xs), stats.fmean(ys)
    denumerator = np.sqrt(stats.fmean(xsSquare) - meanXs ** 2) * np.sqrt(stats.fmean(ysSquare) - meanYs ** 2)
    
    return numerator / denumerator

def getRealPearson() -> float:
    return R[0][1] / np.sqrt(R[0][0] * R[1][1])

def generate2DGaussianVector():
    x1 = stats.NormalDist(0, 1).samples(n)
    x2 = stats.NormalDist(0, 1).samples(n)
    y1 = [np.sqrt(R[0][0]) * x for x in x1]
    y2 = [R[0][1] * x1[i] / np.sqrt(R[0][0]) + np.sqrt(R[1][1] - R[0][1] ** 2 / R[0][0]) * x2[i] for i in range(len(x1))]

    return [y + a[0] for y in y1], [y + a[1] for y in y2]

def visualize(xs, ys):
    plt.scatter(xs, ys, s=0.5)
    plt.show()

if __name__ == "__main__":
    print("===============SYNTHETIC DATASET===============")
    y = generate2DGaussianVector()
    visualize(y[0], y[1])
    estimatedPearson = getPearson(y[0], y[1])
    print("Estimated Pearson coeff={}".format(estimatedPearson))
    print("Real Pearson coeff={}".format(getRealPearson()))
    alpha = 0.05 # Уровень значимости
    tmp = (1 + (n - 2) / (student(df=n-2).ppf(1 - alpha / 2))) ** -1 # r^2 должен быть > чем эта величина
    print("Correlated if {} > {}".format(estimatedPearson ** 2, tmp))

    print("===============REAL DATASET===============")
    qWave, rWave = readDataset("arrhythmia.data")
    
    visualize(qWave, rWave)
    estimatedPearsonFromDataset = getPearson(qWave, rWave)
    datasetLen = len(qWave)
    print("Estimated Pearson coeff={}".format(estimatedPearsonFromDataset))
    tmp = (1 + (datasetLen - 2) / (student(df=datasetLen-2).ppf(1 - alpha / 2))) ** -1 # r^2 должен быть > чем эта величина
    print("Correlated if {} > {}".format(estimatedPearsonFromDataset ** 2, tmp))

    print("There are 162 entries with zero values at R WAVE series (y axis)")