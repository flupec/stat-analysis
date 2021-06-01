import matplotlib.pyplot as plt
import statistics as stats
import numpy as np
from scipy.stats import t as student
import scipy.stats

# Variant 21

def readDataset(filepath: str)->list:
    qWave, rWave = [], []
    with open(filepath, "r") as f:
        zeros = []
        while True:
            line = f.readline()
            # print(line)
            if not line:
                break
            splitted = line.split(",")
            if np.isclose(float(splitted[162]), 0.0):
                print(splitted[162])
            qWave.append(np.float(splitted[161]))
            rWave.append(np.float(splitted[162]))
        print(len(zeros))
    
    return qWave, rWave

if __name__ == "__main__":
    pass