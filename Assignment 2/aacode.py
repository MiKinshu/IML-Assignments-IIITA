import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

y = np.arange(16)
y = y.reshape(4, 4)
z = np.arange(4)
print(z)
print()
print(y)
print()
print(np.matmul(y, z))

def FeatureScale(y):
    return np.array(y - np.mean(y))

x = FeatureScale(y)
# print(x)