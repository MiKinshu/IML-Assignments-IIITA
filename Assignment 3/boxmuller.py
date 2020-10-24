import numpy as np
import math
import random
import matplotlib.pyplot as plt

U1 = np.random.uniform(low=0.0, high=1.0, size=(1000))
U2 = np.random.uniform(low=0.0, high=1.0, size=(1000))

x = []
for i in range(len(U1)):
    x.append(math.sqrt(-2 * math.log(U1[i])) * math.cos((2 * 3.14) * U2[i]))

y = []
for i in range(len(U2)):
    y.append(math.sqrt(-2 * math.log(U1[i])) * math.sin((2 * 3.14) * U2[i]))

plt.plot(x, y)
plt.show()