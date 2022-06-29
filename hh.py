import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
np.random.seed(19680801)
Z = np.random.rand(5, 11)
x = np.arange(-0.5, 11, 1)  # len = 11
y = np.arange(0, 11, 1)  # len = 7

fig, ax = plt.subplots()
ax.pcolormesh(x, y, Z)

plt.show()