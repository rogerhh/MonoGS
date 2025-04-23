import matplotlib
matplotlib.use("TkAgg")  # Use a non-interactive backend

import matplotlib.pyplot as plt
import numpy as np

# Plot a simple sine wave
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title("Sine Wave")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid()
plt.show()
