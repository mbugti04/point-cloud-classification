import matplotlib.pyplot as plt
import numpy as np

# Generate random points
num_points = 100
x = np.random.rand(num_points)
y = np.random.rand(num_points)

# Plot the points
plt.scatter(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Random Points')
plt.show()