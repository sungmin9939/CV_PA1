import numpy as np

a = np.ones((1,3,3))
a[0][1][1] = 3
print(a)
b = np.ones((1,3,3))
print(a.flatten()[4])
print(a)
