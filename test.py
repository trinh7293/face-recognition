import numpy as np

a = np.array([1, 2, 3])
print('a.shape', a.shape)
np.save('test1', a)
b = np. load('test1.npy')
print('b = ', b)
print('b.shape', b.shape)
