import numpy as np
grid = np.arange(9).reshape([3,3])
iterator = np.nditer(grid, flags=['multi_index'])
while not iterator.finished:
    s = iterator.iterindex
    print(s)
print(iterator.multi_index)
print(iterator.iterindex)
print(iterator.iternext())
