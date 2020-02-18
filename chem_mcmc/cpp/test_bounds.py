from staging import Bounds
import numpy as np

#test = Test('name', [1.])
cpp_bounds = Bounds("p",[0., 0.], [10., 10.])
coordinates2 = np.asarray([1.1, 9.])
coordinates1 = np.asarray([1., 0.1])
print(coordinates1 - coordinates2)
print(np.linalg.norm(coordinates1-coordinates2))
print(cpp_bounds.areInBounds(coordinates1))
print(cpp_bounds.areInBounds(coordinates2))
d = cpp_bounds.getWrappedDistance(coordinates1, coordinates2)
print(d)
