from staging_cpp import Bounds
import numpy as np

#test = Test('name', [1.])
cpp_bounds = Bounds("p",[0., 0.], [10., 10.])
coordinates2 = np.asarray([1.1, 9.])
coordinates1 = np.asarray([1., 0.1])
print(coordinates1 - coordinates2)
print(np.linalg.norm(coordinates1-coordinates2))
print(cpp_bounds.are_in_bounds(coordinates1))
print(cpp_bounds.are_in_bounds(coordinates2))
d = cpp_bounds.get_distance(coordinates1, coordinates2)
print(d)

cpp_bounds = Bounds.square(lower=0.0, upper=10.0, dimension=4, kind="p")
print(cpp_bounds.sizes)
