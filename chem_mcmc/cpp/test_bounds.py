from staging_cpp import Bounds
import numpy as np

#test = Test('name', [1.])
cpp_bounds = Bounds("p",[0., 0.], [10., 10.])
coordinates2 = np.asarray([1.1, 9.])
coordinates1 = np.asarray([1., 0.1])
coordinates3 = np.asarray([31., -1])
print(cpp_bounds.wrap_coordinates(coordinates3))

coordinates2 = np.asarray(cpp_bounds.wrap_coordinates(coordinates3))
print(coordinates2)
exit()
print(coordinates1 - coordinates2)
print(np.linalg.norm(coordinates1-coordinates2))
print(cpp_bounds.are_in_bounds(coordinates1))
print(cpp_bounds.are_in_bounds(coordinates2))
d = cpp_bounds.get_distance(coordinates1, coordinates2)
print(d)

cpp_bounds = Bounds.square(lower=0.0, upper=10.0, dimension=4, kind="p")
print(cpp_bounds.sizes)
cpp_bounds.kind = "p"
print(cpp_bounds.kind)
cpp_bounds.kind = "n"
print(cpp_bounds.kind)
cpp_bounds.kind = "p"
print(cpp_bounds.kind == "p")
