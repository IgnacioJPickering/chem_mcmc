import potentials_cpp
import math
from chem_mcmc import potentials

#double sigma, double center, double height){
print(potentials_cpp.log_gaussian(30.,[1., 3.], [1., 3], [1., 4]))
lg = potentials.LogGaussian([1., 3.], [1., 3], [1., 4])
print(lg(30.))


print(potentials_cpp.gaussian(1., 3., 0., 3.))
g = potentials.Gaussian( 3., 3., 0.)
print(g(1.))

print(potentials_cpp.lennard_jones(1., 3., 1., 0.0, 1e-10, math.inf))
g = potentials.LennardJonesOpt(epsilon=3.,sigma=1.)
f = potentials.LennardJones(epsilon=3.,sigma=1.)
print(g(1.), f(1.))

print(potentials_cpp.lennard_jones(1., 3., 1., 0.0, 1e-10, 10.))
g = potentials.LennardJonesOpt(epsilon=3.,sigma=1., cutoff=10.)
f = potentials.LennardJones(epsilon=3.,sigma=1., cutoff=10.)
print(g(1.), f(1.))
print(g(11.), f(11.))
