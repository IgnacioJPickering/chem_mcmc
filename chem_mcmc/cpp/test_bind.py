import potentials_cpp
from chem_mcmc import potentials

#double sigma, double center, double height){
print(potentials_cpp.log_gaussian(30.,[1., 3.], [1., 3], [1., 4]))
lg = potentials.LogGaussian([1., 3.], [1., 3], [1., 4])
print(lg(30.))


print(potentials_cpp.gaussian(1., 3., 0., 3.))
g = potentials.Gaussian( 3., 3., 0.)
print(g(1.))
