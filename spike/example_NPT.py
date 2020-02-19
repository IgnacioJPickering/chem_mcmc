import chem_mcmc
import numpy as np
import matplotlib.pyplot as plt
from chem_mcmc.staging import Propagator, ParticleGroup
from chem_mcmc.potentials import LogGaussianOpt, LogGaussian, Constant, Cuadratic, HardSpheres, HardSpheresStep
from chem_mcmc.utils import get_running_mean, get_running_std, PDensity
from chem_mcmc import potentials
from chem_mcmc import plotting
from chem_mcmc import constants
import time

# Simulation setup #
temperature = 10000
pressure = 2 # units should be kcal/(mol ang^2)
# an approximate idea given by the ideal gas law is that T*N * R * 4.184/upper^2
# \approx P
lower = 0.
upper = 12.
mcmc_steps = 50000
potential = HardSpheresStep(sigma1=0.5, sigma2=1.0, epsilon=2)
p_group = ParticleGroup.random_square(number=20, lower=lower, upper=upper, dimension=2, kind='p')
p_group.attach_pairwise_potential(potential)
propagator = Propagator(p_group, termo_properties=['potential_energy', 'trajectory', 'bound_sizes'])
propagator.minimize(50)
propagator.propagate_mcmc_npt(steps=mcmc_steps, temperature=temperature, pressure=pressure, max_delta_coord=1.0)
propagator.burn_in(50)
propagator.dump_to_xyz('npt_low.xyz')
bound_sizes = propagator.get_bound_sizes()
np.set_printoptions(threshold=3000)
print(bound_sizes)
#####




