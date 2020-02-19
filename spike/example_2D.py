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
temperature = 1000
lower = 0.
upper = 12.
mcmc_steps = 10000
potential = HardSpheresStep(sigma1=0.5, sigma2=1.0, epsilon=2)
p_group = ParticleGroup.random_square(number=20, lower=lower, upper=upper, dimension=2, kind='p')
p_group.attach_pairwise_potential(potential)
propagator = Propagator(p_group, termo_properties=['potential_energy', 'trajectory'])
#####

propagator.minimize(50)
propagator.propagate_mcmc_nvt(steps=mcmc_steps, temperature=temperature, max_delta=1.0)
print(propagator.last_run_time)
propagator.burn_in(50)
propagator.dump_to_xyz('test_pbc_opt.xyz')
propagator.clear_properties()

propagator.minimize(50)
propagator.propagate_mcmc_nvt_onep(steps=mcmc_steps, temperature=temperature, max_delta=1.0)
print(propagator.last_run_time)
propagator.burn_in(50)
propagator.dump_to_xyz('test_pbc_noopt.xyz')
propagator.clear_properties()




