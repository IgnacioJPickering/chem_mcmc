import chem_mcmc
import numpy as np
import matplotlib.pyplot as plt
from chem_mcmc.staging import Propagator, ParticleGroup
from chem_mcmc.potentials import LogGaussianOpt, LogGaussian
from chem_mcmc.utils import get_running_mean, get_running_std, PDensity
from chem_mcmc import potentials
from chem_mcmc import plotting
from chem_mcmc import constants
import time
# EXAMPLE 1 do this once
temperature = 600
lower = 0.
upper = 12.
# ~ 80k is converged (10 s) , ~ 800k is very well converged
mcmc_steps = 10000
#potential = LogGaussian(A=[2., 0.8], mu=[4, 8])
potential = LogGaussian(A=[2., 0.8], mu=[4, 8])
# Loggaussianopt doesn't work properly it outputs a veeery small trajectory 
# for some reason, check it !
p_group = ParticleGroup.random_square(number=1, lower=lower, upper=upper, dimension=1, kind='r')
p_group.attach_external_potential(potential)
propagator = Propagator(p_group, termo_properties=['potential_energy', 'trajectory'])
propagator.propagate_mcmc_nvt(steps=mcmc_steps, temperature=temperature, max_delta=1.0)
propagator.dump_to_xyz('test.xyz')

