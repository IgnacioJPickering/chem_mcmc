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
# EXAMPLE 1 do this once
temperature = 1000
lower = 0.
upper = 12.
# ~ 80k is converged (10 s) , ~ 800k is very well converged
mcmc_steps = 10000
#potential = LogGaussian(A=[2., 0.8], mu=[4, 8])
#potential = LogGaussian(A=[2., 0.8], mu=[4, 8])
potential = HardSpheresStep(sigma1=0.5, sigma2=1.0, epsilon=2)
# Loggaussianopt doesn't work properly it outputs a veeery small trajectory 
# for some reason, check it !
p_group = ParticleGroup.random_square(number=100, lower=lower, upper=upper, dimension=2, kind='r')
#p_group.attach_external_potential(potential)
p_group.attach_pairwise_potential(potential)
propagator = Propagator(p_group, termo_properties=['potential_energy', 'trajectory'])
propagator.minimize(50)
propagator.propagate_mcmc_nvt(steps=mcmc_steps, temperature=temperature, max_delta=0.5)
#print(propagator.get_acceptance_percentage())
propagator.dump_to_xyz('test_run.xyz')

