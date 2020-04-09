import chem_mcmc
import numpy as np
import matplotlib.pyplot as plt
from chem_mcmc.staging import Propagator, ParticleGroup
from chem_mcmc.potentials import LennardJonesOpt

from chem_mcmc.utils import get_running_mean, get_running_std, PDensity
from chem_mcmc import potentials
from chem_mcmc import plotting
from chem_mcmc import constants
import time
import pickle
temperatures, acceptances = pickle.load(open('temperatures_acceptances.pkl', 'rb'))
print(temperatures)
print(acceptances)
fig, ax = plt.subplots()
ax.scatter(temperatures, acceptances)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("MCMC acceptance (%)")
plt.show()
exit()
# Simulation setup #
acceptances = []
temperatures = np.linspace(0.1, 1000, 20)
for temperature in np.linspace(0.1, 1000, 20):
    #temperature = 1000
    lower = 0.
    upper = 10.
    mcmc_steps = 10000
    minimization_steps = 50
    burn_in_steps = minimization_steps
    particle_number = 10
    max_delta = 1.0
    cutoff = 0.98 * (upper - lower)/2 
    
    potential = LennardJonesOpt(sigma=1., epsilon=3.0, cutoff=cutoff)
    p_group = ParticleGroup.random_square(number=particle_number, lower=lower, upper=upper, dimension=2, kind='p', use_cpp = False)
    p_group.attach_pairwise_potential(potential)
    propagator = Propagator(p_group, termo_properties=['potential_energy', 'trajectory'])
    #####
    
    propagator.minimize(minimization_steps)
    propagator.propagate_mcmc_nvt_onep(steps=mcmc_steps, temperature=temperature, max_delta=max_delta)
    print(propagator.last_run_time)
    propagator.burn_in(burn_in_steps)
    #propagator.dump_to_xyz('test_lj.xyz')
    acceptances.append(propagator.get_acceptance_percentage())
    print(f"Acceptance is {propagator.get_acceptance_percentage()}")
    propagator.clear_properties()
pickle.dump((temperatures, acceptances), open('temperatures_acceptances.pkl', 'wb'))

#propagator.minimize(50)
#propagator.propagate_mcmc_nvt_onep(steps=mcmc_steps, temperature=temperature, max_delta=1.0)
#print(propagator.last_run_time)
#propagator.burn_in(50)
#propagator.dump_to_xyz('test_pbc_noopt.xyz')
#propagator.clear_properties()




