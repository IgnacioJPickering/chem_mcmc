r"""MCMC examples"""
import numpy as np
import matplotlib.pyplot as plt
from chem_mcmc import potentials
from chem_mcmc import constants
from chem_mcmc import staging
from chem_mcmc import utils
from chem_mcmc import plotting
##### EXAMPLE 1 ######
lower = 0.0
upper = 10.0
particle_group = staging.ParticleGroup.random_square(number=1,
                                  lower=lower, upper=upper, dimension=1, kind='r')
potential = potentials.Cuadratic(1, r0=5)
particle_group.attach_external_potential(potential)
propagator = staging.Propagator(particle_group,
                        termo_properties=[
           'total_energy', 'trajectory'])
steps_min = 100
propagator.minimize(steps_min, alpha=.1)
minimization_position = propagator.get_trajectory()[:,0,0]
propagator.clear_properties()
plotting.plot_minimization(minimization_position, lower, upper)
temperature = 200 #K
steps_mcmc = 2000
max_delta = 1.0
propagator.propagate_mcmc_nvt(steps=steps_mcmc,
                              temperature=temperature, max_delta=max_delta)
position = propagator.get_trajectory()[:,0,0]
tot_traj = propagator.get_total_energy()
mean_tot = tot_traj.mean()
plotting.plot_energy(tot_traj, mean_tot)
pdensity = utils.PDensity(temperature, potential, lower, upper)
plotting.plot_analysis(position, pdensity, lower, upper, potential, 
              title='Analysis of a cuadratic potential')

##### EXAMPLE 2 ######
lower = 0.0
upper = 10.0
particle_group = staging.ParticleGroup.random_square(number=1,
                                  lower=lower, upper=upper, dimension=1, kind='r')
epsilon = 0.2375 #kcal/mol
sigma = 3.4 #angs
potential = potentials.LennardJones(parameter1=epsilon, parameter2=sigma)
particle_group.attach_external_potential(potential)
propagator = staging.Propagator(particle_group,
                        termo_properties=[
          'total_energy', 'trajectory'])
steps_min = 100
propagator.minimize(steps_min, alpha=.1)
minimization_position = propagator.get_trajectory()[:,0,0]
propagator.clear_properties()
plotting.plot_minimization(minimization_position, lower, upper)
temperature = 200 #K
steps_mcmc = 2000
max_delta = 1.0
propagator.propagate_mcmc_nvt(steps=steps_mcmc,
                              temperature=temperature, max_delta=max_delta)
position = propagator.get_trajectory()[:,0,0]
tot_traj = propagator.get_total_energy()
mean_tot = tot_traj.mean()
plotting.plot_energy(tot_traj, mean_tot)
pdensity = utils.PDensity(temperature, potential, lower, upper)
plotting.plot_analysis(position, pdensity, lower, upper, potential, 
              title='Analysis of a LJ potential')
##### EXAMPLE 3 ######
lower = 0.0
upper = 10.0
particle_group = staging.ParticleGroup.random_square(number=1,
                                  lower=lower, upper=upper, dimension=1, kind='r')
epsilon = 0.2375 #kcal/mol
sigma = 3.4*0.33 #angstroems
potential = potentials.LennardJones(parameter1=epsilon, parameter2=sigma)
particle_group.attach_external_potential(potential)
propagator = staging.Propagator(particle_group, 
                        termo_properties=['total_energy', 'trajectory'])
steps_mcmc = 20000
max_delta = 1.0
temperatures = np.arange(5, 150, 5)
position_means = []
position_means_teo = []
position_stds = []
position_stds_teo = []
acceptances = []
for j, temperature in enumerate(temperatures):
  propagator.propagate_mcmc_nvt(steps=steps_mcmc,
                                temperature=temperature, max_delta=max_delta)
  position = propagator.get_trajectory()[:,0,0]
  pdensity = utils.PDensity(temperature, potential, lower, upper)
  if j % 8 == 0:
    plotting.plot_analysis(position, pdensity, lower, upper, potential, 
                  title=f'Temperature = {temperature} K')
  position_means.append(position.mean())
  position_stds.append(position.std())
  position_means_teo.append(pdensity.get_mean_distance())
  position_stds_teo.append(pdensity.get_std_distance())
  acceptances.append(propagator.get_acceptance_percentage())
  propagator.clear_properties()

titles = (r'Mean $x$ ($\AA$)', r'Std. dev. $x$ ($\AA$)', r'Acc. percentage (%)', r'Temperature (K)')
plotting.plot_means_stds_accs(temperatures, position_means, position_means_teo, 
        position_stds, position_stds_teo, acceptances, titles)
plt.show()

##### EXAMPLE 4 ######
lower = 0.0
upper = 10.0
particle_group = staging.ParticleGroup.random_square(number=1,
                                  lower=lower, upper=upper, dimension=1, kind='r')
propagator = staging.Propagator(particle_group, 
                        termo_properties=['total_energy', 'trajectory'])
steps_mcmc = 20000
epsilon = 0.2375 #kcal/mol
max_delta = 1.0
sigmas = np.linspace(0.5,8., 10)
position_means = []
position_means_teo = []
position_stds = []
position_stds_teo = []
acceptances = []
temperature = 80 #K

for j, sigma in enumerate(sigmas):
  potential = potentials.LennardJones(parameter1=epsilon, parameter2=sigma)
  particle_group.attach_external_potential(potential)
  propagator.propagate_mcmc_nvt(steps=steps_mcmc,
                                temperature=temperature, max_delta=max_delta)
  position = propagator.get_trajectory()[:,0,0]
  pdensity = utils.PDensity(temperature, potential, lower, upper)
  if j % 4 == 0:
    plotting.plot_analysis(position, pdensity, lower,upper,  potential, 
                  'Sigma = 'f'{sigma:.4f}'r' $\AA$')
  position_means.append(position.mean())
  position_stds.append(position.std())
  position_means_teo.append(pdensity.get_mean_distance())
  position_stds_teo.append(pdensity.get_std_distance())
  acceptances.append(propagator.get_acceptance_percentage())
  propagator.clear_properties()

titles = (r'Mean $x$ ($\AA$)', r'Std. dev. $x$ ($\AA$)', r'Acc. percentage (%)', r'Sigma ($\AA$)')
plotting.plot_means_stds_accs(sigmas, position_means, position_means_teo, 
        position_stds, position_stds_teo, acceptances, titles)
plt.show()

# plotting code

#EXAMPLE 5 #####
lower = 0.0
upper = 10.0
particle_group = staging.ParticleGroup.random_square(number=1,
                                  lower=lower, upper=upper, dimension=1, kind='r')
propagator = staging.Propagator(particle_group, 
                        termo_properties=['total_energy', 'trajectory'])
steps_mcmc = 20000
epsilon = 0.2375 #kcal/mol
sigma = 3.4
potential = potentials.LennardJones(parameter1=epsilon, parameter2=sigma)
particle_group.attach_external_potential(potential)
max_delta = 1.0
position_means = []
position_means_teo = []
position_stds = []
position_stds_teo = []
acceptances = []
temperature = 80 #K

steps = np.linspace(1, 100, 10).tolist() + np.linspace(100,10000,10).tolist()
for j, s in enumerate(steps):
  s = int(s)
  propagator.propagate_mcmc_nvt(steps=s,
                                temperature=temperature, max_delta=max_delta)
  position = propagator.get_trajectory()[:,0,0]
  pdensity = utils.PDensity(temperature, potential, lower, upper)
  if j % 4 == 0:
    plotting.plot_analysis(position, pdensity, lower, upper, potential,  f'Total Steps = {s}')
  position_means.append(position.mean())
  position_stds.append(position.std())
  position_means_teo.append(pdensity.get_mean_distance())
  position_stds_teo.append(pdensity.get_std_distance())
  acceptances.append(propagator.get_acceptance_percentage())
  propagator.clear_properties()
# plotting code
fig, ax = plt.subplots(1,2, figsize=(18,9))
ax[0].scatter(steps, position_means,
              label='Mean x, hist',color='r', s=5.0)
ax[0].plot(steps, position_means_teo,
           label='Mean x, teo', color='r', linewidth=1.0)
ax[0].set_xlabel(r'Total MCMC steps')
ax[0].set_ylabel(r'Mean Position ($\AA$)')
ax[1].scatter(steps, np.abs(
    np.asarray(position_means_teo) - np.asarray(position_means)), 
    color='red', s=5.0)
ax[1].set_xlabel(r'Total MCMC steps')
ax[1].set_ylabel(r'Absolute difference with theory ($\AA$)')
plt.show()
# plotting code


