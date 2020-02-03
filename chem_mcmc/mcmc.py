r"""MCMC examples"""
import numpy as np
import matplotlib.pyplot as plt
import potentials
import constants
import staging
import utils
import plotting

##### EXAMPLE 1 ######
lower = 0.0
upper = 10.0
particle_group = ParticleGroup.random_square(number=1,
                                  lower=0.0, upper=10.0, dimension=1, kind='r')
potential = Cuadratic(1, r0=5)
particle_group.attach_external_potential(potential)
propagator = Propagator(particle_group,
                        termo_properties=[
           'total_energy', 'trajectory'])
steps_min = 100
propagator.minimize(steps_min, alpha=.1)
minimization_position = propagator.get_trajectory()[:,0,0]
propagator.clear_properties()
plot_minimization(minimization_position, lower, upper)
temperature = 200 #K
steps_mcmc = 2000
max_delta = 1.0
propagator.propagate_mcmc_nvt(steps=steps_mcmc,
                              temperature=temperature, max_delta=max_delta)
position = propagator.get_trajectory()[:,0,0]
tot_traj = propagator.get_total_energy()
mean_tot = tot_traj.mean()
plot_energy(tot_traj, mean_tot)
pdensity = PDensity(temperature, potential, lower, upper)
plot_analysis(position, pdensity, lower, upper, 
              title='Analysis of a cuadratic potential')

##### EXAMPLE 2 ######
lower = 0.0
upper = 10.0
particle_group = ParticleGroup.random_square(number=1,
                                  lower=0.0, upper=10.0, dimension=1, kind='r')
epsilon = 0.2375 #kcal/mol
sigma = 3.4 #angs
potential = LennardJones(parameter1=epsilon, parameter2=sigma)
particle_group.attach_external_potential(potential)
propagator = Propagator(particle_group,
                        termo_properties=[
          'total_energy', 'trajectory'])
steps_min = 100
propagator.minimize(steps_min, alpha=.1)
minimization_position = propagator.get_trajectory()[:,0,0]
propagator.clear_properties()
plot_minimization(minimization_position, lower, upper)
temperature = 200 #K
steps_mcmc = 2000
max_delta = 1.0
propagator.propagate_mcmc_nvt(steps=steps_mcmc,
                              temperature=temperature, max_delta=max_delta)
position = propagator.get_trajectory()[:,0,0]
tot_traj = propagator.get_total_energy()
mean_tot = tot_traj.mean()
plot_energy(tot_traj, mean_tot)
pdensity = PDensity(temperature, potential, lower, upper)
plot_analysis(position, pdensity, lower, upper, 
              title='Analysis of a LJ potential')

##### EXAMPLE 3 ######
lower = 0.0
upper = 10.0
particle_group = ParticleGroup.random_square(number=1,
                                  lower=0.0, upper=10.0, dimension=1, kind='r')
epsilon = 0.2375 #kcal/mol
sigma = 3.4*0.33 #angstroems
potential = LennardJones(parameter1=epsilon, parameter2=sigma)
particle_group.attach_external_potential(potential)
propagator = Propagator(particle_group, 
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
  pdensity = PDensity(temperature, potential, lower, upper)
  if j % 8 == 0:
    plot_analysis(position, pdensity, lower, 
                  upper, title=f'Temperature = {temperature} K')
  position_means.append(position.mean())
  position_stds.append(position.std())
  position_means_teo.append(pdensity.get_mean_distance())
  position_stds_teo.append(pdensity.get_std_distance())
  acceptances.append(propagator.get_acceptance_percentage())
  propagator.clear_properties()
fig, ax = plt.subplots(1,3, figsize=(27,9))
ax[0].scatter(temperatures, position_means,
              label='Mean x, hist',color='r', s=5.0)
ax[0].plot(temperatures, position_means_teo,
           label='Mean x, teo', color='r', linewidth=1.0)
ax[0].set_xlabel('Temperature (K)')
ax[0].set_ylabel(r'Mean Position ($\AA$)')
ax[1].scatter(temperatures, position_stds,
              label='Std. x, hist', color='orange', s=5.0)
ax[1].plot(temperatures, position_stds_teo,
           label='Std. x, teo', color='orange', linewidth=1.0)
ax[1].set_xlabel('Temperature (K)')
ax[1].set_ylabel(r'Standard dev. Position ($\AA$)')
ax[2].scatter(temperatures, acceptances,color='green', s=5.0)
ax[2].set_xlabel('Temperature (K)')
ax[2].set_ylabel(r'Acceptance percentage (%)')
ax[0].legend(loc='lower right')
ax[1].legend(loc='lower right')
plt.show()

##### EXAMPLE 4 ######
lower = 0.0
upper = 10.0
particle_group = ParticleGroup.random_square(number=1,
                                  lower=0.0, upper=10.0, dimension=1, kind='r')
propagator = Propagator(particle_group, 
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
  potential = LennardJones(parameter1=epsilon, parameter2=sigma)
  particle_group.attach_external_potential(potential)
  propagator.propagate_mcmc_nvt(steps=steps_mcmc,
                                temperature=temperature, max_delta=max_delta)
  position = propagator.get_trajectory()[:,0,0]
  pdensity = PDensity(temperature, potential, lower, upper)
  if j % 4 == 0:
    plot_analysis(position, pdensity, lower, 
                  upper, 'Sigma = 'f'{sigma:.4f}'r' $\AA$')
  position_means.append(position.mean())
  position_stds.append(position.std())
  position_means_teo.append(pdensity.get_mean_distance())
  position_stds_teo.append(pdensity.get_std_distance())
  acceptances.append(propagator.get_acceptance_percentage())
  propagator.clear_properties()

# plotting code
fig, ax = plt.subplots(1,3, figsize=(27,9))
ax[0].scatter(sigmas, position_means,
              label='Mean x, hist',color='r', s=5.0)
ax[0].plot(sigmas, position_means_teo,
           label='Mean x, teo', color='r', linewidth=1.0)
ax[0].set_xlabel(r'Sigma ($\AA$)')
ax[0].set_ylabel(r'Mean Position ($\AA$)')
ax[1].scatter(sigmas, position_stds,
              label='Std. x, hist', color='orange', s=5.0)
ax[1].plot(sigmas, position_stds_teo,
           label='Std. x, teo', color='orange', linewidth=1.0)
ax[1].set_xlabel(r'Sigma ($\AA$)')
ax[1].set_ylabel(r'Standard dev. Position ($\AA$)')
ax[2].scatter(sigmas, acceptances,color='green', s=5.0)
ax[2].set_xlabel(r'Sigma ($\AA$)')
ax[2].set_ylabel(r'Acceptance percentage (%)')
ax[0].legend(loc='lower right')
ax[1].legend(loc='lower right')
plt.show()


