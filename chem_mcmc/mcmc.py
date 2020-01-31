import numpy as np
import matplotlib.pyplot as plt
import potentials
import constants
import staging

lower = 0.0
upper = 10.0
particle_group = staging.ParticleGroup.random_square(number=1, lower=lower, upper=upper, dimension=1, kind='r')
epsilon = 0.2375 #kcal/mol
sigma = 3.4 #angstroems
potential = potentials.LennardJones(parameter1=epsilon, parameter2=sigma, parametrization='epsilon_sigma')
#particle_group.attach_pairwise_potential(potential)
particle_group.attach_external_potential(potential)
#prop = Propagator(particle_group, termo_properties=['forces','pressure','potential_energy', 'kinetic_energy', 'total_energy', 'trajectory'])
propagator = staging.Propagator(particle_group, termo_properties=['potential_energy', 'kinetic_energy', 'total_energy', 'trajectory'])
steps_min = 100
propagator.minimize(steps_min, alpha=.01)
mcmc_traj = prop.get_trajectory()
fig, ax = plt.subplots(1,1)
for j, _ in enumerate(particle_group):
    ax.plot(mcmc_traj[:,j,0], linewidth=0.5)
ax.set_xlabel('minimization')
ax.set_ylabel(r'Position ($\AA$)')
ax.set_ylim(lower-10, upper+10)
plt.show()
propagator.clear_properties()
temperature = 200 #K
steps_mcmc = 200000
propagator.propagate_mcmc_nvt(steps=steps_mcmc, temperature=temperature, max_delta=1.0)
mcmc_traj = propagator.get_trajectory()
pot_traj = propagator.get_potential_energy()
kin_traj = propagator.get_kinetic_energy()
tot_traj = propagator.get_total_energy()
plt.gca().set_xlabel('MCMC steps')
plt.gca().set_ylabel(r'Linear pressure ($kcal$ $mol^{-1}$ $\AA^{-1}$)')
xlo, xhi = plt.gca().get_xlim()
plt.legend(loc='upper right')
plt.show()
plt.plot(pot_traj, linewidth = 0.5, label='Potential', color='b')
plt.plot(kin_traj, linewidth = 0.5, label='Kinetic', color='r')
plt.plot(tot_traj, linewidth = 0.5, label='Total', color='g')
mean_tot = tot_traj.mean()
std_tot = tot_traj.std()
heat_capacity = std_tot/(constants.kb*temperature**2)
mean_pot = pot_traj.mean()
plt.gca().set_xlabel('MCMC steps')
plt.gca().set_ylabel(r'Energy ($kcal$ $mol^{-1}$)')
xlo, xhi = plt.gca().get_xlim()
plt.hlines(mean_pot, xmin=xlo, xmax=xhi, linestyles='dashed', colors='darkblue', label=f'Mean Potential {mean_pot:.4f}'r' ($kcal$ $mol^{-1}$)')
plt.hlines(mean_tot, xmin=xlo, xmax=xhi, linestyles='dashed', colors='darkgreen', label=f'Mean Total {mean_tot:.4f}'r' ($kcal$ $mol^{-1}$)')
plt.hlines([mean_tot+std_tot, mean_tot - std_tot], xmin=xlo, xmax=xhi, linestyles='dashed', colors='darkgreen', linewidths=[0.5, 0.5], label=f'Heat capacity {heat_capacity:.4f}'r' ($kcal$ $mol^{-1}$ $K^{-1}$)')
plt.legend(loc='upper right')
plt.show()

fig, ax = plt.subplots(1,1)
for j, _ in enumerate(particle_group):
    ax.scatter(range(len(mcmc_traj[:,j,0])),mcmc_traj[:,j,0], s=1.5)
ax.set_xlabel('MCMC steps')
ax.set_ylabel(r'Position ($\AA$)')
ax.set_ylim(lower, upper)
plt.show()
#exit()

fig, ax = plt.subplots(1,2)
#distance = np.abs(mcmc_traj[:,0,0] - mcmc_traj[:,1,0])
distance = np.abs(mcmc_traj[:,0,0])
ax[0].plot(distance, linewidth=0.5)
ax[0].set_xlabel('MCMC steps')
ax[0].set_ylabel(r'Delta position ($\AA$)')
dummy =  np.arange(lower+3.3, upper, 1e-3)
#dummy =  np.arange(lower, upper, 1e-3)
ax[1].plot(potential(dummy), dummy)
ax[1].set_ylim(lower, upper)
ax[1].set_xlabel('Energy of external potential (kcal/mol)')
plt.show()
print(f'Average distance between particles {distance.mean()} ang')
print(f'rmin {sigma*2**(1/6)} ang')
print(f'Average distance squared between particles {(distance**2).mean()} ang**2')
print(f'Standard deviation of distance between particles {distance.std()} ang')

import scipy.integrate as integrate
def probability_density(r, temperature, normalization):
    beta = 1/(constants.kb*temperature)
    return normalization*np.exp(-beta*potential(r))

zeta, error = integrate.quad(lambda r: probability_density(r, beta=beta, normalization=1), 0, 10.)
assert error*100/zeta < .1
mean_distance_theoretical, error = integrate.quad(lambda r: r*probability_density(r, temperature=temperature, normalization=1/zeta), 0, 10.)
assert error*100/mean_distance_theoretical < .1

dummy =  np.arange(lower+1e-5, upper, 1e-3)
fig, ax = plt.subplots(1,2, sharey=True, sharex=True)
ax[0].plot(probability_density(dummy, temperature=temperature, normalization=(1/zeta)), dummy)
ax[0].set_xlabel('Probability density')
ax[0].set_ylabel(r'Position ($\AA$)')
ax[1].hist(distance, bins=80, facecolor='blue', alpha=0.5, density=True, orientation='horizontal')
ax[1].set_xlabel('Estimated frequency')
ax[1].set_ylabel(r'Position ($\AA$)')
ax[1].set_ylim(lower, upper)
plt.show()
#Theoretical mean distance
print('Theoretical vs MCMC mean distance', mean_distance_theoretical, distance.mean(), np.abs(mean_distance_theoretical - distance.mean()))
