import chem_mcmc
import numpy as np
import matplotlib.pyplot as plt
from chem_mcmc.staging import Propagator, ParticleGroup
from chem_mcmc.potentials import LogGaussian
from chem_mcmc.utils import get_running_mean, get_running_std, PDensity
from chem_mcmc import potentials
from chem_mcmc import plotting
from chem_mcmc import constants
from time import time
# EXAMPLE 1 do this once
temperature = 600
lower = 0.
upper = 12.
# ~ 80k is converged (10 s) , ~ 800k is very well converged
mcmc_steps = 20000
potential = LogGaussian(A=[2., 0.8], mu=[4, 8])
p_group = ParticleGroup.random_square(number=1, lower=lower, upper=upper, dimension=1, kind='r')
p_group.attach_external_potential(potential)
propagator = Propagator(p_group, termo_properties=['potential_energy', 'trajectory'])
propagator.propagate_mcmc_nvt(steps=mcmc_steps, temperature=temperature, max_delta=1.0)
propagator.burn_in(100)
position = propagator.get_trajectory()[:,0,0]
energy = propagator.get_potential_energy()
pdensity = PDensity(temperature, potential, lower, upper)
plotting.plot_analysis(position, pdensity, lower, upper, potential, bounds_potential = [-5, 40])

runmean_position = get_running_mean(position)
runstd_position = get_running_std(position)
runmean_energy = get_running_mean(energy)
runstd_energy = get_running_std(energy)
properties = [get_running_mean(position), get_running_std(position), get_running_mean(energy), get_running_std(energy)]
titles = [r'Mean position ($\AA$)',  r'Std. position ($\AA$)', r'Mean potential energy (kcal/mol)', r'Std. potential energy (kcal/mol)']
titlex = r'MCMC steps'

def plot_4run_properties(properties, titles, titlex, pdensity, axes=None, label_values=None, sharey=False, fit_params=False, fit_func=None):
    colors = ['r', 'g', 'orange', 'b']
    steps = range(len(properties[0]))
    if label_values is None:
        default_values = [pdensity.get_mean_distance(), pdensity.get_std_distance(), pdensity.get_mean_pot_energy(), pdensity.get_std_pot_energy()]
        default_labels = [f'Theo.mean {default_values[0]:.2e}', f'Theo. std {default_values[1]:.2e}', f'Theo. mean {default_values[2]:.2e}', f'Theo. std {default_values[3]:.2e}']
        label_values = True
    if axes is None:
        fig, ax = plt.subplots(1,4, figsize=(18, 6), sharey=sharey)
    else:
        ax = axes
    for j, a in enumerate(ax):
        a.plot(steps, properties[j], color=colors[j])
        a.set_ylabel(titles[j])
        a.set_xlabel(titlex)
    xlims = np.asarray([a.get_xlim() for a in ax])
    if fit_params:
        ylims = ax[0].get_ylim()
        for j, a in enumerate(ax):
            a.plot(steps, fit_func(steps, *fit_params[j]), color=colors[j], linestyle='dashed', linewidth=1.0, label=f'D = {fit_params[j][0]:.2e}')
        ax[0].set_ylim(ylims[0], ylims[1])
    if axes is None and label_values:
        for j, a in enumerate(ax):
            a.hlines(default_values[j], xmin=xlims[j, 0], xmax =xlims[j, 1], linestyles='dashed', linewidths=1.0, label = default_labels[j])
    return ax
axes = plot_4run_properties(properties, titles, titlex, pdensity)
for a in axes:
    a.legend()
plt.show()


# EXAMPLE 2 run a bunch of times to calculate "ergodic measure"
temperature = 600
lower = 0.
upper = 12.
mcmc_steps = 20000
potential = LogGaussian(A=[2., 0.8], mu=[4, 8])
initial_conditions = 10
runmean_positions = []
runstd_positions = []
runmean_energies = []
runstd_energies = []
pdensity = PDensity(temperature, potential, lower, upper)
axes = None
for _ in range(10):
    p_group = ParticleGroup.random_square(number=1, lower=lower, upper=upper, dimension=1, kind='r')
    p_group.attach_external_potential(potential)
    propagator = Propagator(p_group, termo_properties=['potential_energy', 'trajectory'])
    propagator.propagate_mcmc_nvt(steps=mcmc_steps, temperature=temperature, max_delta=1.0)
    propagator.burn_in(100)
    position = propagator.get_trajectory()[:,0,0]
    energy = propagator.get_potential_energy()
    properties = [get_running_mean(position), get_running_std(position), get_running_mean(energy), get_running_std(energy)]
    axes = plot_4run_properties(properties, titles, titlex, pdensity, axes=axes)
    runmean_positions.append(properties[0])
    runstd_positions.append(properties[1])
    runmean_energies.append(properties[2])
    runstd_energies.append(properties[3])
plt.show()

ergmean_position = np.asarray(runmean_positions).std(axis=0)
ergstd_position = np.asarray(runstd_positions).std(axis=0)
ergmean_energy = np.asarray(runmean_energies).std(axis=0)
ergstd_energy = np.asarray(runstd_energies).std(axis=0)
properties = [ergmean_position, ergstd_position, ergmean_energy, ergstd_energy]
# standarize the ergodic measure
from scipy.optimize import curve_fit
def erg_fitting(x, D):
    eps = 1e-10
    return 1/(x*D+eps)
properties = [p/p[0] for p in properties]
params = [curve_fit(erg_fitting, p, range(len(p)))[0] for p in properties]
erg_titles = ['Erg. measure of ' + t.split('(')[0] for t in titles]
axes = plot_4run_properties(properties, erg_titles, titlex, pdensity, label_values=False, sharey=True, fit_params = params, fit_func=erg_fitting)
for a in axes:
    a.legend()
plt.show()
