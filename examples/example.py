import chem_mcmc
import numpy as np
import matplotlib.pyplot as plt
from chem_mcmc.staging import Propagator, ParticleGroup
from chem_mcmc.potentials import LogGaussian
from chem_mcmc.utils import get_running_mean, get_running_std, PDensity
from chem_mcmc import potentials
from chem_mcmc import plotting
from chem_mcmc import constants
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

def plot_4run_properties(properties, titles, titlex, pdensity, axes=None, label_values=None):
    colors = ['r', 'g', 'orange', 'b']
    if label_values is None:
        default_values = [pdensity.get_mean_distance(), pdensity.get_std_distance(), pdensity.get_mean_pot_energy(), pdensity.get_std_pot_energy()]
        default_labels = ['Theo.mean', 'Theo. std', 'Theo. mean', 'Theo. std']
        label_values = True
    if axes is None:
        fig, ax = plt.subplots(1,4, figsize=(18, 6))
    else:
        ax = axes
    for j, a in enumerate(ax):
        a.plot(properties[j], color=colors[j])
        a.set_ylabel(titles[j])
        a.set_xlabel(titlex)
    xlims = np.asarray([a.get_xlim() for a in ax])
    if axes is None and label_values:
        for j, a in enumerate(ax):
            a.hlines(default_values[j], xmin=xlims[j, 0], xmax =xlims[j, 1], linestyles='dashed', linewidths=1.0, label = default_labels[j])
    return ax
plot_4run_properties(properties, titles, titlex, pdensity)
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
erg_titles = ['Erg. measure of ' + t for t in titles]
plot_4run_properties(properties, erg_titles, titlex, pdensity, label_values=False)
plt.show()
