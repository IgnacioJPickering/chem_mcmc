import numpy as np
import matplotlib.pyplot as plt
import potentials
import constants

class Bounds:
    def __init__(self, kind=None, lower=[0.0], upper=[0.0]):
        # upper and lower should be a list [min, ...] [max, ...] etc, for all
        # coordinates bounds kind is by default to repeat the MCMC move if it
        # falls outside the bounds bounds kind should be either 'p' or 'r'
        # dimension as coordinates and can be 'periodic' or 'reflecting' 
        self.kind = kind
        self.lower = np.asarray(lower)
        self.upper = np.asarray(upper)
        self.sizes = self.upper - self.lower
        self.dimension = len(self.sizes)

        if len(self.lower) != self.dimension:
                raise ValueError('Upper and lower bounds should have the same dimension')
        if np.any(self.sizes < 0):
                raise ValueError('Upper bounds should be larger than lower bounds')

    def are_in_bounds(self, coordinates):
        too_low = np.any(coordinates < self.lower)
        too_high = np.any(coordinates > self.upper)
        return not(too_low or too_high)

    def wrap_coordinates(self, coordinates):
        """Takes in coordinates and wraps them to fit inside the bounds"""
        too_low_idx = (coordinates < self.lower).nonzero()
        too_high_idx = (coordinates > self.upper).nonzero()
        coordinates[too_low_idx] += self.sizes[too_low_idx]
        coordinates[too_high_idx] -= self.sizes[too_high_idx]

    @classmethod
    def square(cls, lower=0.0, upper=0.0, kind=None, dimension=1):
        """Makes square bounds"""
        lower = [lower]*dimension
        upper = [upper]*dimension
        return cls(kind=kind, lower=lower, upper=upper)

class ParticleGroup:
    def __init__(self, particles=None, bounds=None):
        if particles is None:
            self.particles = []
        else:
            self.particles = particles
        self.pairwise_potential = [potentials.Constant()]
        self.external_potential = [potentials.Constant()]
        self.bounds = bounds

    @classmethod
    def random_square(cls, number, lower=0.0, upper=10.0, kind=None, dimension=1, seed=None):
        """Creates a given number of particles in random positions inside square bounds"""
        prng = np.random.RandomState(seed=seed)
        bounds = Bounds.square(lower=lower, upper=upper, kind=kind, dimension=dimension)
        random_coordinates = prng.uniform(low=lower, high=upper, size=(number, dimension))
        particle_list = [Particle(r) for r in random_coordinates]
        return cls(particles=particle_list, bounds=bounds)

    def __iter__(self):
        yield from self.particles

    def __len__(self):
        return len(self.particles)
    
    def __getitem__(self, j):
        return self.particles[j]

    def get_coordinates(self, trial=False):
        if trial:
            coordinates_group = [p.trial_coordinates for p in self]
        else:
            coordinates_group = [p.coordinates for p in self]
        return np.asarray(coordinates_group)

    def get_forces(self):
        forces_group = [p.get_force() for p in self]
        return np.asarray(forces_group)

    def get_virial_per_particle(self):
        virial_group = [np.dot(p.pairwise_force, p.coordinates) for p in self]
        return np.asarray(virial_group)

    def get_virial(self):
        return self.get_virial_per_particle().sum()

    def get_volume(self):
        return np.prod(self[0].bounds.upper - self[0].bounds.lower)

    def get_pressure(self, temperature):
        return len(self)*constants.kb*temperature/self.get_volume() + self.get_virial()/(self[0].dimension*self.get_volume())

    def attach_pairwise_potential(self, potential):
        self.pairwise_potential.append(potential)

    def attach_external_potential(self, potential):
        self.external_potential.append(potential)
    
    def get_pairwise_potential(self, trial=False):
        total_potential = 0.
        num_particles = len(self)
        for j in range(num_particles):
            for k in range(j+1, num_particles):
                if trial:
                    r = np.linalg.norm(self[j].trial_coordinates - self[k].trial_coordinates)
                else:
                    r = np.linalg.norm(self[j].coordinates - self[k].coordinates)
                for pp in self.pairwise_potential:
                    total_potential += pp(r)
        return total_potential

    def get_external_potential(self, trial=False):
        total_external = 0.
        for p in self:
            if trial:
                r = np.linalg.norm(p.trial_coordinates)
            else:
                r = np.linalg.norm(p.coordinates)
            for ep in self.external_potential:
                total_external += ep(r)
        return total_external

    def calculate_pairwise_forces(self):
        num_particles = len(self)
        for j in range(num_particles):
            total_pairwise_force = np.zeros_like(self[0].coordinates)
            for k in range(num_particles):
                if k == j:
                    continue
                difference = self[j].coordinates - self[k].coordinates
                r = np.linalg.norm(difference)
                force_magnitude = sum([pp.dv(r) for pp in self.pairwise_potential])
                force = - force_magnitude * difference/r
                total_pairwise_force += force
            self[j].pairwise_force = np.copy(total_pairwise_force)

    def calculate_external_forces(self):
        num_particles = len(self)
        for p in self:
            r = np.linalg.norm(p.coordinates)
            force_magnitude = sum([ep.dv(r) for ep in self.external_potential])
            force = -force_magnitude * p.coordinates/r
            p.external_force = np.copy(force)

    def calculate_forces(self):
        self.calculate_external_forces()
        self.calculate_pairwise_forces()

    def get_potential(self, trial=False):
        # units of kcal/mol
        return self.get_external_potential(trial=trial) + self.get_pairwise_potential(trial=trial)

    def get_potential_difference(self):
        return self.get_potential(trial=True) - self.get_potential(trial=False)

    def get_kinetic(self, temperature):
        # units of kcal/mol
        return (self[0].dimension/2)*len(self)*temperature*constants.kb

    def get_total(self, temperature):
        return self.get_kinetic(temperature) + self.get_potential()

class Particle:
    def __init__(self, coordinates, velocities=None):

        self.dimension = len(coordinates)
        self.coordinates = np.asarray(coordinates)
        self.pairwise_force = np.zeros_like(self.coordinates)


        if velocities is not None:
            if len(velocities) != len(coordinates):
                raise ValueError('Velocities and coordinates should have the same dimension')
            self.velocities = np.asarray(velocities)
        else:
            self.velocities = np.zeros_like(coordinates)

        self.trial_coordinates = np.copy(self.coordinates)

    def get_force(self):
        return self.pairwise_force + self.external_force

class Propagator:
    def __init__(self, particle_group, seed=None, termo_properties=['trajectory']):
        self.termo_properties = termo_properties
        if 'trajectory' in self.termo_properties:
            self.trajectory = []
        if 'potential_energy' in self.termo_properties:
            self.potential_energy = []
        if 'kinetic_energy' in self.termo_properties:
            self.kinetic_energy = []
        if 'total_energy' in self.termo_properties:
            self.total_energy = []
        if 'forces' in self.termo_properties:
            self.forces = []
        if 'pressure' in self.termo_properties:
            self.pressure = []

        self.particle_group = particle_group
        self.prng = np.random.RandomState(seed=seed)

    def clear_properties(self, properties=None):
        if properties is not None:
            if 'trajectory' in properties:
                self.trajectory = []
            if 'potential_energy' in properties:
                self.potential_energy = []
            if 'kinetic_energy' in properties:
                self.kinetic_energy = []
            if 'total_energy' in properties:
                self.total_energy = []
            if 'forces' in properties:
                self.forces = []
            if 'pressure' in properties:
                self.pressure = []
        else:
                self.trajectory = []
                self.potential_energy = []
                self.kinetic_energy = []
                self.total_energy = []
                self.forces = []
                self.pressure = []

    def store_termo(self, temperature=0.0):
        if 'trajectory' in self.termo_properties:
            self.trajectory.append(self.particle_group.get_coordinates())
        if 'potential_energy' in self.termo_properties:
            self.potential_energy.append(self.particle_group.get_potential())
        if 'kinetic_energy' in self.termo_properties:
            self.kinetic_energy.append(self.particle_group.get_kinetic(temperature=temperature))
        if 'total_energy' in self.termo_properties:
            self.total_energy.append(self.particle_group.get_total(temperature=temperature))
        if 'forces' in self.termo_properties:
            self.forces.append(self.particle_group.get_forces())
        if 'pressure' in self.termo_properties:
            self.pressure.append(self.particle_group.get_pressure(temperature=temperature))

    def minimize(self, steps, alpha=0.1):
        for _ in range(steps):
            self.particle_group.calculate_forces()
            self.store_termo()
            for p in self.particle_group:
                p.coordinates =  p.coordinates + alpha*p.get_force()
                self.particle_group.bounds.wrap_coordinates(p.coordinates)

    def propagate_mcmc_nvt(self, steps, temperature=298, max_delta=0.5):
        for _ in range(steps):
            if 'forces' in self.termo_properties or 'pressure' in self.termo_properties:
                self.particle_group.calculate_forces()
            self.store_termo(temperature=temperature)
            beta =  1/(constants.kb*temperature)
            in_bounds = self.mcmc_translation_one(max_delta=max_delta)
            diff = self.particle_group.get_potential_difference()
            mc_factor = np.exp(-beta*diff)
            # If there are particles out of bounds the move is rejected 
            # This shouldn't happen with periodic boundary conditions
            # but it sometimes happens if the conditions are reflecting
            if self.prng.uniform(low=0., high=1.) < mc_factor and in_bounds:
                self.accept_mcmc_move()
            else:
                self.reject_mcmc_move()

    def mcmc_translation_one(self, max_delta=0.5):
        r"""Performs an MCMC translation move on all the coordinates of one particle
        
        According to ``number``, takes a certain number of coordinates and 
        displaces them randomly by taking a displacement from a uniform distribution 
        with bounds [-max_delta, max_delta]
        
        Parameters
        ----------
        number : int or None
            The number of coordinates to update (all by default if it is None).
        """
        particle = self.particle_group[self.prng.randint(0, len(self.particle_group))]
        particle.trial_coordinates = particle.coordinates + self.prng.uniform(low=-max_delta, high=max_delta, size=particle.coordinates.size)
        if self.particle_group.bounds.kind == 'p':
            self.particle_group.bounds.wrap_coordinates(particle.trial_coordinates)
        # Reflecting boundary conditions only means to reject moves that are out of bounds
        return self.particle_group.bounds.are_in_bounds(particle.trial_coordinates)

    def accept_mcmc_move(self):
        for p in self.particle_group:
            p.coordinates = np.copy(p.trial_coordinates)

    def reject_mcmc_move(self):
        for p in self.particle_group:
            p.trial_coordinates = np.copy(p.coordinates)

    def get_trajectory(self):
        return np.asarray(self.trajectory)

    def get_kinetic_energy(self):
        return np.asarray(self.kinetic_energy)

    def get_potential_energy(self):
        return np.asarray(self.potential_energy)

    def get_total_energy(self):
        return np.asarray(self.total_energy)

    def get_pressure(self):
        return np.asarray(self.pressure)

    def get_forces(self):
        return np.asarray(self.forces)

lower = 0.0
upper = 10.0
particle_group = ParticleGroup.random_square(number=1, lower=lower, upper=upper, dimension=1, kind='r')
epsilon = 0.2375 #kcal/mol
sigma = 3.4 #angstroems
potential = potentials.LennardJones(parameter1=epsilon, parameter2=sigma, parametrization='epsilon_sigma')
#particle_group.attach_pairwise_potential(potential)
particle_group.attach_external_potential(potential)
#prop = Propagator(particle_group, termo_properties=['forces','pressure','potential_energy', 'kinetic_energy', 'total_energy', 'trajectory'])
prop = Propagator(particle_group, termo_properties=['potential_energy', 'kinetic_energy', 'total_energy', 'trajectory'])
steps_min = 100
prop.minimize(steps_min, alpha=.01)
mcmc_traj = prop.get_trajectory()
fig, ax = plt.subplots(1,1)
for j, _ in enumerate(particle_group):
    ax.plot(mcmc_traj[:,j,0], linewidth=0.5)
ax.set_xlabel('minimization')
ax.set_ylabel(r'Position ($\AA$)')
ax.set_ylim(lower-10, upper+10)
plt.show()
prop.clear_properties()
temperature = 200 #K
steps_mcmc = 200000
prop.propagate_mcmc_nvt(steps=steps_mcmc, temperature=temperature, max_delta=1.0)
mcmc_traj = prop.get_trajectory()
pot_traj = prop.get_potential_energy()
kin_traj = prop.get_kinetic_energy()
tot_traj = prop.get_total_energy()
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

beta = 1/(constants.kb*temperature)
import scipy.integrate as integrate
def probability_density(r, beta, normalization):
    return normalization*np.exp(-beta*potential(r))

zeta, error = integrate.quad(lambda r: probability_density(r, beta=beta, normalization=1), 0, 10.)
assert error*100/zeta < .1
dummy =  np.arange(lower+1e-5, upper, 1e-3)
fig, ax = plt.subplots(1,2, sharey=True, sharex=True)
ax[0].plot(probability_density(dummy, beta=beta, normalization=(1/zeta)), dummy)
ax[0].set_xlabel('Probability density')
ax[0].set_ylabel(r'Position ($\AA$)')
ax[1].hist(distance, bins=80, facecolor='blue', alpha=0.5, density=True, orientation='horizontal')
ax[1].set_xlabel('Estimated frequency')
ax[1].set_ylabel(r'Position ($\AA$)')
ax[1].set_ylim(lower, upper)
plt.show()
#Theoretical mean distance
mean_distance_theoretical, error = integrate.quad(lambda r: r*probability_density(r, beta=beta, normalization=1/zeta), 0, 10.)
assert error*100/mean_distance_theoretical < .1
print('Theoretical vs MCMC mean distance', mean_distance_theoretical, distance.mean(), np.abs(mean_distance_theoretical - distance.mean()))
