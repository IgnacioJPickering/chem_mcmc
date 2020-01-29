import numpy as np
import matplotlib.pyplot as plt
joule2kcalpermol = 6.022e23/(1000*4.184)
kboltzmann = 1.380649e-23*joule2kcalpermol #kcal/mol* K


class LennardJones:
    def __init__(self,parameter1, parameter2, parametrization='epsilon_sigma', center=0.0, delta=1e-10):
        self.parametrization = parametrization
        if parametrization == 'epsilon_sigma':
            self.epsilon = parameter1
            self.sigma = parameter2
        elif parametrization == 'epsilon_rmin':
            self.epsilon = parameter1
            self.sigma = parameter2/(2**(1/6))
        elif parametrization == 'a_b':
            self.epsilon = parameter2**2/(4*parameter1)
            self.sigma = (parameter1/parameter2)**(1/6)
        else:
            raise ValueError('Incorrect parametrization')
        self.center = center
        self.delta = delta
    
    def __call__(self, r):
        term1 = (self.sigma/(r-self.center+self.delta))**6
        term2 = term1**2
        out = term2 - term1
        return 4*self.epsilon*out

    def dv(self, r):
        term1 = 6*self.sigma**6/(r - self.center + self.delta)**7
        term2 = 12*self.sigma**12/(r - self.center + self.delta)**13
        return 4*self.epsilon*(-term2 + term1)


class Coulomb:
    def __init__(self, q1=1.0, q2=1.0, epsilon=1., delta=1e-10, center=0.0, parametrization='q1_q2_epsilon_delta'):
        self.parametrization = parametrization
        # use a negative parameter for height if you want a repulsive potential
        self.q1 = q1
        self.q2 = q2
        self.epsilon = epsilon
        self.delta = delta
        self.center = center
    
    def __call__(self, r):
        return (1/self.epsilon)*self.q1*self.q2/(r - self.center + self.delta)

    def dv(self, r):
        return  -(1/self.epsilon)*self.q1*self.q2/(r - self.center + self.delta)**2


class Gaussian:
    def __init__(self, sigma, height=1.0, center=0.0, parametrization='center_sigma_height'):
        self.parametrization = parametrization
        # use a negative parameter for height if you want a repulsive potential
        self.center = center
        self.sigma = sigma
        self.height = -height
    
    def __call__(self, r):
        constant =  -1/(2*self.sigma**2)
        exponent = constant*(r-self.center)**2
        return self.height*(1/np.sqrt(2*np.pi*self.sigma**2))*np.exp(exponent) 

    def dv(self, r):
        constant =  -1/(2*self.sigma**2)
        exponent = constant*(r-self.center)**2
        return self.height*constant*(1/np.sqrt(2*np.pi*self.sigma**2))*np.exp(exponent)*2*(r-self.center)


class Cuadratic:
    def __init__(self, k, r0=0.0, parametrization='k_x0'):
        self.parametrization = parametrization
        # use a negative parameter for k if you want an attractive potential
        self.k = k
        self.r0 = r0

    def __call__(self, r):
        return 0.5*self.k*(r - self.r0)**2

    def dv(self, r):
        return self.k*(r - self.r0)


class Constant:
    def __init__(self, c=0.0, parametrization='c'):
        self.parametrization = parametrization
        self.c = c

    def __call__(self, r):
        return np.full_like(r,self.c)

    def dv(self, r):
        return np.full_like(r,0.)


class Buckingham:
    def __init__(self, parameter1, parameter2, parameter3, parametrization='r0_r1_gamma'):
        self.parametrization = parametrization
        if parametrization == 'r0_r1_gamma':
            self.r0 = parameter1
            self.r1 = parameter2
            self.gamma = parameter3
        else:
            raise ValueError('Incorrect parametrization')

    def __call__(self, r):
        return self.gamma*(np.exp(-r/self.r1) - (self.r0/r)**6)

    def dv(self, r):
        return self.gamma*(-(1/self.r1)*np.exp(-r/self.r1) + 6*(self.r0**6/r**7))



class Bounds:
    def __init__(self, kind_lower=None, kind_upper=None, lower=[0.0], upper=[0.0]):
        # upper and lower should be a list [min, ...] [max, ...] etc, for all
        # coordinates bounds kind is by default to repeat the MCMC move if it
        # falls outside the bounds bounds kind should be a list of the same
        # dimension as coordinates and can be 'periodic' or 'reflecting' if
        # bound behaviour is a single word then it is assumed to be applied to
        # all coordinates

        if not isinstance(kind_lower, list):
            self.kind_lower = [kind_lower]*len(lower)
        else:
            self.kind_lower = kind_lower

        if not isinstance(kind_upper, list):
            self.kind_upper = [kind_upper]*len(upper)
        else:
            self.kind_upper = kind_upper

        self.lower = np.asarray(lower)
        self.upper = np.asarray(upper)
        self.dimension = len(self.upper)

        if len(self.lower) != self.dimension:
                raise ValueError('Upper and lower bounds should have the same dimension')
        if len(self.kind_lower) != self.dimension:
                raise ValueError('Kind of bounds and bounds should have the same dimension')
        if len(self.kind_upper) != self.dimension:
                raise ValueError('Kind of bounds and bounds should have the same dimension')

    def is_inside(self, q, j):
        if self.kind_lower[j] is not None:
            if self.kind_lower[j] == 'reflecting':
                if q < self.lower[j]:
                    return False
        if self.kind_upper[j] is not None:
            if self.kind_upper[j] == 'reflecting':
                if q > self.upper[j]:
                    return False
        return True

    @classmethod
    def square(cls, lower=0.0, upper=0.0, kind=None, dimension=1):
        lower = [lower]*dimension
        upper = [upper]*dimension
        return cls(kind_lower=kind, kind_upper=kind, lower=lower, upper=upper)


class ParticleGroup:
    def __init__(self, particles=None, bounds=None):
        if particles is None:
            self.particles = []
        else:
            self.particles = particles
        self.pairwise_potential = [Constant()]
        self.external_potential = [Constant()]
        if bounds is not None:
            self.assign_bounds(bounds)
            self.bounds = bounds

    @classmethod
    def random_square(cls, number, lower=0.0, upper=10.0, kind=None, dimension=1, seed=None):
        prng = np.random.RandomState(seed=seed)
        bounds = Bounds.square(lower=lower, upper=upper, kind=kind, dimension=dimension)
        random_coordinates = prng.uniform(low=lower, high=upper, size=(number, dimension))
        particle_list = [Particle(r) for r in random_coordinates]
        return cls(particles=particle_list, bounds=bounds)

    def members(self):
        yield from self.particles

    def __len__(self):
        return len(self.particles)
    
    def __getitem__(self, j):
        return self.particles[j]

    def assign_bounds(self, bounds):
        for p in self.members():
            p.bounds = bounds

    def mcmc_translation(self, number_coords=None, number_particles=1, max_delta=0.5, gaussian=False):
        prng = np.random.RandomState(None)
        indices = prng.randint(0, len(self), size =(number_particles))
        for j in indices:
            self[j].mcmc_translation(number_coords=number_coords, max_delta=max_delta, gaussian=False)

    def accept_mcmc_move(self):
        for p in self.members():
            p.accept_mcmc_move()

    def reject_mcmc_move(self):
        for p in self.members():
            p.reject_mcmc_move()

    def get_coordinates(self, trial=False):
        if trial:
            coordinates_group = [p.trial_coordinates for p in self.members()]
        else:
            coordinates_group = [p.coordinates for p in self.members()]
        return np.asarray(coordinates_group)

    def get_forces(self):
        forces_group = [p.get_force() for p in self.members()]
        return np.asarray(forces_group)

    def get_virial_per_particle(self):
        virial_group = [np.dot(p.pairwise_force, p.coordinates) for p in self.members()]
        return np.asarray(virial_group)

    def get_virial(self):
        return self.get_virial_per_particle().sum()

    def get_volume(self):
        return np.prod(self[0].bounds.upper - self[0].bounds.lower)

    def get_pressure(self, temperature):
        return len(self)*kboltzmann*temperature/self.get_volume() + self.get_virial()/(self[0].dimension*self.get_volume())

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
        for p in self.members():
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
        for p in self.members():
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
        return (self[0].dimension/2)*len(self)*temperature*kboltzmann

    def get_total(self, temperature):
        return self.get_kinetic(temperature) + self.get_potential()


class Particle:
    def __init__(self, coordinates, velocities=None, seed=None, bounds=None):
        self.dimension = len(coordinates)
        self.coordinates = np.asarray(coordinates)
        self.pairwise_force = np.zeros_like(self.coordinates)
        if bounds is None:
            self.bounds = Bounds(lower=np.zeros_like(self.coordinates), upper=np.zeros_like(self.coordinates))
        else:
            self.bounds = bounds

        if velocities is not None:
            if len(velocities) != len(coordinates):
                raise ValueError('Velocities and coordinates should have the same dimension')
            self.velocities = np.asarray(velocities)
        else:
            self.velocities = np.zeros_like(coordinates)

        if self.dimension != self.bounds.dimension:
                raise ValueError('Bounds and particles should have the same dimensions')
        self.prng = np.random.RandomState(seed=seed)
        self.trial_coordinates = np.copy(self.coordinates)

    def mcmc_translation(self, number_coords=None, max_delta=0.5, gaussian=False):
        r"""Performs an MCMC translation move on the coordinates of the particle
        
        According to ``number``, takes a certain number of coordinates and 
        displaces them randomly by taking a displacement from a uniform distribution 
        with bounds [-max_delta, max_delta]
        
        Parameters
        ----------
        number : int or None
            The number of coordinates to update (all by default if it is None).
        """
        
        if number_coords is None:
            indices = range(len(self.coordinates))
        else:
            indices = self.prng.randint(0,self.dimension, size=(number_coords))
        for j in indices:
            is_in_bounds = False
            while not is_in_bounds:
                if gaussian:
                    new_coordinate = self.coordinates[j] + self.prng.normal(0, max_delta)
                else:
                    new_coordinate = self.coordinates[j] + self.prng.uniform(low=-max_delta, high=max_delta)
                is_in_bounds = self.bounds.is_inside(new_coordinate, j)
            self.trial_coordinates[j] = new_coordinate

    def accept_mcmc_move(self):
        self.coordinates = np.copy(self.trial_coordinates)

    def reject_mcmc_move(self):
        #This is necessary in order to reset the trial coordinates for partial moves on some variables
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
            for p in self.particle_group.members():
                new_coordinates =  p.coordinates + alpha*p.get_force()
                for j, c in enumerate(new_coordinates):
                    if not self.particle_group.bounds.is_inside(c, j):
                        low = self.particle_group.bounds.lower[j]
                        high  = self.particle_group.bounds.upper[j]
                        new_coordinates[j] = self.prng.uniform(low, high)
                p.coordinates = new_coordinates

    def propagate_mcmc_nvt(self, steps, temperature=298, max_delta=0.5, number_coords=None, number_particles=1, gaussian=False):
        for _ in range(steps):
            if 'forces' in self.termo_properties or 'pressure' in self.termo_properties:
                self.particle_group.calculate_forces()
            self.store_termo(temperature=temperature)
            beta =  1/(kboltzmann*temperature)
            self.particle_group.mcmc_translation(max_delta=max_delta, number_coords=number_coords, number_particles=number_particles, gaussian=gaussian)
            diff = particle_group.get_potential_difference()
            mc_factor = np.exp(-beta*diff)
            if self.prng.uniform(low=0., high=1.) < mc_factor:
                self.particle_group.accept_mcmc_move()
            else:
                self.particle_group.reject_mcmc_move()

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
upper = 200.0
#bounds = Bounds.square(upper=upper, lower=lower, kind='reflecting', dimension=1)
#particle_group = ParticleGroup([Particle([3.4]) for _ in range(1)])
#particle_group = ParticleGroup([Particle([4.0]), Particle([8.0])])
#particle_group.assign_bounds(bounds)
particle_group = ParticleGroup.random_square(number=20, lower=lower, upper=upper, dimension=1, kind='reflecting')

epsilon = 0.2375 #kcal/mol
sigma = 3.4 #angstroems
potential = LennardJones(parameter1=epsilon, parameter2=sigma, parametrization='epsilon_sigma')
#potential = Cuadratic(k=0.1, r0=3.4)
#potential = Constant(c=0.0)
#particle_group.attach_external_potential(potential)
particle_group.attach_pairwise_potential(potential)

prop = Propagator(particle_group, termo_properties=['forces','pressure','potential_energy', 'kinetic_energy', 'total_energy', 'trajectory'])
steps_min = 100
prop.minimize(steps_min, alpha=.01)
mcmc_traj = prop.get_trajectory()
fig, ax = plt.subplots(1,1)
print(len(particle_group))
for j, _ in enumerate(particle_group.members()):
    ax.plot(mcmc_traj[:,j,0], linewidth=0.5)
ax.set_xlabel('minimization')
ax.set_ylabel(r'Position ($\AA$)')
ax.set_ylim(lower-10, upper+10)
plt.show()
prop.clear_properties()
#exit()
#particle_group.calculate_forces()
#forces = particle_group.get_forces()
#virial = particle_group.get_virial()
#pressure = particle_group.get_pressure(temperature=15)

#prop = Propagator(particle_group, termo_properties=['forces','pressure','potential_energy', 'kinetic_energy', 'total_energy', 'trajectory'])
#print('done with min')
T = 15 #K
steps_mcmc = 2000
#steps_mcmc = 10
prop.propagate_mcmc_nvt(steps=steps_mcmc, temperature=300, max_delta=10., gaussian=True)
print('done with propr')

mcmc_traj = prop.get_trajectory()
pot_traj = prop.get_potential_energy()
kin_traj = prop.get_kinetic_energy()
tot_traj = prop.get_total_energy()
press_traj = prop.get_pressure()
plt.plot(press_traj, linewidth = 0.5)
plt.gca().set_xlabel('MCMC steps')
plt.gca().set_ylabel(r'Linear pressure ($kcal$ $mol^{-1}$ $\AA^{-1}$)')
xlo, xhi = plt.gca().get_xlim()
mean_press = press_traj.mean()
plt.hlines(mean_press, xmin=xlo, xmax=xhi, linestyles='dashed', colors='k', label=f'Mean Pressure {mean_press:.4f}'r' ($kcal$ $mol^{-1}$ $\AA^{-1}$)')
plt.legend()
plt.show()
#exit()
plt.plot(pot_traj, linewidth = 0.5, label='Potential', color='b')
plt.plot(kin_traj, linewidth = 0.5, label='Kinetic', color='r')
plt.plot(tot_traj, linewidth = 0.5, label='Total', color='g')
mean_tot = tot_traj.mean()
std_tot = tot_traj.std()
heat_capacity = std_tot/(kboltzmann*T**2)
mean_pot = pot_traj.mean()
plt.gca().set_xlabel('MCMC steps')
plt.gca().set_ylabel(r'Energy ($kcal$ $mol^{-1}$)')
xlo, xhi = plt.gca().get_xlim()
plt.hlines(mean_pot, xmin=xlo, xmax=xhi, linestyles='dashed', colors='darkblue', label=f'Mean Potential {mean_pot:.4f}'r' ($kcal$ $mol^{-1}$)')
plt.hlines(mean_tot, xmin=xlo, xmax=xhi, linestyles='dashed', colors='darkgreen', label=f'Mean Total {mean_tot:.4f}'r' ($kcal$ $mol^{-1}$)')
plt.hlines([mean_tot+std_tot, mean_tot - std_tot], xmin=xlo, xmax=xhi, linestyles='dashed', colors='darkgreen', linewidths=[0.5, 0.5], label=f'Heat capacity {heat_capacity:.2f}'r' ($kcal$ $mol^{-1}$ $K^{-1}$)')
plt.legend()
plt.show()
#exit()


#fig, ax = plt.subplots(1,2, sharey=True)
#for j, _ in enumerate(particle_group.members()):
#    ax[0].plot(mcmc_traj[:,j,0], linewidth=0.5)
#ax[0].set_xlabel('MCMC steps')
#ax[0].set_ylabel(r'Position ($\AA$)')
#dummy =  np.arange(lower+3.3, upper, 1e-3)
##dummy =  np.arange(lower, upper, 1e-3)
#ax[1].plot(potential(dummy), dummy)
#ax[1].set_ylim(lower, upper)
#ax[1].set_xlabel('Energy of external potential (kcal/mol)')
#plt.show()

fig, ax = plt.subplots(1,1)
for j, _ in enumerate(particle_group.members()):
    ax.plot(mcmc_traj[:,j,0], linewidth=0.5)
ax.set_xlabel('MCMC steps')
ax.set_ylabel(r'Position ($\AA$)')
ax.set_ylim(lower, upper)
plt.show()
exit()

distance = np.abs(mcmc_traj[:,0,0] - mcmc_traj[:,1,0])
ax[1].plot(distance, linewidth=0.5)
ax[1].set_xlabel('MCMC steps')
ax[1].set_ylabel(r'Delta position ($\AA$)')
dummy =  np.arange(lower+3.3, upper, 1e-3)
#dummy =  np.arange(lower, upper, 1e-3)
ax[2].plot(potential(dummy), dummy)
ax[2].set_ylim(lower, upper)
ax[2].set_xlabel('Energy of external potential (kcal/mol)')
plt.show()
print(f'Average distance between particles {distance.mean()} ang')
print(f'rmin {sigma*2**(1/6)} ang')
print(f'Average distance squared between particles {(distance**2).mean()} ang**2')
print(f'Standard deviation of distance between particles {distance.std()} ang')

beta = 1/(kboltzmann*T)
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
