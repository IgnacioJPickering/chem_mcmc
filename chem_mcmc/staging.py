r"""General staging utilities, Bounds, Particle, ParticleGroup, etc."""
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from chem_mcmc import potentials
from chem_mcmc import constants

class Bounds:
    r"""Bounding box for a ParticleGroup
    
    This class is a bounding box that bounds all particles inside a particle 
    group
    
    Parameters
    ----------
    lower : list(float)
        List of lower bounds
    upper : list(float)
        List of upper bounds
    kind : str or None
        ``r`` for reflecting, ``p`` for periodic
        or ``None`` for no bounds
    
    Raises
    ------
    ValueError:
        If upper and lower bounds don't have the same dimension
        or upper bounds are not larger than lower bounds.
    """
    
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
        r"""Check if a set of coordinates of a particle are inside bounds
        
        Parameters
        ----------
        coordinates : np.ndarray
            set of particle coordinates
        
        Returns
        -------
        bool:
            whether the coordinates are all inside bounds
        """
        
        too_low = np.any(coordinates < self.lower)
        too_high = np.any(coordinates > self.upper)
        return not(too_low or too_high)

    def wrap_coordinates(self, coordinates):
        r"""Wraps a set of coordinates inside bounds
        
        Modifies the object in place
        
        Parameters
        ----------
        coordinates : np.ndarray
            set of particle coordinates
        """
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
                #Note: this is faster than numpy because coordinates is very small
                # and this part of the code is very costly
                # but numpy would be faster if I did everything at the same
                # time, which I SHOULD DO, for larger systems, check in the 
                # future
                r = math.sqrt(sum([c**2 for c in p.trial_coordinates]))
                #r = np.linalg.norm(p.trial_coordinates)
            else:
                r = math.sqrt(sum([c**2 for c in p.coordinates]))
                #r = np.linalg.norm(p.coordinates)
            for ep in self.external_potential:
                # this takes 70% of the time if I use stl for the normalization
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
        # thermo properties should be one of:
        # trajectory, kinetic_energy, total_energy, forces, pressure, potential_energy, 
        for p in termo_properties:
            setattr(self, p, [])

        self.particle_group = particle_group
        self.prng = np.random.RandomState(seed=seed)
        self.acceptance_rate = None
        self.last_run_time = None

    def clear_properties(self, properties=None):
        if properties is None: 
            properties = self.termo_properties
        for p in properties:
            setattr(self, p, []) 

    def burn_in(self,steps=0, properties=None):
        if properties is None: 
            properties = self.termo_properties
        for p in properties:
            setattr(self, p, getattr(self,p)[steps:]) 

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
        self.acceptance_rate = 0.
        start = time.time()
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
        end = time.time()
        self.last_run_time = end - start
        self.acceptance_rate /= steps

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
        self.acceptance_rate +=1
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

    def get_acceptance_percentage(self):
        return self.acceptance_rate*100

