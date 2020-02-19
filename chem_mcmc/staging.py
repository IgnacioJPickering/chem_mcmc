r"""General staging utilities, Bounds, Particle, ParticleGroup, etc."""
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from chem_mcmc import potentials
from chem_mcmc import constants
try:
    from chem_mcmc.cpp import staging_cpp
    CPP_AVAIL = True
except ImportError:
    CPP_AVAIL = False

#TODO external potentials and forces don't work correctly with PBC and 
# NDimensions right now
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
    
    def __init__(self, kind='n', lower=[0.0], upper=[0.0]):
        # upper and lower should be a list [min, ...] [max, ...] etc, for all
        # coordinates bounds kind is by default to repeat the MCMC move if it
        # falls outside the bounds bounds kind should be either 'p' or 'r' or 'n'
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

    def get_distance(self, coordinates1, coordinates2):
        difference = coordinates1 - coordinates2
        # wrap distances for periodic BC in pairwise forces 
        # if self.bounds.kind is "p" the distance is automatically wrapped
        if self.kind == 'p':
            idx = np.nonzero(difference > self.sizes/2)[0]
            difference[idx] -= self.sizes[idx]
            idx = np.nonzero(difference < -self.sizes/2)[0]
            difference[idx] += self.sizes[idx]
        r = np.linalg.norm(difference)
        return r

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
    def random_square(cls, number, lower=0.0, upper=10.0, kind=None, dimension=1, seed=None, use_cpp=True):
        """Creates a given number of particles in random positions inside square bounds"""
        prng = np.random.RandomState(seed=seed)
        if CPP_AVAIL and use_cpp:
            bounds = staging_cpp.Bounds.square(lower=lower, upper=upper, kind=kind, dimension=dimension)
        else:
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
        return np.prod(self.bounds.upper - self.bounds.lower)

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
                # wrap distances for periodic BC in pairwise forces 
                # if self.bounds.kind is "p" the distance is automatically wrapped
                if trial:
                    r = self.bounds.get_distance(self[j].trial_coordinates, self[k].trial_coordinates)
                else:
                    r = self.bounds.get_distance(self[j].coordinates, self[k].coordinates)
                for pp in self.pairwise_potential:
                    pot_value = pp(r) 
                    # if one of the potentials is infinite 
                    # stop doing calculations and return infinity
                    # directly, this should save some time
                    # math.isinf is faster than numpy for small arrays
                    if math.isinf(pot_value):
                        return np.inf
                    total_potential += pp(r)
        return total_potential

    def get_pairwise_potential_contribution(self, particle_idx, trial=False):
        # This function calculates the contribution of ONE
        # particle to the total potential energy.
        # The total potential energy can be obtained by 
        # straight calculation or by summing up the contributions 
        # and dividing by two.
        pairwise_contribution = 0.
        num_particles = len(self)
        for j in range(num_particles):
            if j == particle_idx: continue
        if trial:
            r = self.bounds.get_distance(self[j].trial_coordinates, self[k].trial_coordinates)
        else:
            r = self.bounds.get_distance(self[j].coordinates, self[k].coordinates)
        for pp in self.pairwise_potential:
            pot_value = pp(r) 
            # if one of the potentials is infinite 
            # stop doing calculations and return infinity
            # directly, this should save some time
            # math.isinf is faster than numpy for small arrays
            if math.isinf(pot_value):
                return np.inf
            pairwise_contribution += pp(r)
            # Factor of 0.5 corrects for double counting when 
            # summing the pairwise contributions of all particles, 
            # If you want to sum all contributions to get the total energy 
            # it MUST be present, but if you want to split the energy into 
            # U = U_others + U_particle
            # thin it SHOULDN't be present. This is slightly confusing but it 
            # works out. 
        return pairwise_contribution
    
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
                # wrap distances for periodic BC in pairwise forces 
                # if self.bounds.kind is "p" the distance is automatically wrapped
                difference = self[j].coordinates - self[k].coordinates
                r = self.bounds.get_distance(self[j].coordinates, self[k].coordinates)
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
        # More chances of pairwise being None than external
        # I am currently calculating the potential again for every step 
        # this is very inefficient, I should only calculate it once and then 
        # add a correction each step according to how the 
        # particles have moved
        return self.get_external_potential(trial=trial) + self.get_pairwise_potential(trial=trial)

    def get_potential_difference(self):
        potential_trial = self.get_potential(trial=True)
        # this avoids computation of the current potential if the external one
        # is infinity so it saves some time
        # math.isinf is faster than numpy for small arrays
        if math.isinf(potential_trial):
            return np.inf
        return potential_trial - self.get_potential(trial=False)

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

    def minimize(self, steps, alpha=0.1, max_step=0.5):
        for _ in range(steps):
            # Minimizations will always be carried in periodic boundary
            # conditions for the moment, even if the conditions are reflecting
            # or none, they are set to periodic for the minimization
            save_kind = self.particle_group.bounds.kind
            self.particle_group.bounds.kind = 'p'
            self.particle_group.calculate_forces()
            self.particle_group.bounds.kind = save_kind
            self.store_termo()
            for p in self.particle_group:
                step = alpha*p.get_force()
                # If the step is too large then reduce it to the max step
                step_size = np.linalg.norm(step)
                if step_size > max_step:
                    step = step*max_step/step_size
                p.coordinates =  p.coordinates + alpha*p.get_force()
                self.particle_group.bounds.wrap_coordinates(p.coordinates)

    def propagate_mcmc_nvt(self, steps, temperature=298, max_delta=0.5):
        self.acceptance_rate = 0.
        for _ in range(steps):
            if 'forces' in self.termo_properties or 'pressure' in self.termo_properties:
                self.particle_group.calculate_forces()
            self.store_termo(temperature=temperature)
            beta =  1/(constants.kb*temperature)
            in_bounds, moved_particle_idx = self.mcmc_translation_one(max_delta=max_delta)
            if not in_bounds:
                # If the particle is not in bounds the 
                # move is rejected automatically and there is nothing
                # else to check, this avoids some computation
                # This shouldn't happen with periodic boundary conditions
                # but it sometimes happens if the conditions are reflecting
                self.reject_mcmc_move()
                continue
            diff = self.particle_group.get_potential_difference()
            if math.isinf(diff):
                # If the trial particle is in a potential that is infinite
                # then the function automatically returns and
                # the move can be rejected without doing more computation
                self.reject_mcmc_move()
                continue
            mc_factor = np.exp(-beta*diff)
            if self.prng.uniform(low=0., high=1.) < mc_factor:
                self.accept_mcmc_move()
            else:
                self.reject_mcmc_move()
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
        # generating the random numbers through numpy is pretty costly, it should be
        # optimized
        particle_idx = self.prng.randint(0, len(self.particle_group))
        particle = self.particle_group[particle_idx]
        particle.trial_coordinates = particle.coordinates + self.prng.uniform(low=-max_delta, high=max_delta, size=particle.coordinates.size)
        if self.particle_group.bounds.kind == 'p':
            self.particle_group.bounds.wrap_coordinates(particle.trial_coordinates)
        # Reflecting boundary conditions only means to reject moves that are out of bounds
        return self.particle_group.bounds.are_in_bounds(particle.trial_coordinates), particle_idx

    def accept_mcmc_move(self):
        self.acceptance_rate +=1
        for p in self.particle_group:
            p.coordinates = np.copy(p.trial_coordinates)

    def reject_mcmc_move(self):
        for p in self.particle_group:
            p.trial_coordinates = np.copy(p.coordinates)

    def dump_to_xyz(self, xyz_path):
        with open(xyz_path, 'w') as f:
            trajectory = np.asarray(self.trajectory)
            num_atoms = trajectory.shape[1]
            for snapshot in trajectory:
                f.write(f'{num_atoms}\n')
                f.write('\n')
                for atom in snapshot:
                    if len(atom) > 3:
                        raise ValueError('Not possible to print xyz with dim > 3')
                    elif len(atom) < 3:
                        padded_atom = np.pad(atom, (0,3-len(atom)), constant_values=(0,0))
                        coord_str = ' '.join(padded_atom.astype(str).tolist())
                    else:
                        coord_str = ' '.join(atom.astype(str).tolist())
                    f.write(f'H {coord_str}\n')

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

