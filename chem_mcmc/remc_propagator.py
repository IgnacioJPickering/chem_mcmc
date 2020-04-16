import copy
import multiprocessing as mp

import numpy as np

from chem_mcmc.staging import Propagator, ParticleGroup
from chem_mcmc.potentials import LennardJonesOpt
from chem_mcmc import potentials
from chem_mcmc import constants

# Processes are spawend creating process objects and calling "start()"
# processes have a "target" which is a callable and arguments for that target
# processes are synchronized with "join", which waits until they finish
# execution (the parent process waits until the children finish)

# processes have a "run" method and a "start" method. The "start" method calls
# run by default, which executes the callable
# start -> run -> callable(*args, **kwargs)
# (run is technically called from a child process)

# The process can have a useless name which is given for identification only.
# Multiple processes can have the same name. Processes have an id which you 
# can get with process.id


# I will use the following cyclic convention, if my processes are:
# 0 1 2 3 then:
# the process to the left of 1 is 0 and to the right is 2
# the process to the left of 0 is 3 and to the right is 1
# the process to the left of 3 is 2 and to the right is 0
# processes are assigned increasing temperatures in the "temperatures" array
# Note that I don't necessarily lock processes to a specific CPU, processes 
# can use the same CPU or different ones, I leave this choice to the OS.

# I will give each process the end connection of  three pipes, one pipe will
# allow them to communicate with the neighboring process to their left, one to
# their right and one to the main, parent process

def replica_remc(random_square_kwargs, propagate_mcmc_nvt_onep_kwargs,
        potential, termo_properties, minimization_steps, burn_in_steps, mcmc_steps,
        replica_idx, connections):
    p_group = ParticleGroup.random_square(**random_square_kwargs)
    p_group.attach_pairwise_potential(potential)
    propagator = Propagator(p_group, termo_properties=termo_properties)
    propagator.minimize(minimization_steps)
    
    iterations = mcmc_steps//propagate_mcmc_nvt_onep_kwargs['steps']
    replica_acceptance = 0
    for iteration_idx in range(iterations):
        propagator.propagate_mcmc_nvt_onep(**propagate_mcmc_nvt_onep_kwargs)
        # Only returns true if the parities are different, for even 
        # iterations this has to be done by odd replicas 
        # for odd iterations this has to be done by even replicas
        if iteration_idx % 2 != replica_idx % 2:
            # for even iterations I have to exchange 0-1 2-3 4-5 6-7 so the 
            # processes in charge of the exchange will be 0 2 3 4 6 ... etc
            # This means I will only do the calculation if the process index is 
            # even, but I first need to receive a value from the processes with
            # odd process index
            # odd indices send data to even indices
            right_energy = propagator.get_potential_energy()[-1]
            right_temperature = propagate_mcmc_nvt_onep_kwargs['temperature']
            connections['left'].send((right_energy, right_temperature, replica_idx))
            # after doing this I try to receive some coordinates, if I receive
            # something then the exchange was successful! if I receive None then 
            # the exchange was not successful
            left_particle_group = connections['left'].recv()
            if left_particle_group is not None:
                # if the exchange was successful I do this
                connections['left'].send(propagator.particle_group)
                propagator.particle_group = left_particle_group
                propagator.store_termo(temperature=right_temperature)
                replica_acceptance +=1
            else:
                # if the exchange was unsuccessful I store the same
                # conformation again and continue
                propagator.store_termo(temperature=right_temperature)
        # Only returns true if the parities are equal, for even 
        # iterations this has to be done by even replicas 
        # for odd iterations this has to be done by odd replicas
        # This is the code that actually manages the exchange calculations
        if iteration_idx % 2  == replica_idx % 2:
                # Here I receive data and calculate the mcmc exchange
                # probability
                left_energy = propagator.get_potential_energy()[-1]
                left_temperature = propagate_mcmc_nvt_onep_kwargs['temperature']
                left_beta = 1/(constants.kb*left_temperature)

                # this blocks until there is something to receive
                right_energy, right_temperature, right_replica_idx = connections['right'].recv()   
                right_beta = 1/(constants.kb*right_temperature)

                mc_factor = np.exp((left_beta - right_beta)*(left_energy - right_energy))

                if propagator.prng.uniform(low=0., high=1.) < mc_factor:
                    # first I send the particle group through the pipe
                    connections['right'].send(propagator.particle_group)
                    # afterwards I receive the particle right particle_group
                    # from the pipe I change the coordinates of all the
                    # particles to be those of the particle group and I store
                    # the termo properties as if the temperature was the
                    # temperature of this specific replica
                    propagator.particle_group = connections['right'].recv()
                    propagator.store_termo(temperature=left_temperature)
                    replica_acceptance +=1
                else:
                    # I just send an "exchange rejected signal" and store the
                    # termo properties again
                    connections['right'].send(None)
                    propagator.store_termo(temperature=left_temperature)
    replica_acceptance_rate = replica_acceptance/iterations


def main():
    # these are parameters common to all of the REMC processes that will be called
    lower = 0.
    upper = 10.
    mcmc_steps = 10000
    minimization_steps = 50
    burn_in_steps = minimization_steps
    particle_number = 10
    max_delta = 1.0
    cutoff = 0.98 * (upper - lower)/2 
    
    # I will have to attempt exchange every n steps, this is a specific
    # parameter of the REMC parallel tempering code in this way the code
    # will run for "steps_per_exchange_attempt" before attempting an
    # exchange, then it will go on until it reaches mcmc_steps
    steps_per_exchange_attempt = 100
    temperatures = [300, 400, 500, 700]
    replicas = len(temperatures)
    
    # Checks that ensure variables are correctly set
    if mcmc_steps % steps_per_exchange_attempt != 0:
        raise ValueError("Currently it is only supported for mcmc_steps to be a multiple of steps_per_exchange attempt")
    if (replicas % 2 != 0) or (replicas < 2):
        raise ValueError("Only even an even number of replicas, greater or equal to 2, is currently supported")
    if len(temperatures) != replicas:
        raise ValueError("There should be as many temperatures as replicas")
    
    potential = LennardJonesOpt(sigma=1., epsilon=3.0, cutoff=cutoff)
    termo_properties = ['potential_energy', 'trajectory']
    random_square_kwargs = {'number': particle_number, 'lower':lower, 'upper':upper, 'dimension':2, 'kind':'p', 'use_cpp': False}
    propagate_mcmc_nvt_onep_kwargs = {'steps':steps_per_exchange_attempt, 'temperature':None, 'max_delta':max_delta}
    
    # this creates an array of pairs of pipes where the first pipe points to the left neighbohr
    # and the second pipe points to the right neighbohr
    left_right_connections = np.roll(np.asarray([mp.Pipe() for _ in range(replicas)]).reshape(-1, 2), -1)

    processes = []
    for replica_idx, temperature in zip(range(replicas), temperatures):
        connections = {'left': left_right_connections[replica_idx, 0],'right': left_right_connections[replica_idx, 1]}
        propagate_mcmc_nvt_onep_kwargs['temperature'] = temperature
        args =(random_square_kwargs, copy.deepcopy(propagate_mcmc_nvt_onep_kwargs),
                potential, termo_properties, minimization_steps, burn_in_steps,
                mcmc_steps, replica_idx, connections)
        p = mp.Process(target=replica_remc, args=args)
        processes.append(p)
    
    # I ignore numpy overflows since they are converted to inf automatically
    # which is what I want
    np.seterr(over='ignore')
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    np.seterr(over='warn')

if __name__ == "__main__":
    main()
