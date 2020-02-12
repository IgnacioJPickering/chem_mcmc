"""Utility for plotting the probability density and other calculations"""
import scipy.integrate as integrate
import numpy as np
from chem_mcmc import constants

def get_running_mean(ndarray):
    running_mean = np.cumsum(ndarray)/np.arange(1, len(ndarray)+1)
    return running_mean

def get_running_std(ndarray, mean=None):
    if mean is None:
        mean = ndarray.mean()
    ndarray = (ndarray - mean)**2
    running_std = np.cumsum(ndarray)/np.arange(1, len(ndarray)+1)
    return np.sqrt(running_std)

class PDensity:
    def __init__(self, temperature, potential,
                 lower, upper, normalization=None):
        self.lower = lower
        self.upper = upper
        self.potential = potential
        self.beta = 1/(constants.kb*temperature)
        if normalization is None:
            zeta, error = integrate.quad(lambda r: np.exp(
                -self.beta*potential(r)), lower, upper)
            self.norm = 1/zeta
            if error*100/zeta > .1:
                raise RuntimeError(
                  r'The error in the normalization constant is larger than 1%')
        else:
            self.norm = normalization

    def __call__(self,r):
        return self.norm*np.exp(-self.beta*self.potential(r))

    def get_mean_obs(self, obs, name=''):
        # Note that observable should be a callable function of r
        mean_obs, error = integrate.quad(
            lambda r: obs(r)*self.norm*np.exp(
                -self.beta*self.potential(r)), self.lower, self.upper)
        if error*100/mean_obs > .1:
            raise RuntimeError(
                f'The error in the mean {name} is larger than 1%')
        return mean_obs

    def get_std_obs(self, obs, name=''):
        # Note that observable should be a callable function of r
        mean_obs = self.get_mean_obs(obs, name)
        return np.sqrt(self.get_mean_obs(lambda r: (obs(r) - mean_obs)**2, name + 'square deviation'))

    def get_moment_obs(self, obs, name='', mom=1):
        return self.get_mean_obs(lambda r: obs(r)**mom, name + f'moment {mom}')

    def get_central_moment_obs(self, obs, name='', mom=1):
        mean_obs = self.get_mean_obs(obs, name)
        return self.get_mean_obs(lambda r: (obs(r)-mean_obs)**mom, name + f'moment {mom}')

    def get_mean_distance(self):
        return self.get_mean_obs( lambda r: r, 'distance')

    def get_std_distance(self):
        return self.get_std_obs(lambda r: r, 'distance')

    def get_mean_pot_energy(self):
        return self.get_mean_obs(lambda r: self.potential(r), 'potential')

    def get_std_pot_energy(self):
        return self.get_std_obs(lambda r: self.potential(r), 'potential')


