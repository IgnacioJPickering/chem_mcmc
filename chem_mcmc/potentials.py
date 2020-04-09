r"""Different potential energy functions"""
import numpy as np
import math
try:
    from chem_mcmc.cpp import potentials_cpp
    CPP_AVAIL = True
except ImportError:
    CPP_AVAIL = False


class Potential:
    def __init__(self):
        pass

    def __call__(self, r):
        raise NotImplementedError(
            'Call function not implemented for this potential')

    def dv(self, r):
        raise NotImplementedError(
            'Analytical derivative not implemented for this potential')

    def calc_extrema(self, lower, upper, sampling_precision=0.01, kind='max'):
        from scipy.signal import find_peaks
        dummy = np.arange(lower, upper, sampling_precision)
        if kind == 'max':
            samples = self(dummy)
        elif kind == 'min':
            samples = -self(dummy)
        peak_idxs, _ = find_peaks(samples)
        return dummy[peak_idxs]


class LennardJones(Potential):
    def __init__(self,
                 parameter1=None,
                 parameter2=None,
                 parametrization='epsilon_sigma',
                 epsilon=None, 
                 sigma=None,
                 center=0.0,
                 delta=1e-10,
                 cutoff=None):
        super().__init__()
        self.parametrization = parametrization
        if parametrization == 'epsilon_sigma':
            if parameter1 is None and parameter2 is None:
                assert epsilon is not None
                assert sigma is not None
                self.epsilon = epsilon
                self.sigma = sigma
            else:
                assert parameter1 is not None
                assert parameter2 is not None
                self.epsilon = parameter1
                self.sigma = parameter2
        elif parametrization == 'epsilon_rmin':
            self.epsilon = parameter1
            self.sigma = parameter2 / (2**(1 / 6))
        elif parametrization == 'a_b':
            self.epsilon = parameter2**2 / (4 * parameter1)
            self.sigma = (parameter1 / parameter2)**(1 / 6)
        else:
            raise ValueError('Incorrect parametrization')
        self.center = center
        self.delta = delta
        if cutoff is not None:
            self.cutoff = cutoff
        else:
            self.cutoff = math.inf



    def __call__(self, r):
        if r > self.cutoff:
            return 0.
        term1 = (self.sigma / (r - self.center + self.delta))**6
        term2 = term1**2
        out = term2 - term1
        return 4 * self.epsilon * out

    def dv(self, r):
        if r > self.cutoff:
            return 0.
        term1 = 6 * self.sigma**6 / (r - self.center + self.delta)**7
        term2 = 12 * self.sigma**12 / (r - self.center + self.delta)**13
        return 4 * self.epsilon * (-term2 + term1)


class Coulomb(Potential):
    def __init__(self,
                 q1=1.0,
                 q2=1.0,
                 epsilon=1.,
                 delta=1e-10,
                 center=0.0,
                 parametrization='q1_q2_epsilon_delta'):
        super().__init__()
        self.parametrization = parametrization
        # use a negative parameter for height if you want a repulsive potential
        self.q1 = q1
        self.q2 = q2
        self.epsilon = epsilon
        self.delta = delta
        self.center = center

    def __call__(self, r):
        return (1 / self.epsilon) * self.q1 * self.q2 / (r - self.center +
                                                         self.delta)

    def dv(self, r):
        return -(1 / self.epsilon) * self.q1 * self.q2 / (r - self.center +
                                                          self.delta)**2


class HardSpheres(Potential):
    def __init__(self, sigma, center=0.0, extra_room=1e-6):
        super().__init__()
        self.extra_room = extra_room
        self.sigma = sigma
        #TODO move with center
        self.center = center

    def __call__(self, r):
        if r > self.sigma:
            return np.full_like(r, 0.)
        return np.full_like(r, np.inf)

    def dv(self, r):
        # note: this force is fictitious, it is there to make the particles
        # go away if they are too close when initialized, during a minimization
        if r > (self.sigma + self.extra_room):
            return np.full_like(r, 0.)
        return np.full_like(r, -self.sigma / 2)


class HardSpheresStep(Potential):
    def __init__(self, sigma1, sigma2, epsilon, center=0.0, extra_room=1e-6):
        super().__init__()
        self.extra_room = extra_room
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.epsilon = epsilon
        #TODO move with center
        self.center = center

    def __call__(self, r):
        if r > self.sigma2:
            return np.full_like(r, 0.)
        if r > self.sigma1 and r <= self.sigma2:
            return np.full_like(r, -self.epsilon)
        return np.full_like(r, np.inf)

    def dv(self, r):
        # note: this force is fictitious, it is there to make the particles
        # go away if they are too close when initialized,  during a minimization
        if r > (self.sigma1 + self.extra_room):
            return np.full_like(r, 0.)
        return np.full_like(r, -self.sigma1 / 2)


class Gaussian(Potential):
    def __init__(self,
                 sigma,
                 height=1.0,
                 center=0.0,
                 parametrization='center_sigma_height'):
        super().__init__()
        self.parametrization = parametrization
        # use a negative parameter for height if you want a repulsive potential
        self.center = center
        self.sigma = sigma
        self.height = -height

    def __call__(self, r):
        constant = -1 / (2 * self.sigma**2)
        exponent = constant * (r - self.center)**2
        return self.height * (
            1 / np.sqrt(2 * np.pi * self.sigma**2)) * np.exp(exponent)

    def dv(self, r):
        constant = -1 / (2 * self.sigma**2)
        exponent = constant * (r - self.center)**2
        return self.height * constant * (1 / np.sqrt(
            2 * np.pi * self.sigma**2)) * np.exp(exponent) * 2 * (r -
                                                                  self.center)


class GaussianOpt(Potential):
    def __init__(self,
                 sigma,
                 height=1.0,
                 center=0.0,
                 parametrization='center_sigma_height'):
        super().__init__()
        self.parametrization = parametrization
        # use a negative parameter for height if you want a repulsive potential
        self.center = center
        self.sigma = sigma
        self.height = -height
        if not CPP_AVAIL:
            raise ImportError(
                'potential cant be used since CPP is not available')

    #I'm testing a Cpp optimization of this potential for 1 particle
    def __call__(self, r):
        return potentials_cpp.gaussian(r, self.sigma, self.center, self.height)

    def __call__(self, r):
        return potentials_cpp.gaussian(r, self.sigma, self.center, self.height)

    def dv(self, r):
        constant = -1 / (2 * self.sigma**2)
        exponent = constant * (r - self.center)**2
        return self.height * constant * (1 / np.sqrt(
            2 * np.pi * self.sigma**2)) * np.exp(exponent) * 2 * (r -
                                                                  self.center)


class LogGaussian(Potential):
    def __init__(self, sigma=[0.5, 0.5], A=[1, 1], mu=[4, 8]):
        super().__init__()
        params = zip(sigma, A, mu)
        self.gaussians = [
            Gaussian(sigma=s, height=-a * np.sqrt(2 * np.pi * s**2), center=m)
            for s, a, m in params
        ]

    def __call__(self, r):
        out = sum([g(r) for g in self.gaussians])
        return -np.log(out)


class LogGaussianOpt(Potential):
    def __init__(self, sigma=[0.5, 0.5], A=[1., 1.], mu=[4, 8]):
        super().__init__()
        self.sigmas = sigma
        self.As = A
        self.mus = mu
        if not CPP_AVAIL:
            raise ImportError(
                'potential cant be used since CPP is not available')

    def __call__(self, r):
        return potentials_cpp.log_gaussian(r, self.sigmas, self.As, self.mus)


class LennardJonesOpt(Potential):
    def __init__(self,
                 parameter1=None,
                 parameter2=None,
                 parametrization='epsilon_sigma',
                 epsilon=None, 
                 sigma=None,
                 center=0.0,
                 delta=1e-10, cutoff=None):
        super().__init__()
        self.parametrization = parametrization
        if parametrization == 'epsilon_sigma':
            if parameter1 is None and parameter2 is None:
                assert epsilon is not None
                assert sigma is not None
                self.epsilon = epsilon
                self.sigma = sigma
            else:
                assert parameter1 is not None
                assert parameter2 is not None
                self.epsilon = parameter1
                self.sigma = parameter2
        elif parametrization == 'epsilon_rmin':
            self.epsilon = parameter1
            self.sigma = parameter2 / (2**(1 / 6))
        elif parametrization == 'a_b':
            self.epsilon = parameter2**2 / (4 * parameter1)
            self.sigma = (parameter1 / parameter2)**(1 / 6)
        else:
            raise ValueError('Incorrect parametrization')
        if cutoff is not None:
            self.cutoff = cutoff
        else:
            self.cutoff = math.inf
        self.center = center
        self.delta = delta

    def __call__(self, r):
        return potentials_cpp.lennard_jones(r, self.epsilon, self.sigma, self.center, self.delta, self.cutoff)

    def dv(self, r):
        return potentials_cpp.lennard_jones_dv(r, self.epsilon, self.sigma, self.center, self.delta, self.cutoff)


class Cuadratic(Potential):
    def __init__(self, k, r0=0.0, parametrization='k_x0'):
        super().__init__()
        self.parametrization = parametrization
        # use a negative parameter for k if you want an attractive potential
        self.k = k
        self.r0 = r0

    def __call__(self, r):
        return 0.5 * self.k * (r - self.r0)**2

    def dv(self, r):
        return self.k * (r - self.r0)


class Constant(Potential):
    def __init__(self, c=0.0, parametrization='c'):
        super().__init__()
        self.parametrization = parametrization
        self.c = c

    def __call__(self, r):
        return np.full_like(r, self.c)

    def dv(self, r):
        return np.full_like(r, 0.)


class Buckingham(Potential):
    def __init__(self,
                 parameter1,
                 parameter2,
                 parameter3,
                 parametrization='r0_r1_gamma'):
        super().__init__()
        self.parametrization = parametrization
        if parametrization == 'r0_r1_gamma':
            self.r0 = parameter1
            self.r1 = parameter2
            self.gamma = parameter3
        else:
            raise ValueError('Incorrect parametrization')

    def __call__(self, r):
        return self.gamma * (np.exp(-r / self.r1) - (self.r0 / r)**6)

    def dv(self, r):
        return self.gamma * (-(1 / self.r1) * np.exp(-r / self.r1) + 6 *
                             (self.r0**6 / r**7))
