"""Different potential energy functions"""
import numpy as np
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

class LogGaussian:
    def __init__(self, sigma=[0.5, 0.5], A= [1, 1], mu=[4, 6]):
      params = zip(sigma, A, mu)
      self.gaussians = [Gaussian(sigma = s, height = -a*np.sqrt(2*np.pi*s**2), mu=m) for s, a, m in params]
    def __call__(self,r):
      out = sum([g(r) for g in self.gaussians])
      return -np.log(out)

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

