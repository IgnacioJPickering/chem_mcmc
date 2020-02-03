"""Utility for plotting the probability density"""
import scipy.integrate as integrate

class PDensity:
    def __init__(self, temperature, potential,
                 lower, upper, normalization=None):
        self.lower = lower
        self.upper = upper
        self.potential = potential
        self.beta = 1/(kb*temperature)
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

    def get_mean_distance(self):
        mean_distance, error = integrate.quad(
            lambda r: r*self.norm*np.exp(
                -self.beta*self.potential(r)), self.lower, self.upper)
        if error*100/mean_distance > .1:
            raise RuntimeError(
                r'The error in the std of the distance is larger than 1%')
        return mean_distance

    def get_std_distance(self):
        std_distance, error = integrate.quad(
            lambda r: ((r - self.get_mean_distance())**2)*self.norm*np.exp(
                -self.beta*self.potential(r)), self.lower, self.upper)
        if error*100/std_distance > .1:
            raise RuntimeError(
                r'The error in the std of the distance is larger than 1%')
        return np.sqrt(std_distance)
