
import torch
from abc import ABC, abstractmethod

class Thermostat(ABC):
    def __init__(self, gamma, vocab_size=8):
        self.gamma = gamma
        self.vocab_size = vocab_size

    @abstractmethod
    def _integral(self, t0, t1):
        pass

    def wt_0(self, t):
        wt = self.w_ts(t, 1)
        return wt * self.vocab_size / (1 - wt)

    def wt_1(self, t):
        return self.w_ts(t, 1)

    def w_ts(self, t0, t1):
        return torch.exp(-self.vocab_size * self.gamma * self._integral(t0, t1))

class ConstantThermostat(Thermostat):
    ''' beta(r) = const.
    '''
    def _integral(self, t0, t1):
        return t1 - t0

class InverseThermostat(Thermostat):
    ''' beta(r) = 1/r
    '''
    def _integral(self, t0, t1):
        return torch.log(t1 / t0)

class LinearThermostat(Thermostat):
    ''' beta(r) = r
    '''
    def _integral(self, t0, t1):
        return (t1**2 - t0**2) / 2

class InverseSquareThermostat(Thermostat):
    ''' beta(r) = -1/r^2
    '''
    def _integral(self, t0, t1):
        return (t1 - t0) / (t1 * t0)

class SigmoidThermostat(Thermostat):
    ''' beta(r) = 1/(1+r)
    '''
    def _integral(self, t0, t1):
        return torch.tanh(t1/2) - torch.tanh(t0/2)