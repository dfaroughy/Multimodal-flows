import torch
from torch.nn.functional import softmax
from torch.distributions import Categorical

from tensorclass import TensorMultiModal
from datamodules.datasets import DataCoupling

class UniformFlow:
    """Conditional OT Flow-Matching for continuous states.
    This bridge is a linear interpolation between boundaries states at t=0 and t=1.
    notation:
      - t: time
      - x0: continuous source state at t=0
      - x1: continuous target state at t=1
      - x: continuous state at time t
      - z: delta function regularizer
    """

    def __init__(self, sigma):
        self.sigma = sigma

    def sample(self, t, batch: DataCoupling):
        x0 = batch.source.continuous
        x1 = batch.target.continuous
        xt = t * x1 + (1.0 - t) * x0
        z = torch.randn_like(xt)
        std = self.sigma
        return xt + std * z

    def drift(self, state: TensorMultiModal, batch: DataCoupling):
        x0 = batch.source.continuous
        x1 = batch.target.continuous
        xt = state.continuous
        A = 0.0
        B = 1.0
        C = -1.0
        return A * xt + B * x1 + C * x0

    def diffusion(self, state: TensorMultiModal):
        return 0.0


class SimplexFlow:
    """Conditional Flow-Matching for continuous states on the simplex.
    This bridge is a linear interpolation between boundaries states at t=0 and t=1.
    notation:
      - t: time
      - e0: one-hot encoded source state at t=0
      - e1: one-hot encoded target state at t=1
      - x: simplex state at time t
    """

    def __init__(self, sigma):
        self.sigma = sigma

    def sample(self, t, batch: DataCoupling):
        e0 = batch.source.continuous
        e1 = batch.target.continuous
        x = t * e1 + (1.0 - t) * e0
        concentration = torch.ones(8)
        dirichlet = torch.distributions.dirichlet.Dirichlet(concentration)
        self.z = dirichlet.sample(e0.shape[:-1]).to(e0.device)
        std = self.sigma * t * (1.0 - t)
        return (x + std * self.z) / (1 + std)

    def drift(self, state: TensorMultiModal, batch: DataCoupling):
        t = state.time
        e0 = batch.source.continuous
        e1 = batch.target.continuous        
        xt = state.continuous
        std = self.sigma * t * (1.0 - t)
        std_dot = self.sigma *(1.0 - 2.0 * t) 
        return (e1 - e0 + std_dot * (self.z - xt)) / (1 + std) 

    def diffusion(self, state: TensorMultiModal):
        return 0.0



class SchrodingerBridge:
    """Schrodinger bridge for continuous states
    notation:
      - t: time
      - x0: continuous source state at t=0
      - x1: continuous  target state at t=1
      - x: continuous state at time t
      - z: noise
    """

    def __init__(self, sigma):
        self.sigma = sigma

    def sample(self, t, batch: DataCoupling):
        x0 = batch.source.continuous
        x1 = batch.target.continuous
        x = t * x1 + (1.0 - t) * x0
        z = torch.randn_like(x)
        std = self.sigma * torch.sqrt(t * (1.0 - t))
        return x + std * z

    def drift(self, state: TensorMultiModal, batch: DataCoupling):
        x0 = batch.source.continuous
        x1 = batch.target.continuous
        xt = state.continuous
        t = state.time
        A = (1 - 2 * t) / (t * (1 - t))
        B = t**2 / (t * (1 - t))
        C = -1 * (1 - t) ** 2 / (t * (1 - t))
        return A * xt + B * x1 + C * x0

    def diffusion(self, state: TensorMultiModal):
        state.broadcast_time()
        t = state.time
        return self.sigma * torch.sqrt(t * (1.0 - t))


class RandomTelegraphBridge:
    """Multivariate Random Telegraph bridge for discrete states
    - t: time
    - k0: discrete source state at t=0
    - k1: discrete  target state at t=1
    - k: discrete state at time t
    """

    def __init__(self, gamma, vocab_size, thermostat_fn):
        self.gamma = gamma
        self.vocab_size = vocab_size
        self.thermostat = thermostat_fn

    def rate(self, state: TensorMultiModal, logits: torch.Tensor):
        """ input:
            - state.time (t): (B, 1) time tensor
            - state.discrete (k) : (B, D, 1) current state tensor
            - logits: (B, D, vocab_size) logits tensor

            output:
            - rates: (B, D, vocab_size) transition rates tensor

        """
        
        t = state.time
        k = state.discrete

        assert (k >= 0).all() and (k < self.vocab_size).all(), (
            "Values in `k` outside of bound! k_min={}, k_max={}".format(
                k.min(), k.max()
            )
        )

        qx = softmax(logits, dim=2) # transition probabilities to all states
        qy = torch.gather(qx, 2, k.long())  # current state prob

        # ...Telegraph process rates:

        t = t.squeeze()
        wt = self.thermostat.w_ts(t, 1.0)
        A = 1.0
        B = (wt * self.vocab_size) / (1.0 - wt)
        C = wt
        rate = A + B[:, None, None] * qx + C[:, None, None] * qy
        return rate

    def sample(self, t, batch: DataCoupling):

        k0 = batch.source.discrete
        k1 = batch.target.discrete

        transition_probs = self.transition_probability(t, k0, k1)
        drawn_state = Categorical(transition_probs).sample().to(k1.device)

        if drawn_state.dim() == 2:
            drawn_state = drawn_state.unsqueeze(-1)

        return drawn_state

    def transition_probability(self, t, k0, k1):
        """
        \begin{equation}
        P(x_t=x|x_0,x_1) = \frac{p(x_1|x_t=x) p(x_t = x|x_0)}{p(x_1|x_0)}
        \end{equation}
        """
        # ...reshape input tenors:
        t = t.clone().squeeze()  # shape: (B)

        if k0.dim() == 1:
            k0 = k0.unsqueeze(1)  

        if k1.dim() == 1:
            k1 = k1.unsqueeze(1)

        # ...set state configurations:
        k = torch.arange(0, self.vocab_size)  # (V,)
        k = k[None, None, :].repeat(k0.size(0), k0.size(1), 1).float()  # (B, N, V)
        k = k.to(k0.device)

        # ...compute probabilities:

        p_k_to_k1 = self.conditional_probability(t, 1.0, k, k1)
        p_k0_to_k = self.conditional_probability(0.0, t, k0, k)
        p_k0_to_k1 = self.conditional_probability(0.0, 1.0, k0, k1)

        return (p_k_to_k1 * p_k0_to_k) / p_k0_to_k1

    def conditional_probability(self, t_in, t_out, k_in, k_out):
        """
        \begin{equation}
        P(x(t) = i|x(t_0)) = \frac{1}{s} + w_{t,t_0}\left(-\frac{1}{s} + \delta_{i,x(t_0)}\right)
        \end{equation}

        \begin{equation}
        w_{t,t_0} = e^{-S \int_{t_0}^{t} \beta(r)dr}
        \end{equation}

        """

        t_out = right_time_size(t_out, k_out).to(k_in.device)
        t_in = right_time_size(t_in, k_out).to(k_in.device)

        wt = self.thermostat.w_ts(t_in, t_out)

        k_out, k_in = right_shape(k_out), right_shape(k_in)
        kronecker = (k_out == k_in).float()
        prob = 1.0 / self.vocab_size + wt[:, None, None] * ((-1.0 / self.vocab_size) + kronecker)
        return prob


right_shape = lambda x: x if len(x.shape) == 3 else x[:, :, None]
right_time_size = (
    lambda t, x: t
    if isinstance(t, torch.Tensor)
    else torch.full((x.size(0),), t).to(x.device)
)
