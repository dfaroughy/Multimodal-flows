import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from utils.tensorclass import TensorMultiModal

class HybridSolver:
    def __init__(self, model, config):
        self.method = 'tauleap'
        self.vocab_size = config.vocab_size
        self.model = model
        self.T = config.temperature 
        self.top_k = config.top_k
        self.top_p = config.top_p

    def fwd_step(self, state, delta_t) -> TensorMultiModal:
        if self.method == "tauleap":
            return self.tauleap_step(state, delta_t)
        elif self.method == "euler":
            return self.euler_step(state, delta_t)

    @torch.no_grad()
    def tauleap_step(self, state: TensorMultiModal, delta_t: torch.Tensor) -> TensorMultiModal:
        """ - state.time (t): (B, 1) time tensor
            - state.discrete (k) : (B, D, 1) current state tensor
        """
        self.model.eval()
        vt, logits = self.model(state)      

        if self.T != 1.0: 
            logits = self._temperature_scheduler(logits, state.time)
        
        probs = F.softmax(logits, dim=-1)

        if self.top_k is not None: 
            probs = self._top_k_filter(probs)

        if self.top_p is not None: 
            probs = self._top_p_filter(probs)

        rates = self.model.bridge_discrete.rate(state, probs)     # (B, D, vocab_size)

        # tau leaping:
        state.discrete = state.discrete.squeeze(-1)
        delta_n = torch.poisson(rates * delta_t).to(state.time.device)  # all jumps
        jump_mask = (torch.sum(delta_n, dim=-1).type_as(state.discrete) <= 1)  # for categorical data
        diff = torch.arange(self.vocab_size, device=state.time.device).view(1, 1, self.vocab_size) - state.discrete[:, :, None]  
        net_jumps = torch.sum(delta_n * diff, dim=-1).type_as(state.discrete)

        # leap step
        state.discrete = (state.discrete + net_jumps * jump_mask) % self.vocab_size
        state.discrete = state.discrete.unsqueeze(-1)

        # ODE euler step
        state.continuous = state.continuous + vt * delta_t 

        return state, rates

    @torch.no_grad()
    def euler_step(self, state: TensorMultiModal, delta_t: torch.Tensor) -> TensorMultiModal:
        self.model.eval()

        vt, logits = self.model(state)      

        if self.T != 1.0: 
            logits = self._temperature_scheduler(logits, state.time)
        
        probs = F.softmax(logits, dim=-1)
        rates = self.model.bridge_discrete.rate(state, probs)     # (B, D, vocab_size)

        # off diagonal probs:
        state.discrete = state.discrete.squeeze(-1)
        delta_p = (rates * delta_t).clamp(max=1.0)

        # diagonal probs:
        delta_p.scatter_(-1, state.discrete[:, :, None], 0.0)
        delta_p.scatter_(-1, state.discrete[:, :, None], (1.0 - delta_p.sum(dim=-1, keepdim=True)).clamp(min=0.0),)

        if self.top_k is not None: 
            delta_p = self._top_k_filter(delta_p)

        if self.top_p is not None: 
            delta_p = self._top_p_filter(delta_p)

        state.discrete   = Categorical(delta_p).sample().unsqueeze(-1)
        state.continuous = state.continuous + vt * delta_t 

        return state, rates

    # rate engeneering methods:

    def _temperature_scheduler(self, logits, time, lam=0.5):
        scheduler = lambda t: torch.exp(-lam * t) 
        # scheduler = lambda t: torch.sigmoid(lam * (t - 0.5))
        temp = self.T # (1.0 - self.T) * scheduler(time)
        # temp = temp.view(-1, 1, 1).to(logits.device)
        return  logits / (temp + 1e-8)

    def _top_k_filter(self, probs):
        if self.top_k == self.vocab_size:
            return probs
        else:
            _, idx = torch.topk(probs, self.top_k, dim=-1)
            mask = torch.zeros_like(probs, device=probs.device).scatter_(-1, idx, 1.0)
            probs = probs * mask.to(probs.dtype)
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
            return probs

    def _top_p_filter(self, probs):
        probs_sort, idx = torch.sort(probs, dim=-1, descending=True)
        probs_cum = probs_sort.cumsum(dim=-1)
        mask = probs_cum <= self.top_p
        mask[..., 0] = 1
        mask = torch.zeros_like(probs, device=probs.device).scatter(-1, idx, mask.float())
        probs = probs * mask.to(probs.dtype)
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        return probs


class DiscreteSolver:

    def __init__(self, model, config):
        self.method = config.markov_jump_solver
        self.vocab_size = config.vocab_size
        self.model = model
        self.T = config.temperature 
        self.top_k = config.top_k
        self.top_p = config.top_p

    def fwd_step(self, state, delta_t) -> TensorMultiModal:

        if self.method == "tauleap-poisson":
            return self.tauleap_step(state, delta_t, mode='poisson')

        elif self.method == "tauleap-bernouilli":
            return self.tauleap_step(state, delta_t, mode='bernoulli')

        elif self.method == "euler":
            return self.euler_step(state, delta_t)

        elif self.method == "jump_or_stay":
            return self.jump_or_stay_step(state, delta_t)

    @torch.no_grad()
    def tauleap_step(self, state: TensorMultiModal, delta_t: torch.Tensor, mode='poisson') -> TensorMultiModal:
        """ - state.time (t): (B, 1) time tensor
            - state.discrete (k) : (B, D, 1) current state tensor
        """

        self.model.eval()
        logits = self.model(state)      

        if self.T != 1.0: 
            logits = logits / self.T
        
        probs = F.softmax(logits, dim=-1)
        rates = self.model.bridge_discrete.rate(state, probs)     # (B, D, vocab_size)
        state.discrete = state.discrete.squeeze(-1)

        if mode == 'poisson':
            delta_n = torch.poisson(rates * delta_t).to(state.time.device)  # all jumps
            jump_mask = (delta_n.sum(dim=-1).type_as(state.discrete) <= 1)  # for categorical data
            diff = torch.arange(self.vocab_size, device=state.time.device).view(1, 1, self.vocab_size) - state.discrete[:, :, None]  
            net_jumps = torch.sum(delta_n * diff, dim=-1).type_as(state.discrete)
            state.discrete = (state.discrete + net_jumps * jump_mask) % self.vocab_size
            state.discrete = state.discrete.unsqueeze(-1)
            return state, rates


        elif mode == 'bernoulli':
            prob_jump = (rates * delta_t).clamp(max=1.0)                
            delta_n = torch.bernoulli(prob_jump)        
            diff = torch.arange(self.vocab_size, device=state.time.device).view(1, 1, self.vocab_size) - state.discrete[:, :, None]  
            net_jumps = torch.sum(delta_n * diff, dim=-1).type_as(state.discrete)
            state.discrete = (state.discrete + net_jumps) % self.vocab_size
            state.discrete = state.discrete.unsqueeze(-1)

            return state, rates

    def euler_step(self, state: TensorMultiModal, delta_t: torch.Tensor) -> TensorMultiModal:
        self.model.eval()

        logits = self.model(state)      

        if self.T != 1.0: 
            logits = self._temperature_scheduler(logits, state.time)
        
        probs = F.softmax(logits, dim=-1)
        rates = self.model.bridge_discrete.rate(state, probs)     # (B, D, vocab_size)

        # off diagonal probs:
        state.discrete = state.discrete.squeeze(-1)
        delta_p = (rates * delta_t).clamp(max=1.0)

        # diagonal probs:
        delta_p.scatter_(-1, state.discrete[:, :, None], 0.0)
        delta_p.scatter_(-1, state.discrete[:, :, None], (1.0 - delta_p.sum(dim=-1, keepdim=True)).clamp(min=0.0),)
        state.discrete   = Categorical(delta_p).sample().unsqueeze(-1)

        return state, rates

    @torch.no_grad()
    def jump_or_stay_step(self, state: TensorMultiModal, delta_t: torch.Tensor) -> TensorMultiModal:
        """Binomial tau-leaping step adapted for discrete class labels.
        Allows at most one jump per particle. Transitions are drawn via:
        - Bernoulli for whether a particle jumps,
        - Categorical for which class it jumps to (excluding the current one).
        """
        self.model.eval()
        logits = self.model(state)  # (B, D, vocab_size)

        if self.T != 1.0:
            logits = self._temperature_scheduler(logits, state.time)

        probs = F.softmax(logits, dim=-1)
        rates = self.model.bridge_discrete.rate(state, probs)  # (B, D, vocab_size)
        state.discrete = state.discrete.squeeze(-1)  # (B, D)

        # Extract the rate to leave the current state
        state_current = state.discrete.long()  # (B, D)
        rate_leave = torch.gather(rates, dim=2, index=state_current.unsqueeze(-1)).squeeze(-1)  # (B, D)

        # Compute Bernoulli probs for jumping
        p_leave = (rate_leave * delta_t).clamp(max=1.0)
        jump_mask = torch.bernoulli(p_leave).bool()  # (B, D)

        # Build categorical over destination classes (excluding current class)
        probs_ = probs.clone()
        probs_.scatter_(-1, state_current.unsqueeze(-1), 0.0)
        probs_ = probs_ / probs_.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Sample new target states
        state_target = Categorical(probs_).sample()  # (B, D)
        updated_state = torch.where(jump_mask, state_target, state_current)
        state.discrete = updated_state.unsqueeze(-1)  # (B, D, 1)

        return state, rates

class ContinuousSolver:
    def __init__(self, model, config):
        self.method = 'euler'
        self.model = model

    def fwd_step(self, state: TensorMultiModal, delta_t: torch.Tensor) -> TensorMultiModal:
        if state.has_continuous:
            if self.method == "euler":
                return self.euler_step(state, delta_t)

            elif self.method == "euler_maruyama":
                return self.euler_maruyama_step(state, delta_t)
        else:
            return state

    @torch.no_grad()
    def euler_step(self, state: TensorMultiModal, delta_t: torch.Tensor) -> TensorMultiModal:
        self.model.eval()
        vt = self.model(state)
        state.continuous += vt * delta_t 
        return state

    @torch.no_grad()
    def euler_maruyama_step(self, state: TensorMultiModal, delta_t: torch.Tensor) -> TensorMultiModal:
        self.model.eval()
        heads = self.model(state)
        diffusion = self.model.bridge_continuous.diffusion(state)
        vt = heads.continuous
        delta_w = torch.randn_like(state.continuous)
        state.continuous += delta_t * vt + diffusion * delta_w
        return state
