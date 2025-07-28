import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from utils.tensorclass import TensorMultiModal

class HybridSolver:
    def __init__(self, model, config):
        self.method = 'euler-leap'
        self.vocab_size = config.vocab_size
        self.model = model
        self.T = config.temperature 
        self.top_k = config.top_k
        self.top_p = config.top_p

    def fwd_step(self, state, delta_t) -> TensorMultiModal:
        if self.method == "euler-leap":
            return self.euler_leap_step(state, delta_t)

    def euler_leap_step(self, state: TensorMultiModal, delta_t: torch.Tensor) -> TensorMultiModal:
        """ - state.time (t): (B, 1) time tensor
            - state.discrete (k) : (B, D, 1) current state tensor
        """

        vt, logits = self.model(state)      

        if self.T != 1.0: 
            logits = self._temperature_scheduler(logits, state.time)
        
        probs = F.softmax(logits, dim=-1)

        if self.top_k is not None: 
            probs = self._top_k_filter(probs)

        if self.top_p is not None: 
            probs = self._top_p_filter(probs)

        rates = self.model.bridge_discrete.rate(state, probs)     # (B, D, vocab_size)

        assert rates.shape == logits.shape, "Rates and logits must have the same shape."
        state.discrete = state.discrete.squeeze(-1)

        delta_n = torch.poisson(rates * delta_t).to(state.time.device)  # all jumps
        jump_mask = (torch.sum(delta_n, dim=-1).type_as(state.discrete) <= 1)  # for categorical data
        diff = torch.arange(self.vocab_size, device=state.time.device).view(1, 1, self.vocab_size) - state.discrete[:, :, None]  
        net_jumps = torch.sum(delta_n * diff, dim=-1).type_as(state.discrete)

        # leap step
        state.discrete = (state.discrete + net_jumps * jump_mask) % self.vocab_size
        state.discrete = state.discrete.unsqueeze(-1)

        # euler step
        state.continuous = state.continuous + vt * delta_t 

        # if last_step:
        #     max_rate = torch.max(rates, dim=2)[1]
        #     state.discrete = max_rate.unsqueeze(-1)

        return state

    # rate engeneering methods:

    def _temperature_scheduler(self, logits, time, lam=0.5):
        scheduler = lambda t: torch.exp(-lam * t) 
        # scheduler = lambda t: torch.sigmoid(lam * (t - 0.5))
        temp = self.T / time # + (1.0 - self.T) * scheduler(time)
        temp = temp.view(-1, 1, 1).to(logits.device)
        return  logits / (temp + 1e-8)

    def _top_k_filter(self, probs):
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
        self.method = 'tauleap'
        self.vocab_size = config.vocab_size
        self.model = model
        self.T = config.temperature 
        self.top_k = config.top_k
        self.top_p = config.top_p

    # def __init__(self, model, vocab_size, method='tauleap', topk=None, temperature=1.0):
    #     self.method = method
    #     self.vocab_size = vocab_size
    #     self.model = model
    #     self.topk = topk
    #     self.temperature = temperature
    #     self.topk = topk 

    def fwd_step(self, state, delta_t) -> TensorMultiModal:
        if state.has_discrete:
            if self.method == "tauleap":
                return self.tauleap_step(state, delta_t)

            elif self.method == "euler":
                return self.euler_step(state, delta_t)

            elif self.method == "midpoint":
                return self.midpoint_step(state, delta_t)

            elif self.method == "predictor_corrector":
                return self.predictor_corrector_step(state, delta_t)
        else:
            return state


    def tauleap_step(self, state: TensorMultiModal, delta_t: torch.Tensor) -> TensorMultiModal:
        """ - state.time (t): (B, 1) time tensor
            - state.discrete (k) : (B, D, 1) current state tensor
        """

        logits = self.model(state)      

        if self.T != 1.0: 
            logits = self._temperature_scheduler(logits, state.time)
        
        probs = F.softmax(logits, dim=-1)

        if self.top_k is not None: 
            probs = self._top_k_filter(probs)

        if self.top_p is not None: 
            probs = self._top_p_filter(probs)

        rates = self.model.bridge_discrete.rate(state, probs)     # (B, D, vocab_size)

        assert rates.shape == logits.shape, "Rates and logits must have the same shape."
        state.discrete = state.discrete.squeeze(-1)

        delta_n = torch.poisson(rates * delta_t).to(state.time.device)  # all jumps
        jump_mask = (torch.sum(delta_n, dim=-1).type_as(state.discrete) <= 1)  # for categorical data
        diff = torch.arange(self.vocab_size, device=state.time.device).view(1, 1, self.vocab_size) - state.discrete[:, :, None]  
        net_jumps = torch.sum(delta_n * diff, dim=-1).type_as(state.discrete)

        # leap 
        state.discrete = (state.discrete + net_jumps * jump_mask) % self.vocab_size
        state.discrete = state.discrete.unsqueeze(-1)

        return state


    # def tauleap_step(self, state: TensorMultiModal, delta_t: torch.Tensor, last_step: bool=False) -> TensorMultiModal:
    #     """ - state.time (t): (B, 1) time tensor
    #         - state.discrete (k) : (B, D, 1) current state tensor
    #     """

    #     logits = self.model(state)   # (B, D, vocab_size)
    #     logits = temperature_scaling(logits, self.temperature)  # apply temperature scaling

    #     assert rates.shape == logits.shape, "Rates and logits must have the same shape."
    #     state.discrete = state.discrete.squeeze(-1)

    #     delta_n = torch.poisson(rates * delta_t).to(state.time.device)  # all jumps
    #     jump_mask = (torch.sum(delta_n, dim=-1).type_as(state.discrete) <= 1)  # for categorical data
    #     diff = torch.arange(self.vocab_size, device=state.time.device).view(1, 1, self.vocab_size) - state.discrete[:, :, None]  
    #     net_jumps = torch.sum(delta_n * diff, dim=-1).type_as(state.discrete)

    #     state.discrete = (state.discrete + net_jumps * jump_mask) % self.vocab_size
    #     state.discrete = state.discrete.unsqueeze(-1)

    #     if last_step:
    #         max_rate = torch.max(rates, dim=2)[1]
    #         state.discrete = max_rate.unsqueeze(-1)

    #     return state


    def euler_step(self, state: TensorMultiModal, delta_t: torch.Tensor, last_step: bool=False) -> TensorMultiModal:
        heads = self.model(state, batch=None) 
        rates = self.model.bridge_discrete.rate(state, heads)

        # off diagonal probs:
        state.discrete = state.discrete.squeeze(-1)
        delta_p = (rates * delta_t).clamp(max=1.0)

        # diagonal probs:
        delta_p.scatter_(-1, state.discrete[:, :, None], 0.0)
        delta_p.scatter_(
            -1,
            state.discrete[:, :, None],
            (1.0 - delta_p.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )
        if last_step:
            if self.topk:
                topk_vals, topk_idx = torch.topk(delta_p, self.topk, dim=-1)
                topk_probs = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
                sampled_topk = Categorical(topk_probs).sample()
                state.discrete = torch.gather(topk_idx, -1, sampled_topk.unsqueeze(-1))
            else:
                state.discrete = Categorical(delta_p).sample().unsqueeze(-1)

        return state

    def midpoint_step(self, state,  delta_t):
        heads = self.model(state)
        rates = self.model.bridge_discrete.rate(state, heads)

        state_mid = state.clone()
        state_mid = self.euler_step(state_mid, rates, 0.5 * delta_t)
        state = self.euler_step(state_mid, rates, delta_t)
        del state_mid
        return state, rates

    def predictor_corrector_step(self, state, delta_t, max_iter=10, tol=1e-3):
        pred_state = state.clone()
        pred_state = self.tauleap_step(pred_state, delta_t)  # First estimate
        pred_state.time += delta_t  # Update time

        # Compute transition probabilities at predicted state
        heads = self.model(pred_state)
        rates_next = self.model.bridge_discrete.rate(pred_state, heads)
        delta_p = (rates_next * delta_t).clamp(max=1.0)

        delta_p.scatter_(-1, pred_state.discrete, 0.0)
        delta_p.scatter_(
            -1,
            pred_state.discrete,
            (1.0 - delta_p.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )

        # Corrector Step: Iterative Refinement
        state_corrected = pred_state.clone()
        heads = self.model(state_corrected)
        rates_corrected = self.model.bridge_discrete.rate(state_corrected, heads)
        for i in range(max_iter):
            heads = self.model(state_corrected)
            rates_corrected = self.model.bridge_discrete.rate(state_corrected, heads)

            new_delta_p = (rates_corrected * delta_t).clamp(max=1.0)
            new_delta_p.scatter_(-1, state_corrected.discrete, 0.0)
            new_delta_p.scatter_(
                -1,
                state_corrected.discrete,
                (1.0 - new_delta_p.sum(dim=-1, keepdim=True)).clamp(min=0.0),
            )

            tvd = torch.abs(new_delta_p - delta_p).sum(dim=-1).mean()
            if tvd < tol:
                break  # Stop iterating if probabilities stabilize

            # Update corrected state
            state_corrected.discrete = Categorical(new_delta_p).sample()
            delta_p = new_delta_p  # Update reference probability

        return state_corrected, rates_corrected

    # rate engeneering methods:

    def _temperature_scheduler(self, logits, time, lam=0.5):
        scheduler = lambda t: torch.exp(-lam * t) 
        # scheduler = lambda t: torch.sigmoid(lam * (t - 0.5))
        temp = self.T / time # + (1.0 - self.T) * scheduler(time)
        temp = temp.view(-1, 1, 1).to(logits.device)
        return  logits / (temp + 1e-8)

    def _top_k_filter(self, probs):
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

    def euler_step(self, state: TensorMultiModal, delta_t: torch.Tensor) -> TensorMultiModal:
        vt = self.model(state)
        state.continuous += vt * delta_t 
        return state

    def euler_maruyama_step(self, state: TensorMultiModal, delta_t: torch.Tensor) -> TensorMultiModal:
        heads = self.model(state)
        diffusion = self.model.bridge_continuous.diffusion(state)
        vt = heads.continuous
        delta_w = torch.randn_like(state.continuous)
        state.continuous += delta_t * vt + diffusion * delta_w
        return state
