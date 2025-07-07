import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class ContinuousSolver:
    def __init__(self, config, model):
        self.method = config.model.solver_continuous
        self.model = model

    def fwd_step(self, state, delta_t):
        if state.has_continuous:
            if self.method == "euler":
                return self.euler_step(state, delta_t)

            elif self.method == "euler_maruyama":
                return self.euler_maruyama_step(state, delta_t)
        else:
            return state

    def euler_step(self, state, delta_t):
        heads = self.model(state)
        drift = heads.continuous
        state.continuous += delta_t * drift
        return state

    def euler_maruyama_step(self, state, delta_t):
        heads = self.model(state)
        diffusion = self.model.bridge_continuous.diffusion(state)
        drift = heads.continuous
        delta_w = torch.randn_like(state.continuous)
        state.continuous += delta_t * drift + diffusion * delta_w
        return state


class DiscreteSolver:
    def __init__(self, model, vocab_size, method='tauleap', topk=None):
        self.method = method
        self.vocab_size =vocab_size
        self.model = model
        self.topk = topk

    def fwd_step(self, state, delta_t, last_step):
        if state.has_discrete:
            if self.method == "tauleap":
                return self.tauleap_step(state, delta_t, last_step)

            elif self.method == "euler":
                return self.euler_step(state, delta_t, last_step)

            elif self.method == "midpoint":
                return self.midpoint_step(state, delta_t, last_step)

            elif self.method == "predictor_corrector":
                return self.predictor_corrector_step(state, delta_t, last_step)
        else:
            return state

    def tauleap_step(self, state, delta_t, last_step=False):
        """ - state.time (t): (B, 1) time tensor
            - state.discrete (k) : (B, D, 1) current state tensor
        """

        logits = self.model(state, batch=None).logits         
        rates = self.model.bridge_discrete.rate(state, logits)

        state.discrete = state.discrete.squeeze(-1)

        delta_n = torch.poisson(rates * delta_t).to(state.time.device)  # all jumps
        jump_mask = (torch.sum(delta_n, dim=-1).type_as(state.discrete) <= 1)  # for categorical data
        diff = torch.arange(self.vocab_size, device=state.time.device).view(1, 1, self.vocab_size) - state.discrete[:, :, None]  
        net_jumps = torch.sum(delta_n * diff, dim=-1).type_as(state.discrete)

        state.discrete = (state.discrete + net_jumps * jump_mask) % self.vocab_size
        state.discrete = state.discrete.unsqueeze(-1)

        # if last_step:
        #     max_rate = torch.max(rates, dim=2)[1]
        #     state.discrete = max_rate.unsqueeze(-1)

        return state


    def euler_step(self, state, delta_t, last_step=False):
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
