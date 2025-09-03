import torch
import numpy as np
from torchdyn.datasets import generate_moons
import math

class NGaussians:
    def __init__(
        self,
        dim=2,
        num_gaussians=8,
        num_points_per_gaussian=1000,
        std_dev=0.1,
        scale=5,
        labels_as_state=False,
    ):
        self.dim = dim
        self.num_points_per_gaussian = num_points_per_gaussian
        self.num_gaussians = num_gaussians
        self.N = num_gaussians * num_points_per_gaussian
        self.std_dev = std_dev
        self.scale = scale
        self.continuous, labels = self.sample_N_concentric_gaussians()
        self.discrete = labels.unsqueeze(-1).long()

    def sample_N_concentric_gaussians(self):
        angle_step = 2 * np.pi / self.num_gaussians
        positions = []
        labels = []

        for i in range(self.num_gaussians):
            angle = i * angle_step
            center_x = np.cos(angle)
            center_y = np.sin(angle)
            normal = torch.distributions.multivariate_normal.MultivariateNormal(
                torch.zeros(self.dim), math.sqrt(self.std_dev) * torch.eye(self.dim)
            )
            points = normal.sample((self.num_points_per_gaussian,))
            points += np.array([center_x, center_y]) * self.scale
            positions.append(points)
            labels += [i % self.num_gaussians] * self.num_points_per_gaussian

        positions = np.concatenate(positions, axis=0)
        positions = torch.tensor(positions, dtype=torch.float32)
        labels = np.array(labels)
        labels = torch.tensor(labels)
        idx = torch.randperm(len(labels))
        positions = positions[idx]
        labels =  labels[idx] + 1
    
        return positions, torch.tensor(labels, dtype=torch.long)

    def display(self, num_points=None, ax=None, **kwargs):
        num_points = self.N if num_points is None else num_points
        c = (
            self.discrete[:num_points]
            if hasattr(self, "discrete")
            else (self.context[:num_points] if hasattr(self, "context") else None)
        )
        ax.scatter(
            self.continuous[:num_points, 0],
            self.continuous[:num_points, 1],
            c=c,
            **kwargs,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("equal")

    def __len__(self):
        assert self.continuous.shape[0] == self.N
        return self.N


class TwoMoons:
    def __init__(
        self,
        dim=2,
        num_points_per_moon=1000,
        std_dev=0.2,
        labels_as_state=False,
    ):
        self.dim = dim
        self.num_points_per_moon = num_points_per_moon
        self.N = num_points_per_moon * 2
        self.std_dev = std_dev
        self.continuous, labels = self.sample_moons()
        self.discrete = labels.unsqueeze(-1).long()

    def sample_moons(self):
        positions, labels = generate_moons(self.N, noise=self.std_dev)
        idx = torch.randperm(len(labels))
        positions = positions[idx]
        labels = labels[idx] + 1 
        return positions * 3 - 1, torch.tensor(labels, dtype=torch.long)

    def display(self, num_points=None, ax=None, **kwargs):
        num_points = self.N if num_points is None else num_points
        c = (
            self.discrete[:num_points]
            if hasattr(self, "discrete")
            else (self.context[:num_points] if hasattr(self, "context") else None)
        )
        ax.scatter(
            self.continuous[:num_points, 0],
            self.continuous[:num_points, 1],
            c=c,
            **kwargs,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("equal")

    def __len__(self):
        assert self.continuous.shape[0] == self.N
        return self.N