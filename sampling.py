from model import torch_wrapper
from config import sigma_init, num_epochs, save_path, lr

import torch
from torchdyn.core import NeuralODE

def single_gaussian_sample_alt(sec_dim, sigma, batch_size=batch_size, obs_dim=obs_dim, device=device, var=1.0):
    """
    Generate samples from a single Gaussian distribution centered at the origin.

    Args:
        batch_size (int): Number of batches.
        sec_dim (int): Number of sections (sequence length).
        obs_dim (int): Dimensionality of each trajectory point.
        var (float): Variance of the Gaussian distribution.

    Returns:
        torch.Tensor: Tensor of shape (n, dim) containing sampled points.
    """

    return torch.randn(batch_size, sec_dim, obs_dim, device=device, dtype=torch.float32) * var**0.5


def sample(flow_model, batch_size, shape, sigma, obs_dim=obs_dim, num_steps=10000, solver="euler"):
    
    node = NeuralODE(
        torch_wrapper(flow_model),
        solver=solver,
        sensitivity="adjoint",
        atol=1e-4,
        rtol=1e-4
    )

    with torch.no_grad():
        traj = node.trajectory(
            single_gaussian_sample_alt(sec_dim, sigma),
            t_span = torch.linspace(0, 1, num_steps, device=device, dtype=torch.float32),
        )
    return traj # traj[-1]