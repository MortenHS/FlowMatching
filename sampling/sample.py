from models.model import torch_wrapper
from data.config import SIGMA_INIT, NUM_EPOCHS, SAVE_PATH, LR, BATCH_SIZE, TRAJ_DIM
from data.utils import get_device
from models.model import TrajectoryFlowModel

import torch
from torchdyn.core import NeuralODE

def single_gaussian_sample_alt(sec_dim, batch_size=BATCH_SIZE, traj_dim=TRAJ_DIM, device=get_device(), var=1):
    """
    Generate samples from a single Gaussian distribution centered at the origin.

    Args:
        batch_size (int): Number of batches.
        sec_dim (int): Number of sections (sequence length).
        traj_dim (int): Dimensionality of each trajectory point.
        var (float): Variance of the Gaussian distribution.

    Returns:
        torch.Tensor: Tensor of shape (n, traj_dim) containing sampled points.
    """

    return torch.randn(batch_size, sec_dim, traj_dim, device=device, dtype=torch.float32) * var**0.5


def sample_fm(sec_dim, model_type="flow_matching", traj_dim=TRAJ_DIM, num_steps=10000, solver="euler", device=get_device()):
    
    if model_type == "flow_matching":
        flow_model = TrajectoryFlowModel().to(device)
    else:
        raise ValueError("Unsupported model type")

    node = NeuralODE(
        torch_wrapper(flow_model),
        solver=solver,
        sensitivity="adjoint",
        atol=1e-4,
        rtol=1e-4
    )

    with torch.no_grad():
        traj = node.trajectory(
            single_gaussian_sample_alt(sec_dim),
            t_span = torch.linspace(0, 1, num_steps, device=device, dtype=torch.float32),
        )
    return traj # traj[-1]

if __name__ == '__main__':
    sample_fm()