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

def sample_in_loop(sec_dim, flow_model, device, epoch, num_steps=10000, solver="euler"):
    from visualize import plot_trajectories
    node = NeuralODE(
        torch_wrapper(flow_model),
        solver=solver,
        sensitivity="adjoint",
        atol=1e-4,
        rtol=1e-4
    )

    with torch.no_grad():
        traj_observations = node.trajectory(
            single_gaussian_sample_alt(sec_dim),
            t_span = torch.linspace(0, 1, num_steps, device=device, dtype=torch.float32),
        )

    plot_trajectories(traj_observations, epoch)
    
    


# ------------------------------------------------------------------------------------------------------------------------
# Attempted implementation from TCFM
# ------------------------------------------------------------------------------------------------------------------------
# conditions =  [(time, state), ...] = [(t, (xpos, ypos)), ...]

# batch_collate_fn = lambda batch: self.collate_fn_repeat(batch, self.n_test_samples)
# self.test_dataloader = cycle(torch.utils.data.DataLoader(
#     self.test_dataset, batch_size=self.n_test_batch_size, num_workers=1, shuffle=True, pin_memory=True, collate_fn=batch_collate_fn))

# def pad_collate_repeat(batch, num_samples):
#     (data, global_cond, cond) = zip(*batch)
#     data = torch.tensor(np.stack(data, axis=0))
#     global_cond = torch.tensor(np.stack(global_cond, axis=0)).float()
#     # cond = torch.tensor(np.stack(cond, axis=0))

#     data = data.repeat(num_samples, 1, 1)
#     global_cond = global_cond.repeat(num_samples, 1, 1)
#     return data.float(), global_cond, cond * num_samples



if __name__ == '__main__':
    sample_fm()