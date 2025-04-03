import torch

from data.utils import get_device
from data.config import SIGMA_INIT, NUM_EPOCHS, TRAJ_DIM, FIXED_LENGTH, BATCH_SIZE
from models.model import TrajectoryFlowModel
from training.train import sample_and_compute, compute_xt_ut

def test_fm(method, sigma=SIGMA_INIT, sec_dim=FIXED_LENGTH, traj_dim=TRAJ_DIM, model_type="flow_matching", batch_size=BATCH_SIZE, device=get_device()):
    
    if model_type == "flow_matching":
        flow_model = TrajectoryFlowModel().to(device)
    else:
        raise ValueError("Unsupported model type")
    
    x0, x1 = [torch.randn(batch_size, sec_dim, traj_dim), torch.randn(batch_size, sec_dim, traj_dim)]

    torch.manual_seed(1234)

    t = torch.rand((x1.shape[0],), device=device)
    t, xt, ut, eps = sample_and_compute(method, x0, x1, t, sigma)

    torch.manual_seed(1234)
    epsilon = torch.randn_like(x0)
    t_given_init = torch.rand(batch_size)
    t_given = t_given_init.reshape(-1, *([1] * (x0.dim() - 1)))

    comp_xt, comp_ut = compute_xt_ut(method, x0, x1, t_given, sigma, epsilon)

    assert torch.all(xt.eq(comp_xt))
    assert torch.all(ut.eq(comp_ut))
    assert torch.all(eps.eq(epsilon))

if __name__ == '__main__':
    # Run with: python -m tests.test_cfm
    test_fm("VF")