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
    
    x0, x1 = [torch.randn(batch_size, sec_dim, traj_dim, device=device), torch.randn(batch_size, sec_dim, traj_dim, device=device)]

    torch.manual_seed(1234)

    t = torch.rand((batch_size), device=device)
    t, xt, ut, eps = sample_and_compute(method, x0, x1, t, sigma)

    torch.manual_seed(1234)
    epsilon = torch.randn_like(x0)

    assert torch.all(eps.eq(epsilon))
    
    torch.manual_seed(1234)
    t_given_init = torch.rand((batch_size), device=device)
    t_given = t_given_init[:, None, None].expand(-1, x0.shape[1], -1)

    comp_xt, comp_ut = compute_xt_ut(method, x0, x1, t_given, sigma, epsilon)
    
    assert any(t_given_init == t)
    assert torch.all(xt.eq(comp_xt))
    assert torch.all(ut.eq(comp_ut))
    

if __name__ == '__main__':
    # Run with: python -m tests.test_cfm
    test_fm("VF")