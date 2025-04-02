import torch
from torch import nn, Tensor
from data.utils import get_device
from data.config import TRAJ_DIM, LR, BATCH_SIZE, SIGMA_INIT
from data.dataset import get_data_loader


class TrajectoryFlowModel(nn.Module):
    def __init__(self, traj_dim=TRAJ_DIM, hidden_dim=512, num_layers=5): # Samme som Lipman
        """
        A neural network that estimates the velocity field for flow matching.

        Args:
            traj_dim (int): Dimensionality of observations [xpos, ypos]]
            hidden_dim (int): Number of hidden units in the MLP
            num_layers (int): Number of layers in the MLP
        """
        super().__init__()

        layers = []
        input_dim = traj_dim + 1  # We include time `t` as an input
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            # Alternating activation functions for better gradient flow and expressivity
            if i % 2 == 0:
                layers.append(nn.SiLU())  # Swish/SiLU for smoothness
            else:
                layers.append(nn.ReLU())  # ReLU for stability
            
            input_dim = hidden_dim 
        
        # OUTPUT LAYER
        layers.append(nn.Linear(hidden_dim, traj_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, xt):
        """
        Forward pass for the trajectory flow model.
        """
        velocity = self.network(xt) 
        return velocity

class torch_wrapper(nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):

        # t is [] going in, so we use torch.full to create the correct shape and fill it.

        t_expanded = torch.full((x.shape[0], x.shape[1], 1), t.item(), device=x.device, dtype=x.dtype)
        xt = torch.cat([x, t_expanded], dim=-1)  # Concatenating time to match model input

        return self.model(xt)

def check_gradients_and_shape(flow_model, device, batch_size=BATCH_SIZE, traj_dim=TRAJ_DIM):
    xt = torch.randn(batch_size, traj_dim + 1).to(device, dtype=torch.float32)  # Adding time `t` as an input feature
    
    # One pass
    velocity_pred = flow_model(xt)

    # Shapes:
    print(f"Input shape: {xt.shape}")
    print(f"Output shape: {velocity_pred.shape}")  

    target_velocity = torch.randn_like(velocity_pred)

    # Loss
    loss_fn = nn.MSELoss()
    loss = loss_fn(velocity_pred, target_velocity)
    loss.backward()

    for name, param in flow_model.named_parameters():
        if param.grad is None:
            print(f"Gradient not computed for {name}")
        else:
            print(f"Gradient exists for {name} with shape {param.grad.shape}")


def test_fwd_pass(device, sigma=SIGMA_INIT, DL=get_data_loader()):
    from training.train import sample_and_compute

    batch = next(iter(DL))
    observations = batch['observations'].to(device)
    x1 = observations
    x0 = torch.rand_like(x1).to(device)
    t = torch.rand((x1.shape[0],), device=device)

    print(f"Shapes of variables x1, x0 and t before sample_and_compute: {x1.shape}, {x0.shape}, {t.shape}")
    t, xt, ut = sample_and_compute(x0, x1, t, sigma)
    print(f"Shapes of variables t, xt and ut from sample_and_compute: {t.shape}, {xt.shape}, {ut.shape}")
    t_expanded = t[:, None, None].expand(-1, xt.shape[1], -1)
    xt = torch.cat([xt, t_expanded], dim=-1).to(device, dtype=torch.float32)
    print(f"Shape of xt post torch.cat: {xt.shape}")
    vt_p = flow_model(xt)
    print(f"vt_p.shape: {vt_p.shape}")

    return vt_p

if __name__ == '__main__':
    device = get_device()
    flow_model = TrajectoryFlowModel().to(device)
    
    # to run use: "python -m models.model"
    vt = test_fwd_pass(device)

    
    # check_gradients_and_shape(flow_model, device)
    # optimizer = torch.optim.AdamW(flow_model.parameters(), lr=LR)

