import torch
from torch import nn, Tensor

from config import obs_dim, lr
from utils import device

class TrajectoryFlowModel(nn.Module):
    def __init__(self, obs_dim, hidden_dim=512, num_layers=5): # Samme som Lipman
        """
        A neural network that estimates the velocity field for flow matching.

        Args:
            obs_dim (int): Dimensionality of observations [xpos, ypos]]
            hidden_dim (int): Number of hidden units in the MLP
            num_layers (int): Number of layers in the MLP
        """
        super().__init__()

        layers = []
        input_dim = obs_dim + 1  # We include time `t` as an input
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            # Alternating activation functions for better gradient flow and expressivity
            if i % 2 == 0:
                layers.append(nn.SiLU())  # Swish/SiLU for smoothness
            else:
                layers.append(nn.ReLU())  # ReLU for stability
            
            input_dim = hidden_dim 
        
        layers.append(nn.Linear(hidden_dim, obs_dim))  # Output layer, no activation
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


flow_model = TrajectoryFlowModel(obs_dim).to(device)
optimizer = torch.optim.AdamW(flow_model.parameters(), lr=lr)

