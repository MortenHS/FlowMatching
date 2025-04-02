import torch
from tqdm import tqdm

from models.model import TrajectoryFlowModel
from data.dataset import get_data_loader
from data.utils import get_device
from data.config import SIGMA_INIT, NUM_EPOCHS, SAVE_PATH, LR, TRAJ_DIM
from visualize import vis_losses

# ---------------------------------------------------------------------------------------
# ---------------------------------------- Functions ------------------------------------
# ---------------------------------------------------------------------------------------

def compute_mu_t(x1, t): # Eq. 3a in Ye (t*tau_1)
    '''
    Calculates the mean, according to Eq.20 in Lipman 2023
    '''
    # Diffusion Cond. VFs
    # del t
    # return t * x1 + (1 - t) * x0 

    # Optimal Transport
    return t * x1

def compute_sigma_t(t, sigma): # 
    '''
    Calculates the standard deviation, sigma, according to Eq. 20 in Lipman 2023
    '''
    return 1 - (1 - sigma) * t # sigma_min in Eq.20 ?
    
    # ---VFs:---
    # del t
    # return sigma

def compute_conditional_flow(x0, x1, t, xt, sigma):
    '''
    Calculates the conditional velocity field u_t defined by Eq. 21 in Lipman 2023
    The corresponding conditional flow: Eq.22
    '''
    # return x1 - x0 # For "normal" CFM, Eq. 3b in Ye

    # For Optimal Transport (OT):
    t_expanded = t[:, None, None].expand(-1, xt.shape[1], -1)
    return (x1 - (1 - sigma) * xt) / (1 - (1 - sigma) * t_expanded) # Eq. 21, Lipman 2023, 

def sample_conditional_pt(x0, x1, t, sigma): # Sample xt
    ''' 
    xt is the flow model (satisfies boundary conditions Eq 4.6 Lipman 2024)
    Equals conditional phi in Lipman 2024

    Draws a sample from probability path
    '''
    sigma_t = compute_sigma_t(t, sigma)
    sigma_t = sigma_t[:, None, None].expand(-1, x1.shape[1], -1)

    epsilon = torch.randn_like(x0).to(device=x0.device, dtype=torch.float32) # Skal kanskje bare være x0?

    t_expanded = t[:, None, None].expand(-1, x1.shape[1], -1)
    mu_t = compute_mu_t(x1, t_expanded)
   
    # return mu_t + sigma_t * epsilon # Eq. 4.50 in Lipman 2024?
    return mu_t + (1 - t_expanded) * x0

def sample_and_compute(x0, x1, t, sigma):
    
    xt = sample_conditional_pt(x0, x1, t, sigma) # conditional probability path
    ut = compute_conditional_flow(x0, x1, t, xt, sigma) # Velocity field

    return t, xt, ut


def show_untrained_model(model_type="flow_matching", DL=get_data_loader(), sigma_init=SIGMA_INIT, device=get_device(), traj_dim=TRAJ_DIM):
    sigma = sigma_init # How much noise is added when interpolating, calculated earlier
    
    if model_type == "flow_matching":
        flow_model = TrajectoryFlowModel().to(device)
    else:
        raise ValueError("Unsupported model type")

    print("Initial sigma:", sigma_init)

    for batch in tqdm(DL, desc=f"Training Progress"):
        observations = batch['observations'].to(device)
        x1 = observations
        x0 = torch.rand_like(x1).to(device)
        t = torch.rand((x1.shape[0],), device=device)

        t, xt, ut = sample_and_compute(x0, x1, t, sigma)

        t_expanded = t[:, None, None].expand(-1, xt.shape[1], -1)
        xt = torch.cat([xt, t_expanded], dim=-1).to(device, dtype=torch.float32)
        
        vt = flow_model(xt)

    return vt, xt, ut


def train_fm(model_type="flow_matching", DL=get_data_loader(), sigma=SIGMA_INIT, num_epochs=NUM_EPOCHS, lr=LR, device=get_device(), save_path=SAVE_PATH):
    
    if model_type == "flow_matching":
        flow_model = TrajectoryFlowModel().to(device)
    else:
        raise ValueError("Unsupported model type")

    optimizer = torch.optim.AdamW(flow_model.parameters(), lr)
    losses = []

    for epoch in range(num_epochs):
        for batch in tqdm(DL, desc=f"Training Progress"):
            observations = batch['observations'].to(device)
            x1 = observations
            x0 = torch.rand_like(x1).to(device)
            t = torch.rand((x1.shape[0],), device=device)

            t, xt, ut = sample_and_compute(x0, x1, t, sigma)

            t_expanded = t[:, None, None].expand(-1, xt.shape[1], -1)
            xt = torch.cat([xt, t_expanded], dim=-1).to(device, dtype=torch.float32)

            vt = flow_model(xt) # Eq 13/14 Lipman 2023
    
            loss = torch.mean((vt - ut) ** 2)
            loss = loss.to(device, dtype=torch.float32)
            loss.backward()
            losses.append(loss.item())

            optimizer.step()
            optimizer.zero_grad()
            
        # Save checkpoint after each epoch
        checkpoint = {
            'model_state_dict': flow_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses,
            'sigma': sigma,
            'xt' : xt,
            'vt': vt,
            'ut' : ut
        }

        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path} after epoch {epoch + 1}")

def load_checkpoint(model_type="flow_matching", lr=LR, save_path=SAVE_PATH, map_location=get_device()):
    
    if model_type == "flow_matching":
        flow_model = TrajectoryFlowModel().to(map_location)
    else:
        raise ValueError("Unsupported model type")

    optimizer = torch.optim.AdamW(flow_model.parameters(), lr)
    checkpoint = torch.load(save_path, map_location=map_location)

    flow_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    losses = checkpoint.get('losses', [])
    sigma = checkpoint['sigma']
    vt = checkpoint['vt']
    xt = checkpoint['xt']
    ut = checkpoint['ut']

    print("Checkpoint loaded successfully!")
    return flow_model, optimizer, losses, sigma, vt, xt, ut


if __name__ == '__main__':
    train_fm()