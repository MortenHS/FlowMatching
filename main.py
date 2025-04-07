from training.train import train_fm, show_untrained_model, load_checkpoint
from sampling.sample import sample_fm
from visualize import vis_dataset_distribution, vis_two_trajs, vis_losses, plot_trajectories

'''
TODO:
Lag scripts for trening og sampling

Test forskjellige hyperparametere
'''

def main():
    # Training:
    train_fm("VF")
    flow_model, optimizer, losses, sigma, vt, xt, ut = load_checkpoint()
    vis_losses(losses)

    # Sampling:
    # flow_model, optimizer, losses, sigma, vt, xt, ut = load_checkpoint()
    # sec_dim = xt.shape[1]
    # traj = sample_fm(sec_dim)
    # plot_trajectories(traj)
    
    return None # Similar to void func in c/c++


if __name__ == '__main__':
    main()