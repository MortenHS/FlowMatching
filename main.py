from training.train import train_fm, show_untrained_model, load_checkpoint
from sampling.sample import sample_fm
from visualize import vis_dataset_distribution, vis_two_trajs, vis_losses, plot_trajectories

'''
TODO:
Lag scripts for trening og sampling

'''

def main():
    # train_fm()
    flow_model, optimizer, losses, sigma, vt, xt, ut = load_checkpoint()
    # vis_losses(losses)
    # observation_lst = vis_dataset_distribution(max_batches=20)
    # vt_u, xt_s, ut_u = show_untrained_model()
    sec_dim = xt.shape[1]
    traj = sample_fm(sec_dim)
    # print("traj: ", traj[0][-1])
    # print("traj[0, -1, :sec_dim, 0]: \n", traj[0, -1, :sec_dim, 0], "\ntraj[0, -1, :sec_dim, 1]: \n", traj[0, -1, :sec_dim, 1])

    plot_trajectories(traj)
    
    # vis_two_trajs()
    
    return None


if __name__ == '__main__':
    main()