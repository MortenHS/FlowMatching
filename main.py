from training.train import train_fm, show_untrained_model, load_checkpoint
from sampling.sample import sample_fm
from visualize import vis_dataset_distribution, vis_two_trajs, vis_losses

'''
TODO:
Check single_gaussian_sample_alt function for the use of sec_dim, obs_dim.
'''


def main():
    train_fm()
    flow_model, optimizer, losses, sigma, vt, xt, ut = load_checkpoint()
    vis_losses(losses)

    # vt_u, xt_s, ut_u = show_untrained_model()

    # Must define sec_dim
    # sample_fm(sec_dim)

    # vis_dataset_distribution(max_batches=20)
    # vis_two_trajs()
    
    return None

if __name__ == '__main__':
    main()