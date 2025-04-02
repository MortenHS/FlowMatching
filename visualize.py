import os
import torch
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from collections import Counter

from data.dataset import get_data_loader
from data.config import FIXED_LENGTH

# ---------------------------------------------------------------------------------------
# ---------------------------------------- Functions ------------------------------------
# ---------------------------------------------------------------------------------------

def vis_dataset_distribution(max_batches, DL=get_data_loader(), save_dir='results/Dataset_plots'):
    observation_lst = []
    for count, batch in enumerate(DL):
        if count == max_batches:
            break
        obs = batch['observations']
        observation_lst.append(obs)

    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, 'dataset_distribution.png')

    plt.figure(figsize=(8, 6))
    for observations in observation_lst:
        plt.plot(observations[0, :, 0].cpu().numpy(), observations[0, :, 1].cpu().numpy(), color='g', alpha=0.7)
    plt.title('MazeTrajs')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()

    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Image saved in: {plot_path}")
    return observation_lst

def vis_two_trajs(DL=get_data_loader(), idx = 0, save_dir='results/Dataset_plots'):
    test_batch = next(iter(DL))["observations"]
    trajectory = test_batch[idx].cpu().numpy()
    trajectory_2 = test_batch[idx+1].cpu().numpy()
    traj_tuples = [tuple(point[:2]) for point in trajectory]  # Only (x, y) positions
    traj_tuples_2 = [tuple(point[:2]) for point in trajectory_2]  # Only (x, y) positions

    # Count occurrences
    counts = Counter(traj_tuples)
    counts_2 = Counter(traj_tuples_2)

    # Filter duplicates (points appearing more than once)
    duplicates = {point: count for point, count in counts.items() if count > 1}
    duplicates_2 = {point: count for point, count in counts_2.items() if count > 1}

    print(f"Total duplicate points 1: {len(duplicates)}, 2: {len(duplicates_2)}, the final one")
    print("Duplicate points and their counts: 1:", duplicates, " 2: ", duplicates_2)

    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, 'two_trajs.png')

    plt.figure(figsize=(10, 6))

    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='red', alpha=0.5, label="Trajectory", zorder=1)
    plt.scatter(trajectory[-1, 0], trajectory[-1, 1], marker='v', color='green',s=100, alpha=1, label="Final Pos", zorder=2)

    plt.plot(trajectory_2[:, 0], trajectory_2[:, 1], marker='o', color='blue', alpha=0.5, label="Trajectory 2", zorder=1)
    plt.scatter(trajectory_2[-1, 0], trajectory_2[-1, 1], marker='v', color='maroon',s=100, alpha=1, label="Final Pos", zorder=2)

    plt.title('Positions in Trajectory')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Image saved in: {plot_path}")

def vis_losses(losses, save_dir='results/Plotted loss'):
    avg_loss = sum(losses)/len(losses)
    med_loss = stat.median(losses)

    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, 'loss_plot.png')

    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Loss', marker='o')
    plt.axhline(y=avg_loss, color='r', linestyle='--', label=f'Average Loss: {avg_loss:.4f}')
    plt.axhline(y=med_loss, color='y', linestyle='--', label=f'Median Loss: {med_loss:.4f}')
    plt.title('Loss Progression FM')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Image saved in: {plot_path}")


def plot_trajectories(traj, sec_dim=FIXED_LENGTH, save_dir='results/Sample_plots'):
    """Plot trajectories of some selected samples."""
    
    n = sec_dim
    if isinstance(traj, torch.Tensor):
        traj = traj.cpu().numpy()
    
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, 'sample_trajs.png')

    plt.figure(figsize=(8, 8))
    plt.scatter(traj[0, 0, :n, 0], traj[0, 0, :n, 1], s=10, alpha=0.8, c="black", label="Noisy Sample", zorder=2)
    plt.scatter(traj[0, :, :n, 0], traj[0, :, :n, 1], s=1, alpha=0.2, c="red", label="Flow", zorder=1)
    plt.scatter(traj[0, -1, :n, 0], traj[0, -1, :n, 1], s=10, alpha=1, c="blue", label="Target Sample", zorder=2)
    
    plt.legend(loc='upper left')
    plt.title("Flow Matching Evolution")
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Image saved in: {plot_path}")


if __name__ == '__main__':
    vis_dataset_distribution(max_batches=20)
    vis_two_trajs()