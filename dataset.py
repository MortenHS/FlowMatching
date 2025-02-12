# Test datasets
import gymnasium as gym
import gymnasium_robotics
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# gym.register_envs(gymnasium_robotics)

# env = gym.make("PointMaze_UMaze-v3")
# observations = []
# trajectories = []
# max_trajs = 10000
# for _ in range(max_trajs):
#     for _ in tqdm(range(max_trajs)):
#         observation, info = env.reset()
#         episode_over = False
#         while not episode_over:
#             action = env.action_space.sample()
#             observation, reward, terminated, truncated, info = env.step(action)
#             observations.append(observation['observation'][:2])
#             episode_over = terminated or truncated
#         trajectories.append(np.array(observations))

# torch.save(trajectories, 'trajectories.pth')

# Register and create environment
gym.register_envs(gymnasium_robotics)
env = gym.make("PointMaze_UMaze-v3")

trajectories = []
max_trajs = 10000

for _ in tqdm(range(max_trajs)):
    observations = []  # Reset observations per trajectory
    times = []  # Store time steps
    observation, info = env.reset()
    
    episode_over = False
    t = 0  # Initialize time step
    while not episode_over:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        observations.append(observation['observation'][:2])  # Store position
        times.append(t)  # Store time step

        episode_over = terminated or truncated
        t += 1  # Increment time

    # Save (trajectory, time steps) tuple
    trajectories.append((np.array(observations), np.array(times)))

# Save the full dataset (including time steps)
torch.save(trajectories, 'trajectories.pth')
print(f"Lenght of trajectories dataset: {len(trajectories)}")


# # Plot the points
# for i, observations in enumerate(trajectories):
#     plt.plot(observations[:, 0], observations[:, 1], marker='o', label=f"Trajectory {i+1}")
#     plt.scatter(observations[0, 0], observations[0, 1], color='green', s=100, label='Start' if i == 0 else "")
#     plt.scatter(observations[-1, 0], observations[-1, 1], color='red', s=100, label='End' if i == 0 else "")

# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Trajectories of the agent')
# plt.legend()
# plt.grid()
# plt.show()

# Save the trajectories to a .pth file