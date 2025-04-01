import minari
import torch
import statistics

from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from config import batch_size

# ---------------------------------------------------------------------------------------
# ---------------------------------------- Functions ------------------------------------
# ---------------------------------------------------------------------------------------

class FixedLengthCollate:
    def __init__(self):
        self.fixed_length = 70  

    def __call__(self, batch):
        """
        Pads the trajectories to the same length, determined by the mean length of the first batch,
        and keeps only xpos and ypos.
        """
        observations = [torch.as_tensor(x.observations['observation'])[:, :2] for x in batch]  # Keep only xpos, ypos
        lengths = torch.tensor([obs.shape[0] for obs in observations])  # Lengths of the trajs, stored in tensor

        if self.fixed_length is None:
            self.fixed_length = int(round(statistics.mean(lengths.cpu().numpy())))  # If fixed_length = None, Store avg length of first batch to use

        # Pad sequences to the fixed length
        padded_obs = torch.nn.utils.rnn.pad_sequence(observations, batch_first=True)
        
        # Trim down to fixed length
        padded_obs = padded_obs[:, :self.fixed_length]  

        # Extend sequences with last valid value if shorter than fixed_length
        for i, obs in enumerate(observations):
            if obs.shape[0] < self.fixed_length:
                padded_obs[i, obs.shape[0]:] = obs[-1]  # Repeat last valid value

        return {
            "observations": padded_obs
        }

def get_data_loader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=FixedLengthCollate(), drop_last=True)

# ---------------------------------------------------------------------------------------
# ----------------------------------- Initialization ------------------------------------
# ---------------------------------------------------------------------------------------


dataset = minari.load_dataset('D4RL/pointmaze/umaze-v2')
dataLoaderFM = get_data_loader(dataset, batch_size)

for batch in dataLoaderFM:
    test_batch = batch['observations']
    print(f"test_batch.shape: {test_batch.shape}")
    break
