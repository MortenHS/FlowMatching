import torch

# print(f"Torch version: {torch.__version__}")
# print(f"Torch cuda version: {torch.version.cuda}")
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

