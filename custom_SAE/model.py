import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

def collate_token_batches(batch):
    # batch: list of [Lᵢ, 320]
    return torch.cat(batch, dim=0)  # output: [sum Lᵢ, 320]

class TokenRepresentationDataset(Dataset):
    def __init__(self, data):
        """
        data: (seq_ids , Tensor [B, 370, 320])
        """
        self.seq_ids = data[0]
        self.tokens = data[1]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx] # shape: 1, [370, 320]

class MatryoshkaSAE(nn.Module):
    def __init__(self, input_dim=320, hidden_dims=[64, 128, 256]):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(inplace=True)
        )

        # Decoders from nested hidden codes
        self.decoders = nn.ModuleList([
            nn.Linear(dim, input_dim) for dim in hidden_dims
        ])
        self.hidden_dims = hidden_dims
        self.latent_state = None

    def forward(self, x):
        """
        x: [B * 370, 320]
        Returns:
          code: latent code of size [B * 370, max_hidden_dim]
          recons: list of reconstructions from nested hidden subsets
        """
        z = self.encoder(x)  # [N, hidden_dims[-1]]
        recons = [decoder(z[:, :dim]) for dim, decoder in zip(self.hidden_dims, self.decoders)]
        return z, recons
