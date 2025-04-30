import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
import model

# functions
def train_sae(model, dataloader, epochs=10, lr=1e-3, l1_weight=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:  # batch: [B, 370, 320]
            inputs = batch.view(-1, 320)  # Flatten to [B*370, 320]

            z, recons = model(inputs)
            recon_loss = loss_fn(recons[-1], inputs)
            sparsity = l1_weight * torch.mean(torch.abs(z))
            loss = recon_loss + sparsity

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")


# load data
data_path = "data/model_outputs_8M_complete/"
master = False
df = None
if master:
    df = pd.read_csv(data_path + "master_attributes.csv")
else:
    df = pd.read_csv(data_path + "train.csv")

# try to load all data
seq_ids = list(df[["Name"]])
token_paths = df["Token Representations"]

token_reps = []

for path in token_paths:
    token_reps.append(torch.load("data/" + str(path)))
# token_reps = torch.stack(token_reps, dim=0)
# create splits
train_split_prop = 0.80
split_idx = int(len(token_reps) * train_split_prop)
train_token_reps = token_reps[:split_idx]
train_seq_ids = seq_ids[:split_idx]

test_token_reps = token_reps[split_idx:]
test_seq_ids = seq_ids[split_idx:]

# create datasets
train_dataset = model.TokenRepresentationDataset((train_seq_ids, train_token_reps))
test_dataset = model.TokenRepresentationDataset((test_seq_ids, test_token_reps))
# create dataloaders
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=model.collate_token_batches)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True, collate_fn=model.collate_token_batches)

# train
model_0 = model.MatryoshkaSAE(320, [64])
train_sae(model_0, train_loader, epochs=100)