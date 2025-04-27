# Model Outputs for ESM‑2 8M

## Overview
This directory packages pre‑computed outputs from the ESM‑2 8 M parameter model for 1 645 UniRef50 protein sequences (<1000 aa). It also contains two companion dataframes with sequence‑level metadata and biochemical descriptors.

## Directory layout

```text
model_outputs_8M/
│
├── sample_outputs/              # per‑sequence model tensors (.pt)
│   ├── token_representations/   # (L × 320) hidden states
│   ├── sequence_representations/# (320,) mean‑pooled vectors
│   └── contact/                 # (L × L) contact probability matrices
│
├── master_dataframe.csv         # joined view linking every sequence to its files
└── UniRef50_n1645.csv           # raw sequences + Biopython descriptors
```

## CSV schemas

### `master_dataframe.csv`
Columns:

| column | description |
| ------ | ----------- |
| Name | Sequence identifier (portion of FASTA header before the first space) |
| Sequence | Amino‑acid sequence |
| Token Representations | relative path to the per‑residue tensor (`sample_outputs/token_representations/`) |
| Sequence Representations | relative path to the mean‑pooled embedding tensor (`sample_outputs/sequence_representations/`) |
| Contacts | relative path to the contact‑probability tensor (`sample_outputs/contact/`) |
| ID | Original FASTA identifier |
| Description | Full FASTA description line |
| Length | Sequence length (residues) |
| Num_Features | Count of Biopython annotation features, if present |
| Molecular Weight | Calculated molecular mass (Daltons) |
| Aromaticity | Fraction of aromatic residues |
| Instability Index | Guruprasad instability score |
| Flexibility | Mean flexibility score |
| GRAVY | Grand average of hydropathicity |
| Isoelectric Point | Theoretical pI |
| Charge at pH:7.0 | Net charge at physiological pH |

### `UniRef50_n1645.csv`
Contains: 

| column | description |
| ------ | ----------- |
| Name | Sequence identifier (portion of FASTA header before the first space) |
| Sequence | Amino‑acid sequence |
| Token Representations | relative path to the per‑residue tensor (`sample_outputs/token_representations/`) |
| Sequence Representations | relative path to the mean‑pooled embedding tensor (`sample_outputs/sequence_representations/`) |
| Contacts | relative path to the contact‑probability tensor (`sample_outputs/contact/`) |

## File formats
All tensors are stored as **PyTorch `.pt` files** saved with `torch.save`. Each file holds a single tensor of type `float32`.

Load any representation or contact map with:

```python
import torch

# token-level hidden states for sequence 0
tok = torch.load("sample_outputs/token_representations/UniRef50_A0A318QM10.pt")      # (L, 320)

# mean-pooled sequence embedding for sequence 0
seq = torch.load("sample_outputs/sequence_representations/UniRef50_A0A318QM10.pt")   # (320,)

# contact probability matrix for sequence 0
cmap = torch.load("sample_outputs/contact/UniRef50_A0A318QM10.pt")                   # (L, L)
```
```python

## Re‑generation

Future scripts will be added to support regeneration at different model tiers

## License
Dataset released under CC‑BY 4.0.  
ESM‑2 model © Meta AI, licensed under CC‑BY‑NC 4.0.

