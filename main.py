
import torch
from transformers import EsmForMaskedLM, AutoTokenizer, AutoModelForMaskedLM
from dataclasses import dataclass
from typing import Optional, Tuple
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

@dataclass
class ModelOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor

class HookedESM:
    def __init__(self, model: EsmForMaskedLM):
        self.model = model
        self.n_layers = len(model.esm.encoder.layer)
        
    def __get_layer_info(self, layer_idx: int) -> Tuple[str, str]:
        """Determine layer name and override parameter based on layer index.
        
        This function handles two intervention cases that align with ESM2's representation indexing.
        For reference, using ESM2-3B as an example, when running the model's forward method normally:
        ```python
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[0,i 18, 36], return_contacts=True)
            # results["representations"][0] is input of layer 0, output of embedding layer
            # results["representations"][18] is input of layer 18 
            # results["representations"][36] is final normalized output of the full ESM2 encoder model
        ```
        
        Case 1 (layer_idx != n_layers):
            When accessing intermediate layers (e.g., layer_idx=18), we modify the input to that layer.
            This corresponds to modifying the input to layer_{layer_idx} (0-indexed internally).
            After forward pass, results["representations"][18] would contain the output of layer 18,
            which is the middle layer of the 36-layer model.
        
        Case 2 (layer_idx == n_layers):
            When accessing the final layer (e.g., layer_idx=36 for ESM2-3B), we modify the output
            of the final layer normalization (esm.encoder.emb_layer_norm_after).
            After forward pass, results["representations"][36] would contain these final normalized 
            representations, which come after all 36 transformer layers.
        
        Args:
            layer_idx: 1-indexed layer number (e.g., 36 for final layer in ESM2-3B)
        
        Returns:
            Tuple of (layer_name, override_param) where:
            - layer_name: the internal ESM module name to hook
            - override_param: whether to override 'input' or 'output' of that module
        """
        if layer_idx > self.n_layers:
            raise ValueError(f"Layer index {layer_idx} is out of bounds (max: {self.n_layers})")
        
        if layer_idx == self.n_layers:
            # Case 2: Final layer - modify output of layer norm
            return "esm.encoder.emb_layer_norm_after", "output"
        # Case 1: Intermediate layer - modify input to layer
        return f"esm.encoder.layer.{layer_idx}", "input"

    def __create_hook(self, name: str, override_param: str, hidden_state_override: Optional[torch.Tensor]):
        """Create a hook function for the specified layer."""
        activations = {}
        
        def hook(module, input, output):
            # Store activation
            if name == 'esm.encoder.emb_layer_norm_after':
                activations[name] = output.detach()
            else:
                activations[name] = output[0].detach()
            
            if hidden_state_override is None:
                return output
                
            # Handle override based on parameter type
            if override_param == 'input':
                assert hidden_state_override.shape == input[0].shape, \
                    f"Override shape {hidden_state_override.shape} doesn't match input shape {input[0].shape}"
                return (hidden_state_override,)
            
            if override_param == 'output':
                if name == 'esm.encoder.emb_layer_norm_after':
                    assert hidden_state_override.shape == output.shape
                    return hidden_state_override
                assert hidden_state_override.shape == output[0].shape and len(output) == 1
                return (hidden_state_override,)
                
            raise ValueError(f"Invalid override parameter: {override_param}")
            
        return hook, activations

    def forward_with_intervention(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_idx: int,  # 1-indexed
        hidden_state_override: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """Run forward pass with optional hidden state modification at specified layer."""
        
        layer_name, override_param = self.__get_layer_info(layer_idx)

        module = dict(self.model.named_modules())[layer_name]
        
        # Create and register hook
        hook, activations = self.__create_hook(layer_name, override_param, hidden_state_override)
        hook_handle = module.register_forward_hook(hook)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(
                tokens,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
        # Cleanup
        hook_handle.remove()
        
        # Return results
        hidden_states = hidden_state_override if hidden_state_override is not None else activations[layer_name]
        return ModelOutput(logits=output.logits, hidden_states=hidden_states)

def get_model_output_no_nnsight( #TODO: Change function name
    esm_model: EsmForMaskedLM,
    batch_tokens: torch.Tensor,
    batch_attn_mask: torch.Tensor,
    hidden_layer_idx: int,  # 1-indexed
    hidden_state_override: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Legacy wrapper for backward compatibility."""
    hooked_esm = HookedESM(esm_model)
    output = hooked_esm.forward_with_intervention(
        batch_tokens,
        batch_attn_mask,
        hidden_layer_idx,
        hidden_state_override
    )
    return output.logits, output.hidden_states

def calculate_cross_entropy(model_output, batch_tokens, batch_attn_mask):
    """Calculate cross entropy for each sequence in batch, excluding start/end tokens."""
    losses = []
    for j, mask in enumerate(batch_attn_mask):
        length = mask.sum()
        seq_logits = model_output[j, 1:length-1]
        seq_tokens = batch_tokens[j, 1:length-1]
        loss = F.cross_entropy(seq_logits, seq_tokens)
        losses.append(loss.item())
    return losses

def CE_from_sae_recon(esm_model, tokenized_batches, hidden_layer_idx,
                      sae_model=None, device=torch.device('cpu')):
    """Calculate cross entropy using SAE reconstructions."""
    sae_losses = []

    for batch_tokens, batch_attn_mask in tqdm(tokenized_batches):
        batch_tokens = batch_tokens.to(device)
        batch_attn_mask = batch_attn_mask.to(device)

        _, orig_hidden = get_model_output_no_nnsight(
            esm_model, batch_tokens, batch_attn_mask, hidden_layer_idx
        )
        print(orig_hidden.shape)

        # reconstructions = sae_model(orig_hidden)
        # sae_logits, _ = get_model_output_no_nnsight(
        #     esm_model, batch_tokens, batch_attn_mask,
        #     hidden_layer_idx, reconstructions
        # )

        # sae_losses.extend(calculate_cross_entropy(
        #     sae_logits, batch_tokens, batch_attn_mask))

    # return np.mean(sae_losses)

if __name__ == "__main__":
    model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

    data = [
        ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
        ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ("protein3",  "K A <mask> I S Q"),
    ]
    sequences = [seq for _, seq in data]

    tokenized = tokenizer(
        sequences,
        return_tensors="pt",         # return PyTorch tensors
        padding=True,                # pad to the longest sequence
        truncation=True,             # truncate if necessary
        add_special_tokens=True      # add CLS, SEP if the model expects it
    )

    # Create a TensorDataset
    dataset = TensorDataset(tokenized["input_ids"], tokenized["attention_mask"])

    # Create a DataLoader with batch_size
    batch_size = 2
    tokenized_batches_list = DataLoader(dataset, batch_size=batch_size)

    CE_from_sae_recon(model, tokenized_batches_list, 6)
