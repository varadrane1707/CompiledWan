import torch
from typing import Tuple, Optional

def ensure_hnd_format(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert tensors to HND format if they aren't already."""
    if query.dim() == 4 and query.shape[1] != query.shape[2]:
        return query, key, value
    
    return (
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2)
    )

def ensure_nhd_format(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert tensors to NHD format if they aren't already."""
    if query.dim() == 4 and query.shape[1] == query.shape[2]:
        return query, key, value
    
    return (
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2)
    )

def get_tensor_format(tensor: torch.Tensor) -> str:
    """Determine if a tensor is in HND or NHD format."""
    if tensor.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got shape {tensor.shape}")
    
    if tensor.shape[1] == tensor.shape[2]:
        return "HND"
    return "NHD"

def convert_tensor_format(
    tensor: torch.Tensor,
    target_format: str,
    current_format: Optional[str] = None
) -> torch.Tensor:
    """Convert a tensor between HND and NHD formats."""
    if current_format is None:
        current_format = get_tensor_format(tensor)
    
    if current_format == target_format:
        return tensor
    
    return tensor.transpose(1, 2) 