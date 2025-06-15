import contextlib
import functools
import inspect
import math
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from sageattention.core import sageattn
from .attention_utils import ensure_hnd_format, ensure_nhd_format, get_tensor_format

def dispatch_attention_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    backend: str = "sage",
) -> torch.Tensor:
    """
    Dispatch attention computation to the appropriate backend.
    
    Args:
        query: Query tensor
        key: Key tensor
        value: Value tensor
        attn_mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether to use causal attention
        backend: Attention backend to use ("sage" or "para")
    
    Returns:
        Attention output in the same format as input
    """
    # Store original format
    original_format = get_tensor_format(query)
    
    if backend == "sage":
        # Ensure HND format for Sage attention
        query, key, value = ensure_hnd_format(query, key, value)
        
        # Call Sage attention
        output = sageattn(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            tensor_layout="HND",
        )
        
        # Convert back to original format if needed
        if original_format == "NHD":
            output = output.transpose(1, 2)
    else:
        # Ensure NHD format for Para attention
        query, key, value = ensure_nhd_format(query, key, value)
        
        # Call Para attention
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
        
        # Convert back to original format if needed
        if original_format == "HND":
            output = output.transpose(1, 2)
    
    return output

def _sage_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
    return_lse: bool = False,
) -> torch.Tensor:
    # print(f"Sage attention")
    return sageattn(
        q=query,
        k=key,
        v=value,
        tensor_layout="HND",
        is_causal=is_causal,
        sm_scale=scale,
        return_lse=return_lse,
    )