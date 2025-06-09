import contextlib
import functools
import inspect
import math
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from sageattention import sageattn

def dispatch_attention_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    *,
    backend: str = "sage",
) -> torch.Tensor:
    attention_kwargs = attention_kwargs or {}

    kwargs = {
        "query": query,
        "key": key,
        "value": value,
        # "attn_mask": attn_mask,
        # "dropout_p": dropout_p,
        "is_causal": is_causal,
        "scale": scale,
        # "enable_gqa": enable_gqa,
        # **attention_kwargs,
    }
    
    if backend == "sage":
        return _sage_attention(**kwargs)
    else:
        raise ValueError(f"Backend {backend} not supported")
    
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