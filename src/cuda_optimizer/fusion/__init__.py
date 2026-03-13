"""
Fusion module providing fused optimizers and auto-replacement utilities.
"""

from .adam_fused import FusedAdamW, is_available as fused_adamw_available
from .optim_fusion import (
    install_fused_optimizers,
    uninstall_fused_optimizers,
    FusedOptimizerContext,
    get_available_fused_optimizers,
)

__all__ = [
    "FusedAdamW",
    "is_available",
    "install_fused_optimizers",
    "uninstall_fused_optimizers",
    "FusedOptimizerContext",
    "get_available_fused_optimizers",
]
