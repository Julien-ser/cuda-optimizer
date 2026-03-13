"""
Optimizer fusion utilities.

Provides automatic replacement of standard PyTorch optimizers with their
fused CUDA counterparts for seamless integration.
"""

from .adam_fused import FusedAdamW, is_available as fused_adamw_available
import torch.optim as optim
import warnings

# Store original optimizers for restoration
_original_adamw = optim.AdamW if hasattr(optim, "AdamW") else None


def install_fused_optimizers():
    """
    Monkey-patch torch.optim to use fused optimizers when available.

    This replaces torch.optim.AdamW with FusedAdamW if CUDA kernels
    are loaded successfully.

    Example:
        >>> from cuda_optimizer.fusion import install_fused_optimizers
        >>> install_fused_optimizers()
        >>> # Now torch.optim.AdamW uses fused kernels automatically
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    """
    if not fused_adamw_available():
        warnings.warn("Fused optimizers not available. Skipping installation.")
        return

    if hasattr(optim, "AdamW"):
        optim.AdamW = FusedAdamW
        optim._original_adamw = _original_adamw  # Store original for restoration
        warnings.warn("Replaced torch.optim.AdamW with FusedAdamW")


def uninstall_fused_optimizers():
    """
    Restore original PyTorch optimizers.

    Example:
        >>> from cuda_optimizer.fusion import uninstall_fused_optimizers
        >>> uninstall_fused_optimizers()
        >>> # Back to standard torch.optim.AdamW
    """
    if hasattr(optim, "_original_adamw") and _original_adamw is not None:
        optim.AdamW = _original_adamw
        delattr(optim, "_original_adamw")


class FusedOptimizerContext:
    """
    Context manager for temporary fused optimizer substitution.

    Example:
        >>> with FusedOptimizerContext():
        ...     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ...     # Uses fused optimizer inside context
        >>> # Outside context, original optimizer restored
    """

    def __enter__(self):
        install_fused_optimizers()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        uninstall_fused_optimizers()


def get_available_fused_optimizers():
    """
    List all available fused optimizers.

    Returns:
        dict: Mapping of optimizer names to their fused implementations.

    Example:
        >>> from cuda_optimizer.fusion import get_available_fused_optimizers
        >>> fused = get_available_fused_optimizers()
        >>> print(fused)
        {'AdamW': <class 'cuda_optimizer.fusion.adam_fused.FusedAdamW'>}
    """
    available = {}
    if fused_adamw_available():
        available["AdamW"] = FusedAdamW
    return available
