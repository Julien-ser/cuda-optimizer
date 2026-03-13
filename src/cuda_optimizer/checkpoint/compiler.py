"""
CheckpointCompiler: Compiles a model with gradient checkpointing.

This module provides the CheckpointCompiler class that applies selective
gradient checkpointing to PyTorch models using torch.utils.checkpoint.
"""

import torch.nn as nn
from typing import Callable, Set
from .selective_checkpoint import SelectiveCheckpoint


class CheckpointCompiler:
    """Compiles a model with gradient checkpointing based on a selector."""

    def __init__(self, selector: SelectiveCheckpoint):
        """
        Initialize the compiler.

        Args:
            selector: A SelectiveCheckpoint instance that specifies which layers
                     should be checkpointed and with what recompute functions.
        """
        self.selector = selector

    def compile(self, model: nn.Module) -> nn.Module:
        """
        Apply checkpointing to the model.

        This method modifies the model in-place by replacing the forward methods
        of selected layers with checkpointed versions.

        Args:
            model: The PyTorch model to compile.

        Returns:
            The same model instance with checkpointing applied.
        """
        selected_layers = self.selector.get_selected_layers(model)

        for layer in selected_layers:
            self._wrap_layer(layer)

        return model

    def _wrap_layer(self, layer: nn.Module) -> None:
        """
        Wrap a single layer's forward method with checkpointing.

        Args:
            layer: The layer to wrap.
        """
        # Avoid double-wrapping
        if getattr(layer, "_checkpoint_wrapped", False):
            return

        original_forward = layer.forward
        recompute_fn = self.selector.get_recompute_fn(layer)

        def checkpointed_forward(*args, **kwargs):
            return recompute_fn(original_forward, *args, **kwargs)

        layer.forward = checkpointed_forward
        layer._checkpoint_wrapped = True
