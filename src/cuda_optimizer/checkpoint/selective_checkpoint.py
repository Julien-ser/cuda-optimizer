"""
SelectiveCheckpoint: Select which layers in a model should use gradient checkpointing.

This module provides the SelectiveCheckpoint class that allows fine-grained
selection of layers for gradient checkpointing, including by name patterns,
by type, or explicit layer references. It also supports custom recompute
functions per layer.
"""

import re
import torch.nn as nn
from typing import Callable, List, Set, Pattern, Union, Dict, Any


class SelectiveCheckpoint:
    """Selective gradient checkpointing with per-layer control."""

    def __init__(self):
        """Initialize an empty selector."""
        self._selected_layers: Set[nn.Module] = set()
        self._name_patterns: List[Pattern] = []
        self._type_whitelist: List[type] = []
        self._custom_recompute: Dict[nn.Module, Callable] = {}

    def select_layers(self, layers: List[nn.Module]) -> None:
        """
        Explicitly select specific layer instances for checkpointing.

        Args:
            layers: List of nn.Module instances to checkpoint.
        """
        self._selected_layers.update(layers)

    def select_by_name(self, pattern: str) -> None:
        """
        Select layers whose names match the given regex pattern.

        Args:
            pattern: Regular expression pattern to match layer names.
        """
        compiled = re.compile(pattern)
        self._name_patterns.append(compiled)

    def select_by_type(self, layer_type: type) -> None:
        """
        Select all layers of a specific type.

        Args:
            layer_type: A PyTorch module class (e.g., nn.Linear, nn.Conv2d).
        """
        self._type_whitelist.append(layer_type)

    def set_custom_recompute(self, layer: nn.Module, recompute_fn: Callable) -> None:
        """
        Set a custom recompute function for a specific layer.

        Args:
            layer: The layer to which the custom recompute applies.
            recompute_fn: A function with signature (forward_fn, *args, **kwargs)
                         that returns the recomputed output.
        """
        self._custom_recompute[layer] = recompute_fn

    def get_selected_layers(self, model: nn.Module) -> Set[nn.Module]:
        """
        Get the full set of layers that should be checkpointed for a given model.

        This combines explicitly selected layers, layers matching name patterns,
        and layers matching type whitelist.

        Args:
            model: The model to analyze.

        Returns:
            Set of nn.Module instances selected for checkpointing.
        """
        selected = set(self._selected_layers)

        # Match by name patterns
        for name, module in model.named_modules():
            for pattern in self._name_patterns:
                if pattern.search(name):
                    selected.add(module)

        # Match by type
        for module in model.modules():
            for t in self._type_whitelist:
                if isinstance(module, t):
                    selected.add(module)

        return selected

    def get_recompute_fn(self, layer: nn.Module) -> Callable:
        """
        Get the recompute function for a layer.

        Returns the custom recompute function if set, otherwise the default.

        Args:
            layer: The layer to query.

        Returns:
            A callable with signature (forward_fn, *args, **kwargs).
        """
        return self._custom_recompute.get(layer, self._default_recompute)

    @staticmethod
    def _default_recompute(forward_fn: Callable, *args, **kwargs):
        """
        Default recompute using torch.utils.checkpoint.

        Args:
            forward_fn: The original forward method of the layer.
            *args: Positional arguments to the forward method.
            **kwargs: Keyword arguments to the forward method.

        Returns:
            The output of the forward pass, with checkpointing hooks installed.
        """
        from torch.utils.checkpoint import checkpoint

        return checkpoint(forward_fn, *args, **kwargs)
