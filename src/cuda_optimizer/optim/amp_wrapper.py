"""
Automatic Mixed Precision (AMP) wrapper with advanced features.

This module provides an enhanced AMP wrapper that extends PyTorch's native
torch.cuda.amp with:
- Dynamic loss scaling per layer (adaptive scaling based on gradient norms)
- Configurable gradient accumulation strategy
- Accuracy validation utilities
- Fine-grained control over precision per layer
"""

import torch
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, List, Optional, Tuple, Union
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class LayerAwareLossScaler:
    """
    Per-layer dynamic loss scaling based on gradient norms.

    Scales loss differently for different layers based on their gradient
    statistics to maintain numerical stability while maximizing FP16 benefits.
    """

    def __init__(
        self,
        initial_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        min_scale: float = 1.0,
        max_scale: float = 2.0**24,
        layer_names: Optional[List[str]] = None,
    ):
        """
        Initialize layer-aware loss scaler.

        Args:
            initial_scale: Starting scale factor
            growth_factor: Multiplier for increasing scale when stable
            backoff_factor: Multiplier for decreasing scale on overflow
            growth_interval: Steps between scale increases
            min_scale: Minimum allowed scale
            max_scale: Maximum allowed scale
            layer_names: List of layer names to track separately (None = all)
        """
        self.initial_scale = initial_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.min_scale = min_scale
        self.max_scale = max_scale

        # Current scale for each layer (or global if layer_names is None)
        self._scales: Dict[str, float] = {}
        self._global_scale = initial_scale

        # Tracking statistics per layer
        self._grad_norm_sums: Dict[str, float] = defaultdict(float)
        self._grad_norm_counts: Dict[str, int] = defaultdict(int)
        self._overflow_counts: Dict[str, int] = defaultdict(int)

        # Step counter for growth interval
        self._step = 0

        # Layer names to track (None = track all)
        self.layer_names = layer_names

    def _get_layer_key(self, param_name: str) -> str:
        """Extract layer key from parameter name."""
        if self.layer_names is None:
            return "global"

        # Match layer name from parameter name
        for layer in self.layer_names:
            if layer in param_name:
                return layer
        return "other"

    def get_scale(self, param_name: Optional[str] = None) -> float:
        """Get current scale for a specific layer or global."""
        if param_name is None:
            return self._global_scale

        key = self._get_layer_key(param_name)
        return self._scales.get(key, self._global_scale)

    def update_gradient_norm(self, param_name: str, grad_norm: float) -> None:
        """
        Update gradient norm statistics for a layer.

        Args:
            param_name: Parameter name
            grad_norm: L2 norm of gradient
        """
        key = self._get_layer_key(param_name)
        self._grad_norm_sums[key] += grad_norm
        self._grad_norm_counts[key] += 1

    def check_overflow(self, param_name: str, has_overflow: bool) -> None:
        """Record if a parameter had gradient overflow."""
        key = self._get_layer_key(param_name)
        if has_overflow:
            self._overflow_counts[key] += 1

    def step(self) -> None:
        """Update scales based on collected statistics."""
        self._step += 1

        # Update per-layer scales periodically
        if self._step % self.growth_interval == 0:
            self._update_scales()

    def _update_scales(self) -> None:
        """Update scales for all tracked layers."""
        all_keys = set(self._grad_norm_sums.keys()) | set(self._overflow_counts.keys())

        for key in all_keys:
            # Compute average gradient norm
            if self._grad_norm_counts[key] > 0:
                avg_norm = self._grad_norm_sums[key] / self._grad_norm_counts[key]
            else:
                avg_norm = 0.0

            overflow_rate = (
                self._overflow_counts[key] / max(self._step, 1)
                if self._step > 0
                else 0.0
            )

            current_scale = self._scales.get(key, self._global_scale)

            # Adjust scale based on gradient statistics
            if overflow_rate > 0.1:
                # Too many overflows, reduce scale
                new_scale = max(current_scale * self.backoff_factor, self.min_scale)
            elif avg_norm > 0.0 and avg_norm < 0.5:
                # Gradients are small, can increase scale
                new_scale = min(current_scale * self.growth_factor, self.max_scale)
            else:
                # Maintain current scale
                new_scale = current_scale

            self._scales[key] = new_scale

            logger.debug(
                f"Layer '{key}': scale={new_scale:.2e}, "
                f"avg_norm={avg_norm:.4f}, overflow_rate={overflow_rate:.3f}"
            )

        # Reset statistics for next interval
        self._grad_norm_sums.clear()
        self._grad_norm_counts.clear()
        self._overflow_counts.clear()

    def state_dict(self) -> Dict:
        """Return state dict for checkpointing."""
        return {
            "global_scale": self._global_scale,
            "scales": self._scales.copy(),
            "step": self._step,
            "grad_norm_sums": dict(self._grad_norm_sums),
            "grad_norm_counts": dict(self._grad_norm_counts),
            "overflow_counts": dict(self._overflow_counts),
        }

    def load_state_dict(self, state: Dict) -> None:
        """Load state from checkpoint."""
        self._global_scale = state["global_scale"]
        self._scales = state["scales"].copy()
        self._step = state["step"]
        self._grad_norm_sums = defaultdict(float, state["grad_norm_sums"])
        self._grad_norm_counts = defaultdict(int, state["grad_norm_counts"])
        self._overflow_counts = defaultdict(int, state["overflow_counts"])


class AMPWrapper:
    """
    Enhanced AMP wrapper with dynamic loss scaling and gradient accumulation.

    Extends torch.cuda.amp with:
    - Layer-aware dynamic loss scaling
    - Gradient accumulation with automatic scaling
    - Accuracy validation utilities
    - Integration with PyTorch optimizers
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        layers_for_individual_scaling: Optional[List[str]] = None,
        accumulation_steps: int = 1,
        init_scale: float = 2.0**16,
        enabled: bool = True,
        validate_accuracy: bool = False,
    ):
        """
        Initialize AMP wrapper.

        Args:
            model: PyTorch model to optimize
            optimizer: Optimizer (will be wrapped for AMP)
            layers_for_individual_scaling: List of layer name patterns for individual scaling
            accumulation_steps: Number of steps for gradient accumulation
            init_scale: Initial loss scale
            enabled: Whether AMP is enabled
            validate_accuracy: Run accuracy validation after optimization
        """
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.enabled = enabled
        self.validate_accuracy = validate_accuracy

        # Initialize layer-aware scaler
        self.scaler = LayerAwareLossScaler(
            initial_scale=init_scale, layer_names=layers_for_individual_scaling
        )

        # Track accumulation step counter
        self._current_step = 0

        # Store FP32 reference weights for accuracy validation
        self._fp32_weights: Optional[Dict[str, torch.Tensor]] = None

        # Performance metrics
        self.metrics = {
            "overflow_count": 0,
            "total_steps": 0,
            "scaled_ops": 0,
            "fp16_ops": 0,
        }

        logger.info(
            f"AMPWrapper initialized: enabled={enabled}, "
            f"accumulation_steps={accumulation_steps}, "
            f"layers={layers_for_individual_scaling}"
        )

    def train_step(
        self,
        batch: Tuple[torch.Tensor, ...],
        loss_fn: callable,
        apply_optimizer_step: bool = True,
    ) -> Dict[str, float]:
        """
        Perform a single training step with AMP.

        Args:
            batch: Input batch (inputs, targets, ...)
            loss_fn: Loss function to compute
            apply_optimizer_step: Whether to apply optimizer step (useful for gradient accumulation)

        Returns:
            Dictionary with metrics (loss, scale, overflow flag, etc.)
        """
        if not self.enabled:
            # Fall back to FP32 training
            return self._train_step_fp32(batch, loss_fn, apply_optimizer_step)

        self.metrics["total_steps"] += 1

        # Scale loss for AMP
        with autocast():
            loss = loss_fn(*batch)
            # Scale loss for backward
            scaled_loss = loss * self.scaler.get_scale()

        # Backward pass
        scaled_loss.backward()

        # Track scaled operations
        self.metrics["scaled_ops"] += 1

        # Only update gradients after accumulation steps
        self._current_step += 1
        should_step = (
            self._current_step % self.accumulation_steps == 0
        ) and apply_optimizer_step

        if should_step:
            # Unscale gradients before optimizer step
            self._unscale_gradients()

            # Perform optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Update scaler
            self.scaler.step()
            self._current_step = 0
        else:
            # Still accumulating gradients
            logger.debug(
                f"Accumulating gradients: {self._current_step}/{self.accumulation_steps}"
            )

        metrics = {
            "loss": loss.item(),
            "scale": self.scaler.get_scale(),
            "step_taken": should_step,
            "accumulation_step": self._current_step,
        }

        return metrics

    def _train_step_fp32(
        self,
        batch: Tuple[torch.Tensor, ...],
        loss_fn: callable,
        apply_optimizer_step: bool = True,
    ) -> Dict[str, float]:
        """FP32 training fallback."""
        loss = loss_fn(*batch)
        loss.backward()

        self._current_step += 1
        should_step = (
            self._current_step % self.accumulation_steps == 0
        ) and apply_optimizer_step

        if should_step:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._current_step = 0

        return {
            "loss": loss.item(),
            "scale": 1.0,
            "step_taken": should_step,
            "accumulation_step": self._current_step,
        }

    def _unscale_gradients(self) -> None:
        """Unscale gradients for all parameters."""
        for param_name, param in self.model.named_parameters():
            if param.grad is not None:
                # Check for overflow
                grad_norm = param.grad.norm().item()
                has_overflow = (
                    torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                )

                # Update scaler statistics
                self.scaler.check_overflow(param_name, has_overflow)
                self.scaler.update_gradient_norm(param_name, grad_norm)

                if has_overflow:
                    self.metrics["overflow_count"] += 1
                    # Zero out gradient if overflow
                    param.grad.zero_()
                    logger.warning(
                        f"Gradient overflow in {param_name}, zeroing gradient"
                    )
                else:
                    # Unscale gradient
                    scale = self.scaler.get_scale(param_name)
                    param.grad.data.div_(scale)

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for backward pass."""
        if not self.enabled:
            return loss
        return loss * self.scaler.get_scale()

    def step(self) -> None:
        """Manually trigger optimizer step (for manual gradient accumulation control)."""
        if not self.enabled:
            self.optimizer.step()
            return

        self._unscale_gradients()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scaler.step()

    def zero_grad(self) -> None:
        """Zero gradients."""
        self.optimizer.zero_grad()

    def state_dict(self) -> Dict:
        """Return state dict for checkpointing."""
        state = {
            "scaler": self.scaler.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "metrics": self.metrics.copy(),
            "current_step": self._current_step,
        }
        return state

    def load_state_dict(self, state: Dict) -> None:
        """Load state from checkpoint."""
        self.scaler.load_state_dict(state["scaler"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.metrics = state["metrics"].copy()
        self._current_step = state["current_step"]

    def validate_accuracy(
        self, val_loader: torch.utils.data.DataLoader, max_batches: int = 100
    ) -> Tuple[float, float]:
        """
        Validate model accuracy and compare with FP32 baseline.

        Args:
            val_loader: Validation data loader
            max_batches: Maximum number of batches to validate on

        Returns:
            Tuple of (amp_accuracy, fp32_accuracy)
        """
        logger.info("Starting accuracy validation...")

        # Store current model state
        model_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # Enable AMP temporarily
        original_enabled = self.enabled
        self.enabled = False

        # Run FP32 baseline
        fp32_acc = self._run_validation(val_loader, max_batches)

        # Restore AMP and run
        self.enabled = original_enabled
        amp_acc = self._run_validation(val_loader, max_batches)

        # Restore model state
        self.model.load_state_dict(model_state)

        # Calculate difference
        diff = abs(amp_acc - fp32_acc)
        logger.info(
            f"Accuracy validation: FP32={fp32_acc:.4f}, AMP={amp_acc:.4f}, diff={diff:.4f}"
        )

        if diff > 0.001:  # 0.1%
            logger.warning(
                f"AMP accuracy differs from FP32 by {diff:.4f} (>0.1%). "
                "Consider adjusting loss scaling or disabling AMP for sensitive layers."
            )

        return amp_acc, fp32_acc

    def _run_validation(
        self, val_loader: torch.utils.data.DataLoader, max_batches: int
    ) -> float:
        """Run validation and return accuracy."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                if i >= max_batches:
                    break

                inputs, targets = inputs.cuda(), targets.cuda()

                if self.enabled:
                    with autocast():
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        self.model.train()
        return correct / total if total > 0 else 0.0

    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        metrics = self.metrics.copy()
        if metrics["total_steps"] > 0:
            overflow_rate = metrics["overflow_count"] / metrics["total_steps"]
            metrics["overflow_rate"] = overflow_rate
        return metrics

    def get_scaling_stats(self) -> Dict:
        """Get detailed scaling statistics."""
        stats = {
            "global_scale": self.scaler._global_scale,
            "per_layer_scales": self.scaler._scales.copy(),
            "step": self.scaler._step,
        }
        return stats
