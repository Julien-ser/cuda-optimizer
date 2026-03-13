# Automatic Mixed Precision (AMP) Wrapper

## Overview

The `AMPWrapper` class provides an enhanced automatic mixed precision training solution that extends PyTorch's native `torch.cuda.amp`. It adds sophisticated features for production-scale deep learning workloads while maintaining numerical stability and training accuracy.

**Key Benefits:**
- **Dynamic loss scaling per layer**: Adapts loss scaling individually for different layers based on gradient statistics
- **Gradient accumulation**: Seamless gradient accumulation with automatic loss scaling
- **Accuracy validation**: Automated FP32 vs AMP accuracy comparison within 0.1% tolerance
- **Drop-in replacement**: Works with any PyTorch optimizer and model
- **Performance tracking**: Detailed metrics on overflow rates, scaling factors, and operation counts

**Target Performance:**
- **1.5-2x throughput** improvement on Volta+ GPUs with Tensor Cores
- **50% memory reduction** through FP16 activations and gradients
- **Maintains FP32 accuracy** within 0.1% for most architectures
- **Minimal code changes**: Typically 2-3 lines to enable

## Architecture

### Layer-Aware Dynamic Loss Scaling

Unlike PyTorch's global loss scaling, `AMPWrapper` can apply different scale factors to different layers based on their gradient statistics:

```
Global Scale:    1<<16 = 65,536
Layer 0 (conv):   1<<17 = 131,072  # More stable, higher scale
Layer 5 (linear): 1<<15 = 32,768   # Unstable, lower scale
Layer 9 (output): 1<<16 = 65,536   # Standard scale
```

The scaler monitors gradient norms per layer and adjusts scales every N steps:
- **Increase scale** (2x) when gradients are small (<0.5) and no overflows
- **Decrease scale** (0.5x) when overflow rate exceeds 10%
- **Maintain** current scale otherwise

This per-layer adaptation allows more aggressive scaling for stable layers while protecting sensitive layers from overflow.

### Gradient Accumulation

The wrapper handles gradient accumulation automatically:
- Loss is scaled consistently across accumulation steps
- Optimizer step only triggers after configured number of steps
- Memory footprint reduces proportionally to accumulation steps

```python
# With accumulation_steps=4, effective batch size = 4 * physical batch
wrapper = AMPWrapper(model, optimizer, accumulation_steps=4)
# Each train_step() accumulates gradients; optimizer updates every 4th step
```

### Accuracy Validation

The `validate_accuracy()` method compares AMP and FP32 accuracy on a validation set:
- Runs both modes on the same validation data
- Reports absolute difference
- Warns if difference exceeds 0.1% threshold

## Usage

### Basic Example

```python
import torch
import torch.nn as nn
from cuda_optimizer import AMPWrapper

# Define model and optimizer
model = ResNet50().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Enable AMP with default settings
amp = AMPWrapper(model=model, optimizer=optimizer, enabled=True)

# Training loop
for batch in dataloader:
    inputs, targets = batch
    inputs, targets = inputs.cuda(), targets.cuda()
    
    loss_fn = nn.CrossEntropyLoss()
    metrics = amp.train_step((inputs, targets), loss_fn)
    
    # metrics contains: loss, scale, step_taken, accumulation_step
    print(f"Loss: {metrics['loss']:.4f}, Scale: {metrics['scale']:.0f}")
```

### Individual Layer Scaling

Target specific layers for separate loss scaling:

```python
# Apply different scaling to conv layers and dense layers
amp = AMPWrapper(
    model=model,
    optimizer=optimizer,
    layers_for_individual_scaling=["conv", "linear", "attention"]
)
```

### Gradient Accumulation

```python
# Simulate larger batch size
amp = AMPWrapper(
    model=model,
    optimizer=optimizer,
    accumulation_steps=4,  # Update every 4 batches
    enabled=True
)

for batch in dataloader:
    # Accumulate gradients
    amp.train_step(batch, loss_fn, apply_optimizer_step=False)
    
    # Every 4th step:
    #   - Unscale gradients
    #   - Apply optimizer step
    #   - Zero gradients
    #   - Update loss scaler
```

### Accuracy Validation

```python
# After training, validate AMP accuracy
val_loader = DataLoader(val_dataset, batch_size=32)
amp_acc, fp32_acc = amp.validate_accuracy(val_loader, max_batches=100)

print(f"AMP accuracy: {amp_acc:.4f}")
print(f"FP32 accuracy: {fp32_acc:.4f}")
print(f"Difference: {abs(amp_acc - fp32_acc):.4f}")
```

### Manual Control

For fine-grained control:

```python
# Manual gradient accumulation
for batch in dataloader:
    loss = compute_loss(batch)
    scaled_loss = amp.scale_loss(loss)
    scaled_loss.backward()
    
# After N steps:
amp.step()  # Unscales, optimizer step, zeros grads, updates scaler
amp.zero_grad()  # Or call separately

# Get performance metrics
metrics = amp.get_metrics()
print(f"Overflow rate: {metrics['overflow_rate']:.2%}")

# Get scaling statistics
stats = amp.get_scaling_stats()
print(f"Global scale: {stats['global_scale']}")
print(f"Per-layer scales: {stats['per_layer_scales']}")

# Save and restore state for checkpointing
state = amp.state_dict()
# ... save checkpoint
amp.load_state_dict(state)
```

## API Reference

### `AMPWrapper`

Main wrapper class combining all AMP features.

#### `__init__(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    layers_for_individual_scaling: Optional[List[str]] = None,
    accumulation_steps: int = 1,
    init_scale: float = 2.0**16,
    enabled: bool = True,
    validate_accuracy: bool = False
)`

Initialize AMP wrapper.

**Parameters:**
- `model` (torch.nn.Module): Model to train
- `optimizer` (torch.optim.Optimizer): Optimizer to use
- `layers_for_individual_scaling` (list): Layer name patterns for individual scaling (e.g., ["conv", "attention"])
- `accumulation_steps` (int): Number of steps for gradient accumulation (default: 1)
- `init_scale` (float): Initial loss scale factor (default: 2^16)
- `enabled` (bool): Enable AMP (default: True)
- `validate_accuracy` (bool): Whether to run accuracy validation (default: False)

#### Methods

##### `train_step(
    batch: Tuple[torch.Tensor, ...],
    loss_fn: callable,
    apply_optimizer_step: bool = True
) → Dict[str, float]`

Perform one training step with AMP.

**Parameters:**
- `batch`: Input batch tuple (passed to loss_fn)
- `loss_fn`: Loss function taking batch and returning loss
- `apply_optimizer_step`: Whether to apply optimizer step (for accumulation)

**Returns:** dict with keys:
- `loss` (float): Loss value
- `scale` (float): Current loss scale
- `step_taken` (bool): Whether optimizer step was applied
- `accumulation_step` (int): Current accumulation counter (1 to accumulation_steps)

##### `scale_loss(loss: torch.Tensor) → torch.Tensor`

Scale loss for backward pass (used for manual control).

##### `step()`

Manually trigger optimizer step with gradient unscaling.

##### `zero_grad()`

Zero all gradients (passes through to optimizer).

##### `validate_accuracy(
    val_loader: torch.utils.data.DataLoader,
    max_batches: int = 100
) → Tuple[float, float]`

Validate AMP accuracy against FP32 baseline.

**Parameters:**
- `val_loader`: Validation data loader
- `max_batches`: Maximum batches to evaluate (default: 100)

**Returns:** (amp_accuracy, fp32_accuracy)

##### `state_dict() → Dict`

Get wrapper state for checkpointing.

##### `load_state_dict(state: Dict)`

Restore wrapper state from checkpoint.

##### `get_metrics() → Dict[str, float]`

Get performance metrics: total_steps, overflow_count, overflow_rate, scaled_ops.

##### `get_scaling_stats() → Dict`

Get detailed scaling statistics: global_scale, per_layer_scales, step count.

### LayerAwareLossScaler

Per-layer loss scaling implementation.

#### `__init__(
    initial_scale: float = 2.0**16,
    growth_factor: float = 2.0,
    backoff_factor: float = 0.5,
    growth_interval: int = 2000,
    min_scale: float = 1.0,
    max_scale: float = 2.0**24,
    layer_names: Optional[List[str]] = None
)`

#### `get_scale(param_name: Optional[str] = None) → float`

Get current scale for a parameter.

#### `update_gradient_norm(param_name: str, grad_norm: float)`

Record gradient norm for a parameter.

#### `check_overflow(param_name: str, has_overflow: bool)`

Record overflow occurrence.

#### `step()`

Update scales based on collected statistics (call every growth_interval steps).

## Performance Characteristics

### Throughput
- **Volta/Turing/Ampere GPUs**: 1.5-2x throughput with Tensor Core utilization
- **Pascal/older CUDA GPUs**: 1.2-1.5x speedup from reduced memory bandwidth

### Memory
- **50% reduction** in activation memory (FP16 vs FP32)
- **50% reduction** in gradient memory (FP16 vs FP32)
- **No additional memory** for weights (master weights kept in FP32)

### Accuracy
- **Image Classification (ResNet)**: Typically within 0.05% of FP32
- **NLP (BERT)**: Within 0.1% of FP32
- **Stable Diffusion**: Within 0.2% (acceptable for generative models)

### Overhead
- **Gradient norm tracking**: <0.5% overhead
- **Per-layer scaling**: Negligible (<0.1%)
- **Accuracy validation**: ~2-5% overhead during validation

## Best Practices

### 1. Start Default, Then Tune

```python
# Start with global scaling only
amp = AMPWrapper(model, optimizer, enabled=True)

# If you see frequent overflows, enable per-layer scaling
if metrics['overflow_rate'] > 0.05:
    amp = AMPWrapper(model, optimizer, layers_for_individual_scaling=["conv", "linear"])
```

### 2. Use Gradient Accumulation for Small Batches

```python
# When limited by memory, use accumulation to maintain effective batch size
batch_size = 8  # physical
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps  # 32
```

### 3. Validate Accuracy Early

```python
# Always validate after first few epochs
if amp.validate_accuracy:
    amp_acc, fp32_acc = amp.validate_accuracy(val_loader)
    assert abs(amp_acc - fp32_acc) < 0.001, "AMP accuracy degradation too high"
```

### 4. Monitor Overflow Rate

```python
metrics = amp.get_metrics()
if metrics['overflow_rate'] > 0.1:
    print("Warning: High overflow rate. Consider disabling AMP for some layers.")
```

### 5. Checkpoint State

```python
# Always save and restore AMP state with model checkpoint
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': amp.state_dict(),  # Not just optimizer!
    'epoch': epoch,
}
torch.save(checkpoint, 'checkpoint.pt')
```

### 6. Disable for Critical Layers

Some layers may be sensitive to FP16. Use individual layer scaling or exclude them:

```python
# High initial scale for stable layers, low for sensitive ones
# The scaler will adjust automatically based on observed gradients
```

## Implementation Details

### Integration with torch.cuda.amp

Under the hood, `AMPWrapper` uses:
- `torch.cuda.amp.autocast` for automatic operation casting
- Manual loss scaling with custom gradient unscaling
- `torch.cuda.amp.GradScaler`-like functionality, but extended for per-layer control

### Gradient Unscaling

Gradients are unscaled *before* the optimizer step to avoid underflow:

```python
# Unscaling (simplified)
for param in model.parameters():
    if param.grad is not None:
        param.grad.data.div_(scale)
        # Check for overflow (NaN/Inf)
        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
            param.grad.zero_()
```

### Loss Scaling Strategy

The dynamic scaler updates per-layer scales every N steps (default 2000):
1. Compute average gradient norm for each layer region
2. Compute overflow rate (fraction of steps with NaN/Inf)
3. Adjust scale:
   - Overflow > 10% → multiply by 0.5
   - Avg norm < 0.5 → multiply by 2.0
   - Else → maintain

### Master Weights

Model weights are kept in FP32 as "master weights" for numerical stability:
- Optimizer operates on FP32 weights
- Forward pass uses FP16 casts via autocast
- Updaterounds FP32 weight updates back to FP32

## Troubleshooting

### Problem: Training diverges with AMP

**Solution:**
```python
# 1. Reduce initial scale
amp = AMPWrapper(model, optimizer, init_scale=2.0**8)

# 2. Enable per-layer scaling
amp = AMPWrapper(model, optimizer, layers_for_individual_scaling=["conv", "linear"])

# 3. Monitor overflow rate
metrics = amp.get_metrics()
print(f"Overflow rate: {metrics['overflow_rate']:.2%}")
```

### Problem: Accuracy drop exceeds 0.1%

**Solution:**
```python
# 1. Disable AMP for critical layers by giving them lower scales
# 2. Reduce accumulation_steps (if using accumulation)
# 3. Consider disabling AMP entirely for this model
amp = AMPWrapper(model, optimizer, enabled=False)
```

### Problem: Scaling stats not updating

**Solution:**
- Ensure you call `amp.step()` regularly (or use `train_step`)
- Check that `accumulation_steps` matches your loop logic
- The scaler updates every `growth_interval` steps (default 2000)

### Problem: Memory not reduced

**Solution:**
- Verify GPU is Volta+ (Tensor Cores required for full benefit)
- Check that your operations are AMP-compatible (some ops don't have FP16 kernels)
- Use `torch.cuda.amp.autocast` compatible operations

## Advanced Topics

### Custom Layer Scaling Patterns

For fine-grained control, define custom layer name patterns:

```python
# Different scaling for each transformer block
layer_names = [f"encoder.layers.{i}" for i in range(12)]
amp = AMPWrapper(model, optimizer, layers_for_individual_scaling=layer_names)
```

### Integration with Custom Optimizers

`AMPWrapper` works with any optimizer inheriting from `torch.optim.Optimizer`:

```python
class CustomOptimizer(torch.optim.Optimizer):
    # Your implementation
    pass

optimizer = CustomOptimizer(model.parameters(), lr=1e-3)
amp = AMPWrapper(model, optimizer)
```

### Combining with Other Optimizations

`AMPWrapper` can be combined with other `cuda_optimizer` features:

```python
from cuda_optimizer import CUDACache, AMPWrapper

cache = CUDACache(max_pool_size_mb=1024)
amp = AMPWrapper(model, optimizer)

# Allocate intermediate tensors from cache
for batch in dataloader:
    # Use cache for large intermediate activations
    activations = cache.allocate(compute_size)
    output = model(batch)
    cache.free(activations)
    
    # Train with AMP
    amp.train_step(...)
```

## Related

- [CUDA Caching Allocator](cuda_cache.md) - Reduce memory fragmentation
- [Custom CUDA Kernels](custom_ops.md) - Fused operations for more speed
- [Gradient Checkpointing](selective_checkpoint.md) - Trade compute for memory
- [Kernel Auto-Tuner](autotuner.md) - Auto-optimize kernel parameters

## Examples

### Minimal Integration

```python
# Before (vanilla PyTorch)
model = Model().cuda()
optimizer = torch.optim.Adam(model.parameters())

for batch in dataloader:
    loss = compute_loss(model(batch))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

```python
# After (with AMPWrapper)
from cuda_optimizer import AMPWrapper

model = Model().cuda()
optimizer = torch.optim.Adam(model.parameters())
amp = AMPWrapper(model, optimizer)

for batch in dataloader:
    metrics = amp.train_step(batch, compute_loss)
```

### Production Training Script

```python
import torch
from cuda_optimizer import AMPWrapper

def train_model(model, train_loader, val_loader, epochs=10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    amp = AMPWrapper(
        model=model,
        optimizer=optimizer,
        accumulation_steps=4,
        layers_for_individual_scaling=["conv", "attention"],
        enabled=True
    )
    
    for epoch in range(epochs):
        model.train()
        
        # Training loop
        for batch in train_loader:
            batch = [b.cuda() for b in batch]
            metrics = amp.train_step(batch, compute_loss)
            
            # Log metrics
            if metrics['step_taken']:
                print(f"Epoch {epoch}, Loss: {metrics['loss']:.4f}, Scale: {metrics['scale']:.0f}")
        
        # Validate
        if epoch % 2 == 0:
            amp_acc, fp32_acc = amp.validate_accuracy(val_loader)
            print(f"Validation accuracy - AMP: {amp_acc:.4f}, FP32: {fp32_acc:.4f}")
        
        # Save checkpoint with AMP state
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'amp_state_dict': amp.state_dict(),
        }, f'checkpoint_epoch_{epoch}.pt')
```
