# Migration Guide

Moving from vanilla PyTorch to CUDA Optimizer? This guide walks you through the process with minimal code changes.

## Migration Strategies

### Strategy 1: Drop-in Replacement (Recommended)

Add optimizations with minimal code changes using the `Optimizer.optimize()` method.

**Before:**

```python
import torch
import torch.nn as nn
import torch.optim as optim

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.mean(dim=[2, 3])
        return self.fc(x)

model = MyModel().cuda()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
```

**After:**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from cuda_optimizer import Optimizer

# Same model definition - no changes needed
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.mean(dim=[2, 3])
        return self.fc(x)

model = MyModel().cuda()

# Wrap model with optimizations - just one line!
model = Optimizer.optimize(model)

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training loop unchanged
for batch in dataloader:
    inputs, labels = batch[0].cuda(), batch[1].cuda()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Changes:** 1 line added (import + wrap)

---

### Strategy 2: Gradual Integration (For Large Codebases)

Apply optimizations component-by-component for controlled rollout.

#### Step 1: Enable Memory Pool

```python
from cuda_optimizer import CUDACache

model = CUDACache().optimize(model)
# Reduces memory fragmentation, improves allocation speed
```

#### Step 2: Add Mixed Precision Training

```python
from cuda_optimizer import AMPWrapper
from torch.cuda.amp import GradScaler

amp = AMPWrapper()
model, optimizer = amp.prepare(model, optimizer)
scaler = GradScaler()

# Update training loop:
with torch.cuda.amp.autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### Step 3: Enable Gradient Checkpointing

```python
from cuda_optimizer import SelectiveCheckpoint

checkpoint = SelectiveCheckpoint()

# Option A: Apply to specific layers
model = checkpoint.apply(model, layers=['layer1', 'layer2', 'layer3'])

# Option B: Apply to all
model = checkpoint.apply_all(model)
```

#### Step 4: Add Custom CUDA Kernels

```python
from cuda_optimizer import CustomOps

# Fuse layernorm + activation for speed
model = CustomOps.fuse_layernorm_activation(model)

# Fuse conv + bias add
model = CustomOps.fuse_conv_bias(model)
```

#### Step 5: Use Fused Optimizer (optional)

```python
from cuda_optimizer import FusedAdamW

# Replace standard AdamW with fused version
optimizer = FusedAdamW(model.parameters(), lr=1e-3)
```

#### Step 6: Enable Automatic Tuning

```python
from cuda_optimizer import Autotuner

# Auto-tune critical kernels (one-time cost)
autotuner = Autotuner(cache_dir='~/.cache/cuda-optimizer/')
autotuner.tune_operations(model, input_shape=(32, 3, 224, 224))
```

---

## Component-Specific Migration

### 1. Memory Management

**Problem:** Out-of-memory errors, high memory fragmentation

**Solution:** Use `CUDACache`

```python
import torch
from cuda_optimizer import CUDACache

# Before: using standard PyTorch memory allocator
torch.cuda.set_per_process_memory_fraction(0.9)  # manual tuning

# After: intelligent caching
cache = CUDACache(pool_size_gb=4, expiry_policy='lru')
model = cache.optimize(model)
```

---

### 2. Mixed Precision Training

**Problem:** Slow training, low GPU utilization

**Solution:** Use `AMPWrapper` with per-layer scaling

```python
from cuda_optimizer import AMPWrapper

# Before: manual mixed precision
scaler = torch.cuda.amp.GradScaler()
for batch in dataloader:
    with torch.cuda.amp.autocast():
        loss = model(batch)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# After: automatic with dynamic scaling per layer
amp = AMPWrapper(dynamic_loss_scale=True, init_scale=2**16)
model, optimizer = amp.prepare(model, optimizer)
# Training loop remains mostly the same
```

---

### 3. Gradient Checkpointing

**Problem:** Large models don't fit in memory

**Solution:** `SelectiveCheckpoint`

```python
from cuda_optimizer import SelectiveCheckpoint

# Before: reduce batch size or sequence length
batch_size = 8  # had to reduce from 32

# After: checkpoint expensive layers
checkpoint = SelectiveCheckpoint(compute_ratio=0.3)
model = checkpoint.apply(model, layers=['encoder.layers.0-5'])
# batch_size back to 32
```

---

### 4. Tensor Parallelism

**Problem:** Single GPU memory limits

**Solution:** `TensorParallel`

```python
from cuda_optimizer import TensorParallel

# Before: single GPU only
model = MyLargeModel().cuda()  # OOM on 80GB GPU

# After: split across multiple GPUs
parallel = TensorParallel(strategy='1d')  # or '2d' for mesh
model = parallel.parallelize(model, device_ids=[0, 1, 2, 3])
# Now fits on 4x40GB GPUs
```

---

### 5. Optimizer Performance

**Problem:** Optimizer step is bottleneck

**Solution:** `FusedAdamW`

```python
from cuda_optimizer import FusedAdamW

# Before: standard PyTorch AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# After: fused implementation
optimizer = FusedAdamW(model.parameters(), lr=1e-3)
# Same API, 30% faster
```

---

## Validation Checklist

After migration, verify correctness:

- [ ] Model outputs match baseline (within 1e-5 tolerance)
- [ ] Training loss converges identically
- [ ] Final accuracy within 0.1% of baseline
- [ ] No new warnings or errors
- [ ] Memory usage reduced by target %
- [ ] Throughput improved by target %

Run validation tests:

```bash
pytest tests/integration/test_migration.py -v
```

---

## Rollback Plan

If issues arise, revert easily:

```python
# Remove optimizer wrapper
model = model.module  # if wrapped

# Remove individual components
# (These return modified models; original untouched)
model = CUDACache().optimize(model)  # to undo, remove this line
```

Best practice: Keep baseline training code in version control. Test optimizations on a separate branch.

---

## Common Pitfalls

1. **Custom CUDA extensions**: If your model has custom CUDA kernels, they're not automatically optimized. Report an issue for support.

2. **In-place operations**: Some in-place ops may conflict with checkpointing. Use `torch.no_grad()` or disable checkpointing for those layers.

3. **Weight tying**: Models with tied weights (e.g., language models sharing embedding/layer norm) may need manual handling. See `docs/checkpointing.md` for details.

4. **Gradient accumulation**: When using gradient accumulation, ensure AMP wrapper is configured properly:

```python
amp = AMPWrapper(accumulation_steps=4)
model, optimizer = amp.prepare(model, optimizer)
```

---

## Need Help?

- Check [Troubleshooting](troubleshooting.md)
- Review component documentation in `docs/`
- Open an issue on GitHub with reproducibility script
