# Quickstart Guide

Get up and running with CUDA Optimizer in 5 minutes.

## Prerequisites

- NVIDIA GPU with CUDA 11.8+
- Python 3.9+
- PyTorch 2.0+

## Installation

```bash
# Clone repository
git clone <repo-url>
cd cuda-optimizer

# Install dependencies and package
pip install -e .

# Verify installation
python -c "import cuda_optimizer; print(cuda_optimizer.__version__)"
```

## First Optimization

### 1. Profile Your Model

Before optimizing, identify bottlenecks:

```python
import torch
from cuda_optimizer import profile_model

# Load your model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
model = model.cuda()

# Profile to see current performance
profile_model(model, input_shape=(32, 3, 224, 224), num_iters=100)
```

The profiler will output:
- Current FPS (frames per second)
- Memory usage
- Kernel bottlenecks
- Recommendations

### 2. Apply Automatic Optimizations

```python
from cuda_optimizer import Optimizer

# One-line optimization
optimized_model = Optimizer.optimize(model)

# Use as normal PyTorch model
optimizer = torch.optim.AdamW(optimized_model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# Train with improved performance
for batch in dataloader:
    inputs, labels = batch[0].cuda(), batch[1].cuda()
    outputs = optimized_model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

### 3. Enable Real-time Monitoring

```python
from cuda_optimizer import Dashboard

# Start monitoring dashboard
dashboard = Dashboard(port=8501)
dashboard.start()

# In browser: http://localhost:8501
# View live GPU utilization, memory usage, throughput
```

## Manual Optimization Options

If you prefer fine-grained control, use individual components:

```python
from cuda_optimizer import (
    CustomOps, CUDACache, AMPWrapper,
    SelectiveCheckpoint, FusedAdamW
)

# Enable memory caching
cache = CUDACache()
model = cache.optimize(model)

# Add fused operations
model = CustomOps.fuse_layernorm_activation(model)

# Use mixed precision with dynamic scaling
amp = AMPWrapper()
model, optimizer = amp.prepare(model, optimizer)

# Enable gradient checkpointing
checkpoint = SelectiveCheckpoint()
model = checkpoint.apply(model, layers=['layer1', 'layer2'])
```

## Supported Architectures

Out of the box optimization for:

- **CNNs**: ResNet, EfficientNet, VGG, custom conv nets
- **Transformers**: BERT, GPT-2, T5, custom attention models
- **RNNs/LSTMs**: Sequential models with recurrent layers
- **Custom architectures**: Most PyTorch models supported

## Expected Performance Gains

Based on our validation benchmarks:

| Model Type | FPS Improvement | Memory Reduction |
|------------|----------------|------------------|
| ResNet50    | +30%           | -33%             |
| BERT-small  | +30%           | -39%             |
| LSTM        | +20%           | -50%             |
| GPT-2 small | +25%           | -50%             |

Actual gains depend on:
- Model architecture
- Batch size
- Sequence length (for transformers)
- GPU type (A100, V100, RTX 3090+ recommended)

## Next Steps

1. Read the [Migration Guide](migration_guide.md) to integrate with existing codebases
2. Check [Troubleshooting](troubleshooting.md) if you encounter issues
3. Explore [API Reference](api/index.html) for advanced customization
4. See [Optimization Targets](optimization_targets.md) for detailed performance goals

## Docker Quick Start

```bash
# Build Docker image
docker build -t cuda-optimizer -f Dockerfile.cuda-dev .

# Run with GPU access
docker run --gpus all -it cuda-optimizer

# Inside container
python -c "from cuda_optimizer import Optimizer; print('Ready!')"
```

## Verifying Your Setup

Run the built-in validation tests:

```bash
# Test GPU detection
python scripts/check_cuda.py

# Run baseline benchmarks
python scripts/run_baseline.py --models resnet50 bert-small

# Run unit tests
pytest tests/unit/ -v
```

## Uninstalling

```bash
pip uninstall cuda-optimizer
```

Note: Manual optimizations stay in your code. Remove imports and calls to revert to vanilla PyTorch.
