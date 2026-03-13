/*
 * Fused LayerNorm + Activation (GELU) CUDA kernel
 * Optimized for transformer architectures
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <torch/extension.h>

// CUDA kernel for fused LayerNorm + GELU
// This combines two operations that are typically separate, reducing memory bandwidth
__global__ void fused_layernorm_gelu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float eps,
    int num_features,
    int total_elements) {

    // Each block handles one feature dimension for a batch of elements
    int batch_offset = blockIdx.x * blockDim.x;
    int feature_idx = blockIdx.y;

    if (feature_idx >= num_features) return;

    // Compute mean and variance for this feature across the batch
    double sum = 0.0;
    double sum_sq = 0.0;

    for (int i = threadIdx.x; i < blockDim.x; i++) {
        int idx = batch_offset + i;
        if (idx < total_elements / num_features && batch_offset + i < total_elements) {
            int linear_idx = (batch_offset + i) * num_features + feature_idx;
            float val = input[linear_idx];
            sum += val;
            sum_sq += val * val;
        }
    }

    // Reduce within threadblock
    __shared__ double shared_sum[256];
    __shared__ double shared_sum_sq[256];

    double thread_sum = sum;
    double thread_sum_sq = sum_sq;

    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (threadIdx.x < i) {
            shared_sum[threadIdx.x] = thread_sum + __shfl_down_sync(0xffffffff, thread_sum, i);
            shared_sum_sq[threadIdx.x] = thread_sum_sq + __shfl_down_sync(0xffffffff, thread_sum_sq, i);
            thread_sum = shared_sum[threadIdx.x];
            thread_sum_sq = shared_sum_sq[threadIdx.x];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int count = min(blockDim.x, total_elements / num_features);
        double mean = thread_sum / count;
        double variance = thread_sum_sq / count - mean * mean;
        shared_sum[0] = mean;
        shared_sum_sq[0] = variance;
    }
    __syncthreads();

    double mean = shared_sum[0];
    double variance = shared_sum_sq[0];
    float inv_std = rsqrtf(variance + eps);

    // Apply LayerNorm + GELU
    for (int i = threadIdx.x; i < blockDim.x; i++) {
        int idx = batch_offset + i;
        if (idx < total_elements / num_features && batch_offset + i < total_elements) {
            int linear_idx = (batch_offset + i) * num_features + feature_idx;
            float val = input[linear_idx];
            
            // LayerNorm: (val - mean) * inv_std * weight + bias
            float normalized = (val - mean) * inv_std;
            float layernorm_out = normalized * weight[feature_idx] + bias[feature_idx];
            
            // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            // Using fast approximations
            float x = layernorm_out;
            float tanh_arg = 0.7978845608028654f * (x + 0.044715f * x * x * x);
            float gelu_out = 0.5f * x * (1.0f + __tanf(tanh_arg));
            
            output[linear_idx] = gelu_out;
        }
    }
}

// Alternative: Fused ReLU + LayerNorm (for CNN architectures)
__global__ void fused_layernorm_relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float eps,
    int num_features,
    int total_elements) {

    int batch_offset = blockIdx.x * blockDim.x;
    int feature_idx = blockIdx.y;

    if (feature_idx >= num_features) return;

    // Compute mean and variance
    double sum = 0.0;
    double sum_sq = 0.0;

    for (int i = threadIdx.x; i < blockDim.x; i++) {
        int idx = batch_offset + i;
        if (idx < total_elements / num_features && batch_offset + i < total_elements) {
            int linear_idx = (batch_offset + i) * num_features + feature_idx;
            float val = input[linear_idx];
            sum += val;
            sum_sq += val * val;
        }
    }

    __shared__ double shared_sum[256];
    __shared__ double shared_sum_sq[256];

    double thread_sum = sum;
    double thread_sum_sq = sum_sq;

    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (threadIdx.x < i) {
            shared_sum[threadIdx.x] = thread_sum + __shfl_down_sync(0xffffffff, thread_sum, i);
            shared_sum_sq[threadIdx.x] = thread_sum_sq + __shfl_down_sync(0xffffffff, thread_sum_sq, i);
            thread_sum = shared_sum[threadIdx.x];
            thread_sum_sq = shared_sum_sq[threadIdx.x];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int count = min(blockDim.x, total_elements / num_features);
        double mean = thread_sum / count;
        double variance = thread_sum_sq / count - mean * mean;
        shared_sum[0] = mean;
        shared_sum_sq[0] = variance;
    }
    __syncthreads();

    double mean = shared_sum[0];
    double variance = shared_sum_sq[0];
    float inv_std = rsqrtf(variance + eps);

    // Apply LayerNorm + ReLU
    for (int i = threadIdx.x; i < blockDim.x; i++) {
        int idx = batch_offset + i;
        if (idx < total_elements / num_features && batch_offset + i < total_elements) {
            int linear_idx = (batch_offset + i) * num_features + feature_idx;
            float val = input[linear_idx];
            
            float normalized = (val - mean) * inv_std;
            float layernorm_out = normalized * weight[feature_idx] + bias[feature_idx];
            
            // ReLU: max(0, x)
            output[linear_idx] = max(0.0f, layernorm_out);
        }
    }
}

// C++ interface for PyTorch
std::vector<torch::Tensor> fused_layernorm_gelu_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    float eps) {

    // Check inputs are CUDA tensors
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

    // Check dimensions: [batch, features] for LayerNorm over last dimension
    TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dimensions");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");

    int batch_size = 1;
    int num_features = input.size(-1);
    int total_elements = input.numel();

    // Flatten batch dimensions if needed
    TORCH_CHECK(weight.size(0) == num_features, "weight size must match feature dimension");
    TORCH_CHECK(bias.size(0) == num_features, "bias size must match feature dimension");

    // Create output tensor
    torch::Tensor output = torch::empty_like(input);

    // Configure kernel launch
    int threads = 256;
    int batch_blocks = (total_elements / num_features + threads - 1) / threads;
    dim3 grid(batch_blocks, num_features);

    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_layernorm_gelu", ([&] {
        fused_layernorm_gelu_kernel<<<grid, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            eps,
            num_features,
            total_elements);
    }));

    return {output};
}

std::vector<torch::Tensor> fused_layernorm_relu_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    float eps) {

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

    int num_features = input.size(-1);
    int total_elements = input.numel();

    torch::Tensor output = torch::empty_like(input);

    int threads = 256;
    int batch_blocks = (total_elements / num_features + threads - 1) / threads;
    dim3 grid(batch_blocks, num_features);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_layernorm_relu", ([&] {
        fused_layernorm_relu_kernel<<<grid, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            eps,
            num_features,
            total_elements);
    }));

    return {output};
}

// Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_layernorm_gelu", &fused_layernorm_gelu_cuda, "Fused LayerNorm + GELU CUDA kernel");
    m.def("fused_layernorm_relu", &fused_layernorm_relu_cuda, "Fused LayerNorm + ReLU CUDA kernel");
}
