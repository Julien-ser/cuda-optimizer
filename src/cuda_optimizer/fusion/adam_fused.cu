/*
 * Fused AdamW Optimizer CUDA Kernel
 *
 * Implements a fused weight update with L2 regularization (AdamW).
 * Combines multiple operations into a single kernel for better performance.
 *
 * Reference: https://arxiv.org/abs/1711.05101 (AdamW)
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>

/* Fused AdamW kernel */
template <typename scalar_t>
__global__ void adamw_fused_kernel(
    const scalar_t* __restrict__ params,
    const scalar_t* __restrict__ grads,
    scalar_t* __restrict__ exp_avg,
    scalar_t* __restrict__ exp_avg_sq,
    const scalar_t* __restrict__ lr,
    const scalar_t* __restrict__ beta1,
    const scalar_t* __restrict__ beta2,
    const scalar_t* __restrict__ weight_decay,
    const scalar_t* __restrict__ eps,
    const int64_t n_elements,
    const int64_t step
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_elements) {
        const scalar_t p = params[idx];
        const scalar_t g = grads[idx];

        // Load state
        scalar_t m = exp_avg[idx];
        scalar_t v = exp_avg_sq[idx];

        // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * g
        m = (*beta1) * m + (1 - *beta1) * g;

        // Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * g * g
        v = (*beta2) * v + (1 - *beta2) * g * g;

        // Store updated state
        exp_avg[idx] = m;
        exp_avg_sq[idx] = v;

        // Bias correction
        const scalar_t bias_correction1 = 1.0 - pow(*beta1, step);
        const scalar_t bias_correction2 = 1.0 - pow(*beta2, step);

        // Compute denom: sqrt(v) / sqrt(bias_correction2) + eps
        const scalar_t denom = sqrt(v / bias_correction2) + *eps;

        // Compute update: (m / bias_correction1) / denom + weight_decay * p
        const scalar_t update = (m / bias_correction1) / denom + *weight_decay * p;

        // Update parameters: p = p - lr * update
        params[idx] = p - (*lr) * update;
    }
}

/* Fused AdamW step with multiple parameter groups */
template <typename scalar_t>
__global__ void adamw_fused_multi_kernel(
    const scalar_t** __restrict__ params_list,
    const scalar_t** __restrict__ grads_list,
    scalar_t** __restrict__ exp_avg_list,
    scalar_t** __restrict__ exp_avg_sq_list,
    const scalar_t* __restrict__ lr_list,
    const scalar_t* __restrict__ beta1_list,
    const scalar_t* __restrict__ beta2_list,
    const scalar_t* __restrict__ weight_decay_list,
    const scalar_t* __restrict__ eps_list,
    const int64_t* __restrict__ sizes,
    const int64_t n_groups,
    const int64_t step
) {
    const int group_id = blockIdx.x;
    const int idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (group_id < n_groups && idx < sizes[group_id]) {
        const scalar_t* params = params_list[group_id];
        const scalar_t* grads = grads_list[group_id];
        scalar_t* exp_avg = exp_avg_list[group_id];
        scalar_t* exp_avg_sq = exp_avg_sq_list[group_id];

        const scalar_t lr = lr_list[group_id];
        const scalar_t beta1 = beta1_list[group_id];
        const scalar_t beta2 = beta2_list[group_id];
        const scalar_t weight_decay = weight_decay_list[group_id];
        const scalar_t eps = eps_list[group_id];

        const int64_t offset = idx;  // Already offset by group

        const scalar_t p = params[offset];
        const scalar_t g = grads[offset];

        scalar_t m = exp_avg[offset];
        scalar_t v = exp_avg_sq[offset];

        m = beta1 * m + (1 - beta1) * g;
        v = beta2 * v + (1 - beta2) * g * g;

        exp_avg[offset] = m;
        exp_avg_sq[offset] = v;

        const scalar_t bias_correction1 = 1.0 - pow(beta1, step);
        const scalar_t bias_correction2 = 1.0 - pow(beta2, step);

        const scalar_t denom = sqrt(v / bias_correction2) + eps;
        const scalar_t update = (m / bias_correction1) / denom + weight_decay * p;

        params[offset] = p - lr * update;
    }
}

/* PyTorch C++ extension interface */
at::Tensor adamw_fused_cuda(
    at::Tensor& params,
    at::Tensor& grads,
    at::Tensor& exp_avg,
    at::Tensor& exp_avg_sq,
    at::Tensor& lr,
    at::Tensor& beta1,
    at::Tensor& beta2,
    at::Tensor& weight_decay,
    at::Tensor& eps,
    const int64_t step
) {
    const int64_t n_elements = params.numel();

    const int threads = 256;
    const int blocks = (n_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(params.scalar_type(), "adamw_fused", ([&] {
        adamw_fused_kernel<scalar_t><<<blocks, threads>>>(
            params.data_ptr<scalar_t>(),
            grads.data_ptr<scalar_t>(),
            exp_avg.data_ptr<scalar_t>(),
            exp_avg_sq.data_ptr<scalar_t>(),
            lr.data_ptr<scalar_t>(),
            beta1.data_ptr<scalar_t>(),
            beta2.data_ptr<scalar_t>(),
            weight_decay.data_ptr<scalar_t>(),
            eps.data_ptr<scalar_t>(),
            n_elements,
            step
        );
    }));

    return params;
}

/* Multi-tensor version for parameter groups */
at::Tensor adamw_fused_multi_cuda(
    const std::vector<at::Tensor>& params_list,
    const std::vector<at::Tensor>& grads_list,
    std::vector<at::Tensor>& exp_avg_list,
    std::vector<at::Tensor>& exp_avg_sq_list,
    const std::vector<at::Tensor>& hyperparams_list,
    const int64_t step
) {
    const int n_groups = params_list.size();

    // Find maximum size for kernel launch configuration
    int64_t max_size = 0;
    std::vector<int64_t> sizes(n_groups);
    for (int i = 0; i < n_groups; ++i) {
        sizes[i] = params_list[i].numel();
        if (sizes[i] > max_size) max_size = sizes[i];
    }

    // Allocate device memory for pointers and sizes
    at::Tensor sizes_tensor = at::tensor(sizes, at::device(at::kCPU).dtype(at::kLong));

    // We'll use a 2D grid: group_id in x, elements in y
    const int threads = 256;
    const int blocks_per_group = (max_size + threads - 1) / threads;

    for (int g = 0; g < n_groups; ++g) {
        if (sizes[g] == 0) continue;

        dim3 grid(1, blocks_per_group);
        dim3 block(threads);

        AT_DISPATCH_FLOATING_TYPES(params_list[g].scalar_type(), "adamw_fused_multi", ([&] {
            // Create arrays of pointers for this group
            std::vector<scalar_t*> ptrs(5);
            ptrs[0] = const_cast<scalar_t*>(params_list[g].data_ptr<scalar_t>());
            ptrs[1] = const_cast<scalar_t*>(grads_list[g].data_ptr<scalar_t>());
            ptrs[2] = exp_avg_list[g].data_ptr<scalar_t>();
            ptrs[3] = exp_avg_sq_list[g].data_ptr<scalar_t>();
            ptrs[4] = const_cast<scalar_t*>(hyperparams_list[g].data_ptr<scalar_t>());  // Contains lr, beta1, beta2, wd, eps

            // For simplicity, we launch separate kernel per group with different hyperparams
            const scalar_t lr = hyperparams_list[g].data_ptr<scalar_t>()[0];
            const scalar_t beta1 = hyperparams_list[g].data_ptr<scalar_t>()[1];
            const scalar_t beta2 = hyperparams_list[g].data_ptr<scalar_t>()[2];
            const scalar_t weight_decay = hyperparams_list[g].data_ptr<scalar_t>()[3];
            const scalar_t eps = hyperparams_list[g].data_ptr<scalar_t>()[4];

            adamw_fused_kernel<scalar_t><<<grid, block>>>(
                params_list[g].data_ptr<scalar_t>(),
                grads_list[g].data_ptr<scalar_t>(),
                exp_avg_list[g].data_ptr<scalar_t>(),
                exp_avg_sq_list[g].data_ptr<scalar_t>(),
                &lr,
                &beta1,
                &beta2,
                &weight_decay,
                &eps,
                sizes[g],
                step
            );
        }));
    }

    return params_list[0];
}

/* PyBind11 module definition */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adamw_fused", &adamw_fused_cuda, "Fused AdamW optimizer step (CUDA)");
    m.def("adamw_fused_multi", &adamw_fused_multi_cuda, "Fused AdamW optimizer step for multiple tensor groups (CUDA)");
}
