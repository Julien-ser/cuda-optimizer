"""
Tensor parallelism utilities for PyTorch models.

This module provides tools for splitting model computations across multiple GPUs
using tensor parallelism with NCCL communication backend. Supports both 1D and 2D
tensor slicing for efficient multi-GPU training.
"""

import torch
import torch.distributed as dist
from typing import Optional, Tuple, List, Union, Dict, Any
import warnings


class TensorParallel:
    """Tensor parallelism across multiple GPUs using NCCL.

    This class provides utilities for splitting tensors across multiple GPUs,
    enabling efficient tensor parallelism for large models. Supports both 1D
    (row/column slicing) and 2D (grid-based) parallelism.

    Attributes:
        rank (int): Current process rank in the distributed group
        world_size (int): Total number of processes/GPUs
        device (torch.device): Current CUDA device
        process_group (Optional[dist.ProcessGroupNCCL]): NCCL process group
    """

    def __init__(
        self,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        init_method: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize TensorParallel communicator.

        Args:
            rank: Current process rank. If None, inferred from environment.
            world_size: Total number of processes. If None, inferred from env.
            init_method: Distributed initialization method (e.g., 'env://' or tcp://...)
            device: CUDA device to use. If None, uses current device.

        Raises:
            RuntimeError: If distributed initialization fails or CUDA unavailable.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for tensor parallelism")

        self.device = device or torch.device("cuda", torch.cuda.current_device())

        # Get rank and world size from environment if not provided
        if rank is None:
            if not dist.is_initialized():
                raise RuntimeError(
                    "Distributed not initialized. Provide rank/world_size or "
                    "call dist.init_process_group() first."
                )
            rank = dist.get_rank()
        if world_size is None:
            if not dist.is_initialized():
                raise RuntimeError(
                    "Distributed not initialized. Provide rank/world_size or "
                    "call dist.init_process_group() first."
                )
            world_size = dist.get_world_size()

        self.rank = rank
        self.world_size = world_size
        self.process_group = dist.group.WORLD if dist.is_initialized() else None

        # Validate NCCL availability
        if self.process_group is not None:
            backend = dist.get_backend(self.process_group)
            if backend != "nccl":
                warnings.warn(
                    f"Expected NCCL backend, got {backend}. Performance may degrade.",
                    UserWarning,
                )

    def split_1d_row(
        self,
        tensor: torch.Tensor,
        dim: int = 0,
        contiguous_split: bool = True,
    ) -> torch.Tensor:
        """Split tensor along a dimension (1D parallelism).

        Splits the tensor into world_size chunks along the specified dimension.

        Args:
            tensor: Input tensor to split
            dim: Dimension to split along (default: 0)
            contiguous_split: If True, split contiguous chunks; if False, use
                            strided slicing for column-wise parallelism

        Returns:
            Local shard of the tensor for this rank

        Example:
            >>> tp = TensorParallel()
            >>> full = torch.randn(8, 1024)  # 8 GPUs
            >>> local = tp.split_1d_row(full, dim=0)  # Each GPU gets 1 row
        """
        if tensor.size(dim) % self.world_size != 0:
            raise ValueError(
                f"Tensor dimension {dim} size ({tensor.size(dim)}) must be "
                f"divisible by world_size ({self.world_size})"
            )

        chunk_size = tensor.size(dim) // self.world_size

        if contiguous_split:
            start = self.rank * chunk_size
            end = start + chunk_size
            # Build slice tuple
            slices = [slice(None)] * tensor.dim()
            slices[dim] = slice(start, end)
            return tensor[tuple(slices)].to(self.device)
        else:
            # Strided slicing (for column parallelism)
            return tensor.index_select(
                dim,
                torch.arange(
                    self.rank, tensor.size(dim), self.world_size, device=tensor.device
                ),
            ).to(self.device)

    def split_1d_column(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """Split tensor along a dimension using strided slicing (column parallelism).

        Convenience method for column-wise splitting with strided indexing.

        Args:
            tensor: Input tensor to split
            dim: Dimension to split along (default: 1)

        Returns:
            Local shard of the tensor for this rank
        """
        return self.split_1d_row(tensor, dim=dim, contiguous_split=False)

    def split_2d(
        self,
        tensor: torch.Tensor,
        row_dim: int = 0,
        col_dim: int = 1,
        row_rank: Optional[int] = None,
        col_rank: Optional[int] = None,
    ) -> torch.Tensor:
        """Split tensor in 2D grid pattern (row and column parallelism).

        Splits tensor across a 2D grid of GPUs. Requires world_size to be a perfect
        square or product of two factors.

        Args:
            tensor: Input tensor to split
            row_dim: Dimension for row splitting (default: 0)
            col_dim: Dimension for column splitting (default: 1)
            row_rank: Row coordinate of this process (inferred if None)
            col_rank: Column coordinate of this process (inferred if None)

        Returns:
            Local shard resulting from 2D splitting

        Raises:
            ValueError: If world_size cannot be factored into 2D grid
        """
        # Factor world_size into grid dimensions
        factors = self._factorize(self.world_size)
        if len(factors) != 2:
            raise ValueError(
                f"World size {self.world_size} cannot be factored into 2D grid. "
                f"Use a product of two integers (e.g., 4, 9, 16, 32)."
            )

        grid_rows, grid_cols = factors[0], factors[1]

        # Infer row/col ranks from linear rank if not provided
        if row_rank is None or col_rank is None:
            row_rank = self.rank // grid_cols
            col_rank = self.rank % grid_cols

        # Split along row dimension
        row_chunk_size = tensor.size(row_dim) // grid_rows
        row_start = row_rank * row_chunk_size
        row_end = row_start + row_chunk_size

        # Split along column dimension
        col_chunk_size = tensor.size(col_dim) // grid_cols
        col_start = col_rank * col_chunk_size
        col_end = col_start + col_chunk_size

        # Build slices
        slices = [slice(None)] * tensor.dim()
        slices[row_dim] = slice(row_start, row_end)
        slices[col_dim] = slice(col_start, col_end)

        return tensor[tuple(slices)].to(self.device)

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
        async_op: bool = False,
    ) -> Union[torch.Tensor, dist.Work]:
        """Perform all-reduce across all GPUs.

        Args:
            tensor: Input tensor to reduce
            op: Reduction operation (SUM, PRODUCT, MAX, MIN, etc.)
            async_op: If True, returns a Work object for async operation

        Returns:
            Reduced tensor (or Work object if async_op=True)

        Example:
            >>> reduced = tp.all_reduce(tensor)  # Sum across all GPUs
        """
        if not dist.is_initialized():
            raise RuntimeError("Distributed must be initialized for all_reduce")

        if async_op:
            return dist.all_reduce(
                tensor, op=op, group=self.process_group, async_op=True
            )
        else:
            dist.all_reduce(tensor, op=op, group=self.process_group)
            return tensor

    def all_gather(
        self,
        tensor: torch.Tensor,
        dim: int = 0,
        async_op: bool = False,
    ) -> Union[torch.Tensor, dist.Work]:
        """Gather tensors from all ranks and concatenate along a dimension.

        Args:
            tensor: Local tensor to gather
            dim: Dimension to concatenate along (default: 0)
            async_op: If True, returns a Work object for async operation

        Returns:
            Gathered tensor from all ranks

        Example:
            >>> gathered = tp.all_gather(local_tensor, dim=0)  # Gather along batch dim
        """
        if not dist.is_initialized():
            raise RuntimeError("Distributed must be initialized for all_gather")

        # Prepare output list
        output_list = [torch.empty_like(tensor) for _ in range(self.world_size)]

        if async_op:
            work = dist.all_gather(
                output_list, tensor, group=self.process_group, async_op=True
            )
            return output_list, work
        else:
            dist.all_gather(output_list, tensor, group=self.process_group)
            return torch.cat(output_list, dim=dim)

    def all_to_all(
        self,
        tensor: torch.Tensor,
        dim: int = 0,
        async_op: bool = False,
    ) -> Union[torch.Tensor, dist.Work]:
        """All-to-all: Scatter and gather with different dimensions.

        Each rank sends a chunk of tensor to every other rank and receives
        a chunk from each rank. The input tensor is split along `dim` and
        each chunk is sent to a different rank.

        Args:
            tensor: Input tensor to scatter/gather
            dim: Dimension to split and exchange along
            async_op: If True, returns a Work object for async operation

        Returns:
            Tensor with exchanged chunks concatenated along `dim`

        Example:
            >>> # Each rank sends part of batch to others, receives full features
            >>> exchanged = tp.all_to_all(tensor, dim=1)
        """
        if not dist.is_initialized():
            raise RuntimeError("Distributed must be initialized for all_to_all")

        if tensor.size(dim) % self.world_size != 0:
            raise ValueError(
                f"Tensor dimension {dim} size ({tensor.size(dim)}) must be "
                f"divisible by world_size ({self.world_size}) for all_to_all"
            )

        chunk_size = tensor.size(dim) // self.world_size
        send_list = list(tensor.chunk(self.world_size, dim=dim))
        recv_list = [torch.empty_like(send_list[0]) for _ in range(self.world_size)]

        if async_op:
            work = dist.all_to_all(
                recv_list, send_list, group=self.process_group, async_op=True
            )
            return torch.cat(recv_list, dim=dim), work
        else:
            dist.all_to_all(recv_list, send_list, group=self.process_group)
            return torch.cat(recv_list, dim=dim)

    def reduce_scatter(
        self,
        tensor: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
        dim: int = 0,
        async_op: bool = False,
    ) -> Union[torch.Tensor, dist.Work]:
        """Reduce-scatter: Reduce and scatter in one operation.

        Each rank reduces corresponding chunks from all ranks and keeps only
        its chunk. More efficient than separate all-reduce + split.

        Args:
            tensor: Input tensor to reduce-scatter
            op: Reduction operation
            dim: Dimension to split and reduce along
            async_op: If True, returns a Work object for async operation

        Returns:
            Reduced and scattered chunk for this rank

        Example:
            >>> # Sum gradients across all GPUs and scatter result
            >>> local_grad = tp.reduce_scatter(grad_tensor, dim=0)
        """
        if not dist.is_initialized():
            raise RuntimeError("Distributed must be initialized for reduce_scatter")

        if tensor.size(dim) % self.world_size != 0:
            raise ValueError(
                f"Tensor dimension {dim} size ({tensor.size(dim)}) must be "
                f"divisible by world_size ({self.world_size})"
            )

        # For reduce_scatter, we need input to be contiguous in the split dimension
        tensor = tensor.contiguous()

        chunk_size = tensor.size(dim) // self.world_size
        input_list = list(tensor.chunk(self.world_size, dim=dim))
        output = torch.empty_like(input_list[self.rank])

        if async_op:
            work = dist.reduce_scatter(
                output, input_list, op=op, group=self.process_group, async_op=True
            )
            return output, work
        else:
            dist.reduce_scatter(output, input_list, op=op, group=self.process_group)
            return output

    def broadcast(
        self, tensor: torch.Tensor, src: int = 0, async_op: bool = False
    ) -> Union[torch.Tensor, dist.Work]:
        """Broadcast tensor from source rank to all ranks.

        Args:
            tensor: Tensor to broadcast (on src rank) or receive buffer (on others)
            src: Source rank for broadcast
            async_op: If True, returns a Work object for async operation

        Returns:
            Broadcasted tensor

        Example:
            >>> # Broadcast model parameters from rank 0
            >>> tp.broadcast(model_state, src=0)
        """
        if not dist.is_initialized():
            raise RuntimeError("Distributed must be initialized for broadcast")

        if async_op:
            work = dist.broadcast(
                tensor, src=src, group=self.process_group, async_op=True
            )
            return work
        else:
            dist.broadcast(tensor, src=src, group=self.process_group)
            return tensor

    def barrier(self) -> None:
        """Synchronize all ranks (barrier).

        Blocks until all ranks reach this point.
        """
        if dist.is_initialized():
            dist.barrier(group=self.process_group)

    @staticmethod
    def _factorize(n: int) -> List[int]:
        """Factorize an integer into two factors as close as possible.

        Args:
            n: Integer to factorize

        Returns:
            List of two factors [a, b] such that a*b = n
        """
        factors = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                factors = [i, n // i]
        if not factors:
            raise ValueError(f"Could not factorize {n}")
        return factors

    def is_initialized(self) -> bool:
        """Check if distributed communication is initialized.

        Returns:
            True if process group is initialized
        """
        return self.process_group is not None and dist.is_initialized()

    def get_local_rank(self) -> int:
        """Get local rank (GPU device ID) from global rank.

        Returns:
            Local device ID
        """
        return self.rank % torch.cuda.device_count()

    def get_device_ids(self) -> List[int]:
        """Get list of device IDs for all ranks.

        Returns:
            List of CUDA device indices
        """
        return [i % torch.cuda.device_count() for i in range(self.world_size)]

    @classmethod
    def init_from_env(cls) -> "TensorParallel":
        """Initialize from environment variables (for launch scripts).

        Expects:
            - os.environ['RANK']
            - os.environ['WORLD_SIZE']
            - os.environ['MASTER_ADDR']
            - os.environ['MASTER_PORT']

        Returns:
            Initialized TensorParallel instance

        Example:
            >>> # In each process:
            >>> tp = TensorParallel.init_from_env()
        """
        import os

        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", "29500")

        init_method = f"env://{master_addr}:{master_port}"

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method=init_method,
                rank=rank,
                world_size=world_size,
            )

        return cls(rank=rank, world_size=world_size)
