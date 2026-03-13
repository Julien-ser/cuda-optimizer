"""
Real-time monitoring dashboard for GPU metrics.

This module provides live monitoring of GPU utilization, memory usage,
and throughput during PyTorch model training/inference.
"""

from typing import Optional, Dict, List, Any, Callable
import json
import csv
import time
import threading
import subprocess
from datetime import datetime
from pathlib import Path
from collections import deque
from dataclasses import dataclass, asdict
import torch


@dataclass
class GPUMetrics:
    """Container for GPU metrics at a point in time."""

    timestamp: float
    device: int
    gpu_utilization_percent: Optional[float]
    memory_used_mb: float
    memory_total_mb: float
    memory_utilization_percent: float
    fps: Optional[float]
    latency_ms: Optional[float]
    num_iterations: int


class Dashboard:
    """
    Live GPU monitoring dashboard with Streamlit integration.

    Features:
    - Real-time GPU utilization tracking (via nvidia-smi)
    - Memory usage monitoring (allocated, cached, total)
    - Throughput measurement (FPS) and latency tracking
    - Historical data storage for visualization
    - Export to JSON/CSV for analysis
    - Multi-GPU support
    - Thread-safe operations for async training loops

    Usage:
        dashboard = Dashboard(device=0, max_history=1000)
        dashboard.start()

        # In training loop:
        dashboard.update(iteration=step, fps=current_fps, latency=current_latency)

        # Stop and export:
        dashboard.stop()
        dashboard.export_json("metrics.json")
        dashboard.export_csv("metrics.csv")
    """

    def __init__(
        self,
        device: int = 0,
        max_history: int = 10000,
        update_interval: float = 1.0,
        use_nvidia_smi: bool = True,
    ):
        """
        Initialize the dashboard.

        Args:
            device: CUDA device index to monitor
            max_history: Maximum number of metric snapshots to keep
            update_interval: How often (seconds) to poll GPU stats
            use_nvidia_smi: Whether to use nvidia-smi for utilization data
        """
        self.device = device
        self.max_history = max_history
        self.update_interval = update_interval
        self.use_nvidia_smi = use_nvidia_smi

        self._metrics: List[GPUMetrics] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        # Counters for throughput calculation
        self._iteration_count = 0
        self._start_time = time.time()
        self._last_fps_update = time.time()
        self._fps_window: deque = deque(maxsize=100)  # Rolling window for smooth FPS

        # Try to import pynvml for better metrics
        self._nvml_handle = None
        if use_nvidia_smi and torch.cuda.is_available():
            try:
                import pynvml

                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device)
                self._pynvml = pynvml
            except ImportError:
                print("pynvml not available, falling back to nvidia-smi CLI")
                self._nvml_handle = None
            except Exception as e:
                print(f"NVML initialization failed: {e}, falling back to nvidia-smi")
                self._nvml_handle = None

    def _get_gpu_utilization(self) -> Optional[float]:
        """Get current GPU utilization percentage."""
        if self._nvml_handle:
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                return util.gpu
            except Exception:
                pass

        # Fallback to nvidia-smi CLI
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
                f"--id={self.device}",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass

        return None

    def _get_memory_info(self) -> tuple[float, float]:
        """
        Get GPU memory info.

        Returns:
            (used_mb, total_mb)
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            used = torch.cuda.memory_allocated(self.device) / 1024**2
            # Get total memory from CUDA
            props = torch.cuda.get_device_properties(self.device)
            total = props.total_memory / 1024**2
            return used, total

        return 0.0, 0.0

    def _collect_metrics(self) -> GPUMetrics:
        """Collect current GPU metrics snapshot."""
        timestamp = time.time()
        gpu_util = self._get_gpu_utilization()
        memory_used, memory_total = self._get_memory_info()
        memory_pct = (memory_used / memory_total * 100) if memory_total > 0 else 0.0

        # Calculate current FPS from rolling window
        fps = None
        if self._fps_window:
            fps = sum(self._fps_window) / len(self._fps_window)

        # Average latency would be tracked separately
        latency = None

        return GPUMetrics(
            timestamp=timestamp,
            device=self.device,
            gpu_utilization_percent=gpu_util,
            memory_used_mb=memory_used,
            memory_total_mb=memory_total,
            memory_utilization_percent=memory_pct,
            fps=fps,
            latency_ms=latency,
            num_iterations=self._iteration_count,
        )

    def update(self, iteration: int, latency_ms: Optional[float] = None) -> None:
        """
        Manually update metrics after an iteration.

        Args:
            iteration: Current iteration number
            latency_ms: Optional latency measurement for this iteration
        """
        with self._lock:
            self._iteration_count = iteration

            # Calculate FPS since last update
            now = time.time()
            if self._last_fps_update:
                time_diff = now - self._last_fps_update
                if time_diff > 0:
                    iterations_since_last = max(
                        0,
                        iteration - self._last_iteration
                        if hasattr(self, "_last_iteration")
                        else 0,
                    )
                    if iterations_since_last > 0:
                        current_fps = iterations_since_last / time_diff
                        self._fps_window.append(current_fps)

            self._last_fps_update = now
            self._last_iteration = iteration

            # Collect and store metrics
            metrics = self._collect_metrics()
            metrics.latency_ms = latency_ms
            self._metrics.append(metrics)

            # Trim old data if exceeding max history
            if len(self._metrics) > self.max_history:
                self._metrics = self._metrics[-self.max_history :]

    def _monitor_loop(self) -> None:
        """Background thread that continuously polls GPU metrics."""
        while self._running:
            try:
                metrics = self._collect_metrics()
                with self._lock:
                    self._metrics.append(metrics)
                    if len(self._metrics) > self.max_history:
                        self._metrics = self._metrics[-self.max_history :]
            except Exception as e:
                print(f"Monitoring error: {e}")

            time.sleep(self.update_interval)

    def start(self) -> None:
        """Start background monitoring thread."""
        if not self._running:
            self._running = True
            self._start_time = time.time()
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_latest_metrics(self) -> Optional[GPUMetrics]:
        """Get the most recent metrics snapshot."""
        with self._lock:
            if self._metrics:
                return self._metrics[-1]
        return None

    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """
        Get all collected metrics as list of dictionaries.

        Returns:
            List of metric dictionaries with serializable values
        """
        with self._lock:
            return [asdict(m) for m in self._metrics]

    def export_json(self, output_path: str) -> None:
        """
        Export metrics history to JSON file.

        Args:
            output_path: Path to save JSON file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        history = self.get_metrics_history()

        export_data = {
            "metadata": {
                "device": self.device,
                "max_history": self.max_history,
                "total_entries": len(history),
                "export_timestamp": datetime.now().isoformat(),
            },
            "metrics": history,
        }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"Dashboard metrics exported to {output_path}")

    def export_csv(self, output_path: str) -> None:
        """
        Export metrics history to CSV file.

        Args:
            output_path: Path to save CSV file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        history = self.get_metrics_history()

        if not history:
            print("No metrics to export")
            return

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

        print(f"Dashboard metrics exported to {output_path}")

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics over the collected metrics.

        Returns:
            Dictionary with avg/min/max for key metrics
        """
        with self._lock:
            if not self._metrics:
                return {}

            gpu_utils = [
                m.gpu_utilization_percent
                for m in self._metrics
                if m.gpu_utilization_percent is not None
            ]
            memories = [m.memory_used_mb for m in self._metrics]
            fps_vals = [m.fps for m in self._metrics if m.fps is not None]

            stats = {
                "duration_seconds": self._metrics[-1].timestamp
                - self._metrics[0].timestamp,
                "total_iterations": self._metrics[-1].num_iterations,
                "gpu_utilization": {
                    "avg": sum(gpu_utils) / len(gpu_utils) if gpu_utils else None,
                    "max": max(gpu_utils) if gpu_utils else None,
                    "min": min(gpu_utils) if gpu_utils else None,
                },
                "memory_mb": {
                    "avg": sum(memories) / len(memories),
                    "max": max(memories),
                    "min": min(memories),
                    "final": memories[-1],
                },
            }

            if fps_vals:
                stats["fps"] = {
                    "avg": sum(fps_vals) / len(fps_vals),
                    "max": max(fps_vals),
                    "min": min(fps_vals),
                    "final": fps_vals[-1],
                }

            return stats

    def __enter__(self) -> "Dashboard":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
