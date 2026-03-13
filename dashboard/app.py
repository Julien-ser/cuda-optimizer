#!/usr/bin/env python3
"""
Real-time CUDA monitoring dashboard.

Streamlit-based UI for live GPU utilization, memory, and throughput monitoring.
Usage: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime
import sys
from pathlib import Path

import torch

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from cuda_optimizer.monitoring.dashboard import Dashboard, GPUMetrics


# Page config
st.set_page_config(
    page_title="CUDA Optimizer Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("🚀 CUDA Optimizer Dashboard")
st.markdown("Real-time monitoring for PyTorch models on CUDA devices")

# Sidebar controls
st.sidebar.header("⚙️ Configuration")

# Device selection
device = st.sidebar.selectbox(
    "CUDA Device",
    options=list(range(torch.cuda.device_count() if torch.cuda.is_available() else 1)),
    format_func=lambda x: (
        f"GPU {x}"
        + (f" ({torch.cuda.get_device_name(x)})" if torch.cuda.is_available() else "")
    ),
)

# Update interval
update_interval = st.sidebar.slider(
    "Update Interval (seconds)",
    min_value=0.5,
    max_value=5.0,
    value=1.0,
    step=0.5,
)

# Max history
max_history = st.sidebar.slider(
    "Max History Points",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100,
)

# Initialize session state
if "dashboard" not in st.session_state:
    st.session_state.dashboard = None
if "is_monitoring" not in st.session_state:
    st.session_state.is_monitoring = False


# Helper to initialize dashboard
def init_dashboard():
    import torch

    st.session_state.dashboard = Dashboard(
        device=device,
        max_history=max_history,
        update_interval=update_interval,
        use_nvidia_smi=True,
    )


# Control buttons
col1, col2, col3 = st.sidebar.columns(3)

with col1:
    if st.button("▶️ Start", type="primary", disabled=st.session_state.is_monitoring):
        if st.session_state.dashboard is None:
            init_dashboard()
        st.session_state.dashboard.start()
        st.session_state.is_monitoring = True
        st.rerun()

with col2:
    if st.button("⏹️ Stop", disabled=not st.session_state.is_monitoring):
        if st.session_state.dashboard:
            st.session_state.dashboard.stop()
        st.session_state.is_monitoring = False
        st.rerun()

with col3:
    if st.button("🔄 Reset"):
        st.session_state.dashboard = None
        st.session_state.is_monitoring = False
        st.rerun()

# Export options
st.sidebar.header("📤 Export")

if st.session_state.dashboard and st.session_state.dashboard.get_metrics_history():
    export_path = st.sidebar.text_input("Export Path", value="dashboard_metrics")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Export JSON"):
            st.session_state.dashboard.export_json(f"{export_path}.json")
            st.sidebar.success(f"Saved to {export_path}.json")

    with col2:
        if st.button("Export CSV"):
            st.session_state.dashboard.export_csv(f"{export_path}.csv")
            st.sidebar.success(f"Saved to {export_path}.csv")

# Main dashboard
if st.session_state.is_monitoring and st.session_state.dashboard:
    # Add placeholder for auto-refresh
    placeholder = st.empty()

    # Auto-refresh every update_interval seconds
    with placeholder.container():
        latest = st.session_state.dashboard.get_latest_metrics()
        history = st.session_state.dashboard.get_metrics_history()
        summary = st.session_state.dashboard.get_summary_stats()

        if latest and history:
            # Top-level metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    label="GPU Utilization",
                    value=f"{latest.gpu_utilization_percent:.1f}%"
                    if latest.gpu_utilization_percent
                    else "N/A",
                    delta=f"Avg: {summary.get('gpu_utilization', {}).get('avg', 0):.1f}%",
                )

            with col2:
                st.metric(
                    label="Memory Used",
                    value=f"{latest.memory_used_mb:.0f} MB",
                    delta=f"{latest.memory_utilization_percent:.1f}%",
                )

            with col3:
                st.metric(
                    label="FPS",
                    value=f"{latest.fps:.1f}" if latest.fps else "N/A",
                    delta=f"Avg: {summary.get('fps', {}).get('avg', 0):.1f}",
                )

            with col4:
                st.metric(
                    label="Iterations",
                    value=f"{latest.num_iterations}",
                    delta=f"Duration: {summary.get('duration_seconds', 0):.1f}s",
                )

            # Create dataframe for plotting
            df = pd.DataFrame(history)
            df["time_seconds"] = (df["timestamp"] - df["timestamp"].iloc[0]).round(2)

            # Plots
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📈 GPU Utilization & Memory")
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                if (
                    "gpu_utilization_percent" in df
                    and df["gpu_utilization_percent"].notna().any()
                ):
                    fig.add_trace(
                        go.Scatter(
                            x=df["time_seconds"],
                            y=df["gpu_utilization_percent"],
                            name="GPU Utilization %",
                            line=dict(color="blue"),
                        ),
                        secondary_y=False,
                    )

                fig.add_trace(
                    go.Scatter(
                        x=df["time_seconds"],
                        y=df["memory_utilization_percent"],
                        name="Memory Utilization %",
                        line=dict(color="red"),
                    ),
                    secondary_y=True,
                )

                fig.update_yaxes(title_text="GPU %", secondary_y=False)
                fig.update_yaxes(title_text="Memory %", secondary_y=True)
                fig.update_xaxes(title_text="Time (seconds)")
                fig.update_layout(height=400, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("⚡ Throughput (FPS)")
                if "fps" in df and df["fps"].notna().any():
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=df["time_seconds"],
                            y=df["fps"],
                            name="FPS",
                            line=dict(color="green"),
                            fill="tozeroy",
                        )
                    )
                    fig.update_yaxes(title_text="Frames/sec")
                    fig.update_xaxes(title_text="Time (seconds)")
                    fig.update_layout(height=400, hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("FPS data will appear once iterations are running")

            # Summary stats expander
            with st.expander("📊 Summary Statistics"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**GPU Utilization:**")
                    if summary.get("gpu_utilization"):
                        st.write(f"- Average: {summary['gpu_utilization']['avg']:.1f}%")
                        st.write(f"- Max: {summary['gpu_utilization']['max']:.1f}%")
                        st.write(f"- Min: {summary['gpu_utilization']['min']:.1f}%")

                with col2:
                    st.write("**Memory (MB):**")
                    if summary.get("memory_mb"):
                        st.write(f"- Average: {summary['memory_mb']['avg']:.0f}")
                        st.write(f"- Peak: {summary['memory_mb']['max']:.0f}")
                        st.write(f"- Current: {summary['memory_mb']['final']:.0f}")

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Throughput (FPS):**")
                    if summary.get("fps"):
                        st.write(f"- Average: {summary['fps']['avg']:.1f}")
                        st.write(f"- Max: {summary['fps']['max']:.1f}")
                        st.write(f"- Min: {summary['fps']['min']:.1f}")

                with col2:
                    st.write("**Session:**")
                    st.write(f"- Duration: {summary['duration_seconds']:.1f}s")
                    st.write(f"- Iterations: {summary['total_iterations']}")
                    st.write(f"- Data Points: {len(history)}")

            # Manual iteration update
            st.divider()
            col1, col2 = st.columns([1, 3])
            with col1:
                iteration = st.number_input(
                    "Current Iteration",
                    min_value=0,
                    value=latest.num_iterations,
                    step=1,
                )
            with col2:
                if st.button("Update Iteration"):
                    st.session_state.dashboard.update(iteration=iteration)
                    st.rerun()

            # Auto-refresh
            time.sleep(update_interval)
            st.rerun()
        else:
            st.info("Waiting for metrics...")
else:
    # Show instructions when not monitoring
    st.info("👈 Click '▶️ Start' in the sidebar to begin monitoring")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Features")
        st.markdown("""
        - **Real-time GPU monitoring** (utilization, memory, temperature)
        - **FPS tracking** for throughput analysis
        - **Historical charts** with zoom/pan
        - **Export to JSON/CSV** for offline analysis
        - **Multi-GPU support**
        - **Thread-safe** operations
        """)

    with col2:
        st.subheader("🚀 Quick Start")
        st.markdown("""
        1. Select your CUDA device from the sidebar
        2. Adjust update interval and history size
        3. Click **Start** to begin monitoring
        4. Integrate with your training loop:
        ```python
        dashboard = Dashboard(device=0)
        dashboard.start()
        
        for epoch in epochs:
            # training step...
            dashboard.update(iteration=step)
        
        dashboard.stop()
        dashboard.export_csv("metrics.csv")
        ```
        """)

# Footer
st.divider()
st.caption("CUDA Optimizer Dashboard • Built with Streamlit & Plotly")
