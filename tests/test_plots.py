import matplotlib
matplotlib.use("Agg")  # no window: draw to memory only

import pandas as pd

from nano_assistant.plots import plot_latency, plot_latency_vs_tokens


def test_plot_latency_saves_file(tmp_path):
    df = pd.DataFrame({
        "config": ["a", "a", "a", "b", "b", "b"],
        "latency_ms": [1000, 1100, 1200, 3000, 3100, 3200],
    })
    out = tmp_path / "sub" / "plot.png"
    fig = plot_latency(df, out)
    assert out.exists()
    assert len(fig.axes[0].patches) == 2   # one bar per config


def test_plot_latency_vs_tokens_saves_file(tmp_path):
    df = pd.DataFrame({
        "config": ["a", "a", "b", "b"],
        "generated_tokens": [50, 60, 400, 450],
        "latency_ms": [3000, 3500, 22000, 25000],
    })
    out = tmp_path / "sub" / "scatter.png"
    fig = plot_latency_vs_tokens(df, out)
    assert out.exists()
    assert len(fig.axes[0].collections) == 2   # one point cloud per config


def test_plot_latency_vs_tokens_accepts_str_path(tmp_path):
    df = pd.DataFrame({
        "config": ["a", "b"],
        "generated_tokens": [50, 400],
        "latency_ms": [3000, 22000],
    })
    out = str(tmp_path / "scatter.png")      # a plain string, not a Path
    plot_latency_vs_tokens(df, out)