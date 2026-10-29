import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_latency(
    df: pd.DataFrame,
    path: str | Path,
    title: str = "Latence par configuration",
) -> plt.Figure:
    """Bar chart of median latency per config, with p10-p90 error bars.

    :param df: Clean benchmark results (warmup removed), with columns `config` and `latency_ms`.
    :param path: Where to save the PNG. Missing folders are created.
    :param title: Figure title. Write the conclusion, not a description.
    :return: The matplotlib figure (already closed: use display(fig) in a notebook).
    """
    # 1. Summary table in seconds, fastest config first
    lat = df.groupby("config")["latency_ms"]
    summary = pd.DataFrame({
        "p10": lat.quantile(0.10),
        "median": lat.median(),
        "p90": lat.quantile(0.90),
    }) / 1000
    summary = summary.sort_values("median")

    # 2. Error bar lengths: distances from the top of each bar, not positions
    lower = summary["median"] - summary["p10"]
    upper = summary["p90"] - summary["median"]

    # 3. Draw
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary.index, summary["median"], yerr=[lower, upper], capsize=6)

    # 4. Labels, title, layout
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Latence médiane (s)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()

    # 5. Save and return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return fig

def plot_latency_vs_tokens(df , path):
    """
    Plot latency vs generated tokens for each config in the DataFrame.

    :param df: DataFrame containing 'config', 'generated_tokens', and 'latency_ms' columns.
    :param path: Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    for config, group in df.groupby("config"):
        ax.scatter(group["generated_tokens"], group["latency_ms"] / 1000,
                   alpha=0.6, label=config)

    ax.set_xlabel("Generated Tokens")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency vs Generated Tokens by Config")
    ax.legend()
    plt.tight_layout()
    # Créer le répertoire si nécessaire
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close(fig)
    return fig