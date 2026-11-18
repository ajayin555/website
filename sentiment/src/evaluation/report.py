from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def save_metrics_table(
    metrics_df: pd.DataFrame,
    regime: str,
    output_dir: str,
) -> str:
    out = Path(output_dir) / f"metrics_{regime}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out)
    logger.info("Metrics table saved to %s", out)
    print(f"\n=== Performance Metrics — {regime} ===")
    print(metrics_df.to_string())
    return str(out)


def generate_regime_comparison(
    high_vol_metrics: pd.DataFrame,
    normal_metrics: pd.DataFrame,
) -> pd.DataFrame:
    hv = high_vol_metrics.add_suffix("_high_vol")
    nm = normal_metrics.add_suffix("_normal")
    return hv.join(nm, how="outer")


def plot_cumulative_returns(
    portfolios: dict[str, pd.Series],
    regime: str,
    output_dir: str,
) -> None:
    out = Path(output_dir) / f"cumulative_returns_{regime}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, rets in portfolios.items():
        cum = (1 + rets).cumprod()
        ax.plot(cum.index, cum.values, label=name)

    ax.set_title(f"Cumulative Returns — {regime.replace('_', ' ').title()}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (base 1.0)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Plot saved to %s", out)
