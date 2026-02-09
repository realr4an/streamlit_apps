# ------------------------------------------------------------
# generate_locubs_static.py
# Create static LOCUB plots (no observations) for all XLSX files in this folder
# ------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

# Output modes (toggle what gets generated).
# The user request for this run: one PDF per XLSX in the usual 2x2 layout
# (4 routing strategies in one PDF).
GENERATE_FACETED_BY_STRATEGY = True
GENERATE_GRID_4X3 = False
GENERATE_SPLIT_BY_STRATEGY = False

plt.rcParams.update(
    {
        "font.size": 13,
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "legend.title_fontsize": 12,
    }
)

ZONE_ORDER = ["BU", "TD", "RA", "SQ"]
SOURCE_ORDER = ["FIX", "NO", "EXP"]

ZONE_LABELS = {
    "BU": "Bottom-up",
    "TD": "Top-down",
    "RA": "Random",
    "SQ": "Shortest queue",
}
SOURCE_LABELS = {
    "FIX": "Fixed",
    "NO": "Truncated normal",
    "EXP": "Exponential",
}
SOURCE_COLORS = {
    "FIX": "#d62728",
    "NO": "#7f7fdb",
    "EXP": "#98df8a",
}
STRATEGY_COLORS = {
    "BU": "#d62728",
    "TD": "#7f7fdb",
    "RA": "#98df8a",
    "SQ": "#ffe600",
}

LABEL_MAP = {
    "predicted_mopt": "Mean\norder processing time (sec)",
    "predicted_90opt": "90% quantile of\norder processing time (sec)",
    "predicted_U_bottomPos": "Mean utilization rate\nfor bottom stations (%)",
    "predicted_U_bottommiddlePos": "Mean utilization rate\nfor bottom middle stations (%)",
    "predicted_U_topmiddlePos": "Mean utilization rate\nfor top middle stations (%)",
    "predicted_U_topPos": "Mean utilization rate\nfor top stations (%)",
    "predicted_U1": "Utilization rate at P1 (%)",
    "predicted_U2": "Utilization rate at P2 (%)",
    "predicted_U3": "Utilization rate at P3 (%)",
    "predicted_U4": "Utilization rate at P4 (%)",
    "predicted_U5": "Utilization rate at P5 (%)",
    "predicted_U6": "Utilization rate at P6 (%)",
    "predicted_U7": "Utilization rate at P7 (%)",
    "predicted_U8": "Utilization rate at P8 (%)",
}

SINGLE_STRATEGY_PRED = {
    "predicted_U_bottommiddlePos",
}


def coerce_numeric(series: pd.Series) -> pd.Series:
    s1 = pd.to_numeric(series, errors="coerce")
    if s1.notna().any() and s1.isna().sum() == 0:
        return s1
    s2 = pd.to_numeric(
        series.astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace("\u00A0", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )
    return s1.where(s1.notna(), s2)


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    rename_map = {
        "assignment_strategy": "zoning",
        "arrival_pattern": "source",
        "coded_meanarrivaltime": "x",
        "coded_mean_arrival_time": "x",
        "coded_mean_interarrival_time": "x",
        "coded_mean_interarrival": "x",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    for col in ["x", "low.delta", "up.delta"]:
        if col in df.columns:
            df[col] = coerce_numeric(df[col])
    if "zoning" in df.columns:
        df["zoning"] = df["zoning"].astype(str).str.upper().str.strip()
    if "source" in df.columns:
        df["source"] = df["source"].astype(str).str.upper().str.strip()
    return df


def resolve_predicted_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if c.startswith("predicted_")]
    if not candidates:
        raise ValueError("No predicted_* column found.")
    if len(candidates) > 1:
        candidates = sorted(candidates)
    return candidates[0]


def label_from_predicted_column(pred_col: str) -> str:
    return LABEL_MAP.get(pred_col, pred_col.replace("predicted_", "").replace("_", " ").strip())


def resolve_y_label(path: Path, pred_col: str) -> str:
    stem = path.stem.replace(" ", "_").upper()
    if "U1" in stem:
        return "Utilization rate at P1 (%)"
    if "U2" in stem:
        return "Utilization rate at P2 (%)"
    if "U3" in stem:
        return "Utilization rate at P3 (%)"
    if "U4" in stem:
        return "Utilization rate at P4 (%)"
    if "U5" in stem:
        return "Utilization rate at P5 (%)"
    if "U6" in stem:
        return "Utilization rate at P6 (%)"
    if "U7" in stem:
        return "Utilization rate at P7 (%)"
    if "U8" in stem:
        return "Utilization rate at P8 (%)"
    return label_from_predicted_column(pred_col)


def compute_y_range(df: pd.DataFrame, pred_col: str) -> tuple[float, float]:
    if pred_col in {"predicted_mopt", "predicted_90opt"}:
        return 100.0, 300.0
    if pred_col.startswith("predicted_U_") or pred_col in {
        "predicted_U1",
        "predicted_U2",
        "predicted_U3",
        "predicted_U4",
        "predicted_U5",
        "predicted_U6",
        "predicted_U7",
        "predicted_U8",
    }:
        return 0.0, 100.0
    parts = []
    for col in [pred_col, "low.delta", "up.delta"]:
        if col in df.columns:
            parts.append(df[col])
    if not parts:
        return 0.0, 1.0
    combined = pd.concat(parts, ignore_index=True)
    vmin = float(combined.min(skipna=True))
    vmax = float(combined.max(skipna=True))
    pad = (vmax - vmin) * 0.05
    if pad == 0:
        pad = max(vmax * 0.05, 1.0)
    return vmin - pad, vmax + pad


def is_utilization(pred_col: str, path: Path | None = None) -> bool:
    if pred_col.startswith("predicted_U_") or pred_col in {
        "predicted_U1",
        "predicted_U2",
        "predicted_U3",
        "predicted_U4",
        "predicted_U5",
        "predicted_U6",
        "predicted_U7",
        "predicted_U8",
    }:
        return True
    if path is not None:
        stem = path.stem.replace(" ", "_").upper()
        return stem.startswith("QREI_U")
    return False


def output_path(path: Path, suffix: str | None = None) -> Path:
    stem = path.stem.replace(" ", "_")
    if stem.startswith("QREI_"):
        stem = stem[len("QREI_"):]
    if stem.endswith("_1") or stem.endswith("_2"):
        stem = stem[:-2]
    if suffix:
        return path.with_name(f"LOCUB_{stem}_{suffix}.pdf")
    return path.with_name(f"LOCUB_{stem}.pdf")


def plot_faceted_by_zone(
    df: pd.DataFrame, pred_col: str, out_path: Path, y_label: str | None = None, source_path: Path | None = None
) -> None:
    zones = [z for z in ZONE_ORDER if z in df["zoning"].dropna().unique().tolist()]
    sources = [s for s in SOURCE_ORDER if s in df["source"].dropna().unique().tolist()]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=False, sharex=True, sharey=True)
    axes = axes.flatten()
    fig.patch.set_facecolor("white")

    y_min, y_max = compute_y_range(df, pred_col)
    x_label = "Mean interarrival time (sec)"
    y_label = y_label or label_from_predicted_column(pred_col)

    for idx, zone in enumerate(ZONE_ORDER):
        ax = axes[idx]
        if zone not in zones:
            ax.axis("off")
            continue

        d_zone = df[df["zoning"] == zone]
        for src in sources:
            d = d_zone[d_zone["source"] == src].sort_values("x")
            if d.empty:
                continue
            x_vals = d["x"].to_numpy(dtype=float)
            y_vals = d[pred_col].to_numpy(dtype=float)
            ax.plot(
                x_vals,
                y_vals,
                color=SOURCE_COLORS.get(src, "#444444"),
                linewidth=1.8,
                label=SOURCE_LABELS.get(src, src),
            )
            if "low.delta" in d.columns and "up.delta" in d.columns:
                y_low = d["low.delta"].to_numpy(dtype=float)
                y_up = d["up.delta"].to_numpy(dtype=float)
                ax.fill_between(
                    x_vals,
                    y_low,
                    y_up,
                    color=SOURCE_COLORS.get(src, "#888888"),
                    alpha=0.18,
                    linewidth=0.0,
                )

        ax.set_title(ZONE_LABELS.get(zone, zone))
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xticklabels(["10", "15", "20", "25", "30"])
        if is_utilization(pred_col, source_path):
            ax.set_yticks([0, 25, 50, 75, 100])
        if idx in (0, 1):
            ax.tick_params(axis="x", labelbottom=True)
        ax.set_facecolor("white")
        ax.grid(True, color="#e5e5e5")
        ax.yaxis.set_label_position("left")
        ax.yaxis.tick_left()
        ax.tick_params(axis="y", labelleft=True, labelright=False)
        ax.yaxis.labelpad = 10

    legend_title = "Interarrival time behavior"
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            frameon=True,
            title=legend_title,
            ncol=len(labels),
            bbox_to_anchor=(0.5, -0.05),
        )

    fig.subplots_adjust(bottom=0.12, wspace=0.35, hspace=0.28)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_single_by_strategy(
    df: pd.DataFrame, pred_col: str, out_path: Path, y_label: str | None = None, source_path: Path | None = None
) -> None:
    strategies = [z for z in ZONE_ORDER if z in df["zoning"].dropna().unique().tolist()]

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=False)
    fig.patch.set_facecolor("white")

    y_min, y_max = compute_y_range(df, pred_col)
    x_label = "Mean interarrival time (sec)"
    y_label = y_label or label_from_predicted_column(pred_col)

    for strat in strategies:
        d = df[df["zoning"] == strat].sort_values("x")
        if d.empty:
            continue
        x_vals = d["x"].to_numpy(dtype=float)
        y_vals = d[pred_col].to_numpy(dtype=float)
        ax.plot(
            x_vals,
            y_vals,
            color=STRATEGY_COLORS.get(strat, "#444444"),
            linewidth=1.8,
            label=ZONE_LABELS.get(strat, strat),
        )
        if "low.delta" in d.columns and "up.delta" in d.columns:
            y_low = d["low.delta"].to_numpy(dtype=float)
            y_up = d["up.delta"].to_numpy(dtype=float)
            ax.fill_between(
                x_vals,
                y_low,
                y_up,
                color=STRATEGY_COLORS.get(strat, "#888888"),
                alpha=0.18,
                linewidth=0.0,
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(["10", "15", "20", "25", "30"])
    if is_utilization(pred_col, source_path):
        ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_facecolor("white")
    ax.grid(True, color="#e5e5e5")
    ax.legend(
        title="Routing strategy",
        frameon=True,
        loc="lower center",
        ncol=len(strategies),
        bbox_to_anchor=(0.5, -0.28),
    )

    fig.subplots_adjust(bottom=0.22)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_grid_4x3(
    df: pd.DataFrame, pred_col: str, out_path: Path, y_label: str | None = None, source_path: Path | None = None
) -> None:
    zones = [z for z in ZONE_ORDER if z in df["zoning"].dropna().unique().tolist()]
    sources = [s for s in SOURCE_ORDER if s in df["source"].dropna().unique().tolist()]

    fig, axes = plt.subplots(4, 3, figsize=(12, 12), constrained_layout=False, sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    y_min, y_max = compute_y_range(df, pred_col)
    x_label = "Mean interarrival time (sec)"
    y_label = y_label or label_from_predicted_column(pred_col)

    for r, zone in enumerate(ZONE_ORDER):
        for c, src in enumerate(SOURCE_ORDER):
            ax = axes[r, c]
            if zone not in zones or src not in sources:
                ax.axis("off")
                continue
            d = df[(df["zoning"] == zone) & (df["source"] == src)].sort_values("x")
            if d.empty:
                ax.axis("off")
                continue

            x_vals = d["x"].to_numpy(dtype=float)
            y_vals = d[pred_col].to_numpy(dtype=float)
            color = SOURCE_COLORS.get(src, "#444444")
            ax.plot(x_vals, y_vals, color=color, linewidth=1.6)
            if "low.delta" in d.columns and "up.delta" in d.columns:
                y_low = d["low.delta"].to_numpy(dtype=float)
                y_up = d["up.delta"].to_numpy(dtype=float)
                ax.fill_between(x_vals, y_low, y_up, color=color, alpha=0.18, linewidth=0.0)

            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks([-1, -0.5, 0, 0.5, 1])
            ax.set_xticklabels(["10", "15", "20", "25", "30"])
            if is_utilization(pred_col, source_path):
                ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_facecolor("white")
            ax.grid(True, color="#e5e5e5")

            # show ticks/labels on every subplot, but axis titles only on outer edges
            if r == 3:
                ax.set_xlabel(x_label)
            if c == 0:
                ax.set_ylabel(y_label)
            ax.set_title(f"{ZONE_LABELS.get(zone, zone)} / {SOURCE_LABELS.get(src, src)}")

            ax.tick_params(axis="x", labelbottom=True)
            ax.tick_params(axis="y", labelleft=True)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.08, wspace=0.35, hspace=0.35)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_split_by_strategy(
    df: pd.DataFrame, pred_col: str, base_out_path: Path, y_label: str | None = None, source_path: Path | None = None
) -> list[Path]:
    """
    Create one PDF per routing strategy (BU/TD/RA/SQ), with the 3 interarrival behaviors
    (Fixed/Truncated normal/Exponential) as lines + uncertainty ribbons.
    """
    zones = [z for z in ZONE_ORDER if z in df["zoning"].dropna().unique().tolist()]
    sources = [s for s in SOURCE_ORDER if s in df["source"].dropna().unique().tolist()]

    y_min, y_max = compute_y_range(df, pred_col)
    x_label = "Mean interarrival time (sec)"
    y_label = y_label or label_from_predicted_column(pred_col)

    outputs: list[Path] = []
    for zone in zones:
        out_path = base_out_path.with_name(f"{base_out_path.stem}_{zone}.pdf")

        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=False)
        fig.patch.set_facecolor("white")

        d_zone = df[df["zoning"] == zone]
        for src in sources:
            d = d_zone[d_zone["source"] == src].sort_values("x")
            if d.empty:
                continue
            x_vals = d["x"].to_numpy(dtype=float)
            y_vals = d[pred_col].to_numpy(dtype=float)
            color = SOURCE_COLORS.get(src, "#444444")
            ax.plot(
                x_vals,
                y_vals,
                color=color,
                linewidth=1.8,
                label=SOURCE_LABELS.get(src, src),
            )
            if "low.delta" in d.columns and "up.delta" in d.columns:
                y_low = d["low.delta"].to_numpy(dtype=float)
                y_up = d["up.delta"].to_numpy(dtype=float)
                ax.fill_between(
                    x_vals,
                    y_low,
                    y_up,
                    color=color,
                    alpha=0.18,
                    linewidth=0.0,
                )

        ax.set_title(ZONE_LABELS.get(zone, zone))
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xticklabels(["10", "15", "20", "25", "30"])
        if is_utilization(pred_col, source_path):
            ax.set_yticks([0, 25, 50, 75, 100])

        ax.set_facecolor("white")
        ax.grid(True, color="#e5e5e5")
        ax.yaxis.set_label_position("left")
        ax.yaxis.tick_left()
        ax.tick_params(axis="y", labelleft=True, labelright=False)
        ax.yaxis.labelpad = 10

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                title="Interarrival time behavior",
                frameon=True,
                loc="lower center",
                ncol=len(labels),
                bbox_to_anchor=(0.5, -0.28),
            )

        fig.subplots_adjust(bottom=0.22)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        outputs.append(out_path)

    return outputs


def plot_file(path: Path) -> Path:
    df = load_predictions(path)
    pred_col = resolve_predicted_column(df)
    df[pred_col] = coerce_numeric(df[pred_col])

    y_label = resolve_y_label(path, pred_col)
    out_path = output_path(path)

    generated: list[Path] = []

    if GENERATE_SPLIT_BY_STRATEGY and pred_col not in SINGLE_STRATEGY_PRED:
        generated.extend(plot_split_by_strategy(df, pred_col, out_path, y_label=y_label, source_path=path))

    if GENERATE_FACETED_BY_STRATEGY:
        if pred_col in SINGLE_STRATEGY_PRED:
            plot_single_by_strategy(df, pred_col, out_path, y_label=y_label, source_path=path)
        else:
            plot_faceted_by_zone(df, pred_col, out_path, y_label=y_label, source_path=path)
        generated.append(out_path)

    if GENERATE_GRID_4X3:
        grid_path = output_path(path, "4x3")
        plot_grid_4x3(df, pred_col, grid_path, y_label=y_label, source_path=path)
        generated.append(grid_path)

    # For compatibility, return the primary output path even if multiple PDFs were created.
    return generated[0] if generated else out_path


def main() -> None:
    files = sorted(BASE_DIR.glob("*.xlsx"))
    if not files:
        raise SystemExit("No xlsx files found in static_plots.")

    outputs = []
    for path in files:
        outputs.append(plot_file(path))

    print("Generated:")
    for out in outputs:
        print(out.name)


if __name__ == "__main__":
    main()
