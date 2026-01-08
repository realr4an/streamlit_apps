# ------------------------------------------------------------
# qrei_locubs_single_plot.py
# Single plot with selectable strategy/behavior for each target metric
# ------------------------------------------------------------
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
OBSERVED_FILE = BASE_DIR / "Simulation_results_24.xlsx"

ZONE_ORDER = ["BU", "TD", "RA", "SQ"]
SOURCE_ORDER = ["FIX", "NO", "EXP"]

ZONE_MAP = {
    "BU": "Bottom-up",
    "TD": "Top-down",
    "RA": "Random",
    "SQ": "Shortest queue",
}
ZONING_NORMALIZE = {
    "BA": "BU",
    "BOTTOM-UP": "BU",
    "BOTTOM UP": "BU",
    "TOP-DOWN": "TD",
    "TOP DOWN": "TD",
    "RANDOM": "RA",
    "SHORTEST QUEUE": "SQ",
}
SOURCE_MAP = {"FIX": "Fixed", "NO": "Normal", "EXP": "Exponential"}
SOURCE_NORMALIZE = {
    "FIXED": "FIX",
    "FIX": "FIX",
    "TA": "FIX",
    "EX": "EXP",
    "EXP": "EXP",
    "EXPONENTIAL": "EXP",
    "EXPO": "EXP",
    "EXP.": "EXP",
    "NORMAL": "NO",
    "NORM": "NO",
    "NO": "NO",
}

TARGET_CONFIGS: dict[str, dict] = {
    "Mean": {
        "prediction_files": {"delta interval": BASE_DIR / "QREI_Meanopt.xlsx"},
        "prediction_column": "predicted_mopt",
        "observed_column": "mopt",
        "title": "LOCUBs of mean order processing time",
        "y_title": "Mean order processing time (sec)",
        "hover_label": "Mean order processing time",
    },
    "Median": {
        "prediction_files": {"delta interval": BASE_DIR / "QREI_medianopt.xlsx"},
        "prediction_column": "predicted_medianopt",
        "observed_column": "median opt",
        "title": "LOCUBs of median order processing time",
        "y_title": "Median order processing time (sec)",
        "hover_label": "Median order processing time",
    },
    "75% Quantile": {
        "prediction_files": {"delta interval": BASE_DIR / "QREI_75opt.xlsx"},
        "prediction_column": "predicted_75opt",
        "observed_column": "75% opt",
        "title": "LOCUBs of 75% quantile order processing time",
        "y_title": "75% quantile order processing time (sec)",
        "hover_label": "75% quantile order processing time",
    },
    "90% Quantile": {
        "prediction_files": {"delta interval": BASE_DIR / "QREI_90opt.xlsx"},
        "prediction_column": "predicted_90opt",
        "observed_column": "90% opt",
        "title": "LOCUBs of 90% quantile order processing time",
        "y_title": "90% quantile order processing time (sec)",
        "hover_label": "90% quantile order processing time",
    },
}


def _normalize_source_value(val: str) -> str:
    s = str(val).upper().strip()
    s = SOURCE_NORMALIZE.get(s, s)
    if s.startswith("FIX") or s == "TA":
        return "FIX"
    if s.startswith("EX") or s.startswith("EXP"):
        return "EXP"
    if s.startswith("NO") or s.startswith("NOR"):
        return "NO"
    return s


def _decode_mean_arrival(coded: np.ndarray | pd.Series | float) -> np.ndarray:
    decoded = 20.0 + 10.0 * np.asarray(coded, dtype=float)
    return np.clip(decoded, 10.0, 30.0)


def _rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _coerce_numeric(series: pd.Series) -> pd.Series:
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


def _resolve_xcol(df: pd.DataFrame) -> str:
    candidates = [
        "systemload",
        "coded_sourceparameter",
        "coded_meanarrivaltime",
        "coded_mean_arrival_time",
        "coded_mean_interarrival_time",
        "coded_mean_interarrival",
        "coded_mean",
        "coded_arrival_time",
        "coded_interarrival_time",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
        except Exception:
            continue
    return candidates[0]


def _extend_to_limits(
    x: np.ndarray, y: np.ndarray, left: float, right: float
) -> tuple[np.ndarray, np.ndarray]:
    if len(x) == 0:
        return x, y
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if x[0] > left:
        if len(x) >= 2 and (x[1] - x[0]) != 0:
            slope = (y[1] - y[0]) / (x[1] - x[0])
        else:
            slope = 0.0
        y_left = y[0] + slope * (left - x[0])
        x = np.insert(x, 0, left)
        y = np.insert(y, 0, y_left)
    if x[-1] < right:
        if len(x) >= 2 and (x[-1] - x[-2]) != 0:
            slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        else:
            slope = 0.0
        y_right = y[-1] + slope * (right - x[-1])
        x = np.append(x, right)
        y = np.append(y, y_right)
    return x, y


def _safe_min_max(values: pd.Series) -> tuple[float | None, float | None]:
    if values is None or values.empty:
        return None, None
    vmin = values.min(skipna=True)
    vmax = values.max(skipna=True)
    if pd.isna(vmin) or pd.isna(vmax):
        return None, None
    return float(vmin), float(vmax)


def _compute_y_range(df: pd.DataFrame, low_col: str | None, up_col: str | None) -> list[float] | None:
    parts = []
    if low_col and low_col in df.columns:
        parts.append(df[low_col])
    if up_col and up_col in df.columns:
        parts.append(df[up_col])
    if "prediction" in df.columns:
        parts.append(df["prediction"])
    if not parts:
        return None
    combined = pd.concat(parts, ignore_index=True)
    vmin, vmax = _safe_min_max(combined)
    if vmin is None or vmax is None:
        return None
    pad = (vmax - vmin) * 0.05
    if pad == 0:
        pad = max(vmax * 0.05, 1.0)
    return [vmin - pad, vmax + pad]


@st.cache_data
def load_predictions(path: Path, prediction_column: str) -> pd.DataFrame:
    warnings.filterwarnings(
        "ignore",
        message="Workbook contains no default style",
        category=UserWarning,
        module="openpyxl",
    )
    df = pd.read_excel(path)
    rename_map = {
        "systemload": "systemload",
        "coded_sourceparameter": "systemload",
        "coded_meanarrivaltime": "systemload",
        "coded_mean_arrival_time": "systemload",
        "coded_mean_interarrival_time": "systemload",
        "coded_mean_interarrival": "systemload",
        "arrival_pattern": "source",
        "assignment_strategy": "zoning",
        "distributionstrategy": "zoning",
        "traycontrol": "zoning",
        "prediction": "prediction",
        prediction_column: "prediction",
        "low.delta": "low_delta",
        "up.delta": "up_delta",
        "low.corr": "low_corr",
        "up.corr": "up_corr",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    if "source" not in df.columns and "Source" in df.columns:
        df["source"] = df["Source"]
    if "source" not in df.columns:
        df["source"] = "ALL"
    for col in ["systemload", "prediction", "low_delta", "up_delta", "low_corr", "up_corr"]:
        if col in df.columns:
            df[col] = _coerce_numeric(df[col])
    if "zoning" in df.columns:
        df["zoning"] = (
            df["zoning"].astype(str).str.upper().str.strip().replace(ZONING_NORMALIZE)
        )
    if "source" in df.columns:
        df["source"] = df["source"].apply(_normalize_source_value)
    return df


@st.cache_data
def load_observed(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    warnings.filterwarnings(
        "ignore",
        message="Workbook contains no default style",
        category=UserWarning,
        module="openpyxl",
    )
    df = pd.read_excel(path)
    rename_map = {
        "coded_sourceparameter": "systemload",
        "coded_meanarrivaltime": "systemload",
        "coded_mean_arrival_time": "systemload",
        "coded_mean_interarrival_time": "systemload",
        "coded_mean_interarrival": "systemload",
        "assignment_strategy": "zoning",
        "distributionstrategy": "zoning",
        "traycontrol": "zoning",
        "arrival_pattern": "source",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    for col in ["systemload", "mopt", "median opt", "75% opt", "90% opt"]:
        if col in df.columns:
            df[col] = _coerce_numeric(df[col])
    if "zoning" in df.columns:
        df["zoning"] = (
            df["zoning"].astype(str).str.upper().str.strip().replace(ZONING_NORMALIZE)
        )
    if "source" in df.columns:
        df["source"] = df["source"].apply(_normalize_source_value)
    return df


def build_single_plot(
    df: pd.DataFrame,
    zone: str,
    sources: list[str],
    colors: dict[str, str],
    line_width: int,
    ribbon_alpha: float,
    font_size: int,
    y_title: str,
    hover_label: str,
    observed: pd.DataFrame | None = None,
    show_obs_points: bool = False,
    observed_value_column: str = "mopt",
) -> go.Figure:
    x_col = _resolve_xcol(df)
    x_title = "Mean interarrival time (sec)"

    d_zone = df[df["zoning"] == zone]
    if {"low_corr", "up_corr"}.issubset(d_zone.columns):
        low_col, up_col = "low_corr", "up_corr"
    elif {"low_delta", "up_delta"}.issubset(d_zone.columns):
        low_col, up_col = "low_delta", "up_delta"
    else:
        low_col, up_col = None, None
    has_bands = low_col is not None and up_col is not None

    y_range = _compute_y_range(d_zone[d_zone["source"].isin(sources)], low_col, up_col)

    obs = pd.DataFrame()
    if show_obs_points and observed is not None and not observed.empty:
        needed_cols = {"zoning", "source", x_col, observed_value_column}
        if needed_cols.issubset(set(observed.columns)):
            obs = observed[
                (observed["zoning"] == zone)
                & (observed["source"].isin(sources))
            ].copy()

    fig = go.Figure()
    x_left, x_right = -1.1, 1.1

    for src in sources:
        s = d_zone[d_zone["source"] == src].sort_values(x_col)
        if s.empty:
            continue

        if has_bands:
            x_low, y_low = _extend_to_limits(
                s[x_col].to_numpy(dtype=float),
                s[low_col].to_numpy(dtype=float),
                x_left,
                x_right,
            )
            x_up, y_up = _extend_to_limits(
                s[x_col].to_numpy(dtype=float),
                s[up_col].to_numpy(dtype=float),
                x_left,
                x_right,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_low,
                    y=y_low,
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=src,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_up,
                    y=y_up,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=_rgba(colors.get(src, "#888888"), ribbon_alpha),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=src,
                )
            )

        x_pred, y_pred = _extend_to_limits(
            s[x_col].to_numpy(dtype=float),
            s["prediction"].to_numpy(dtype=float),
            x_left,
            x_right,
        )
        decoded_pred = _decode_mean_arrival(x_pred)
        fig.add_trace(
            go.Scatter(
                x=x_pred,
                y=y_pred,
                mode="lines",
                name=SOURCE_MAP.get(src, src),
                line=dict(color=colors.get(src, "#444444"), width=line_width),
                legendgroup=src,
                customdata=np.column_stack((decoded_pred,)),
                hovertemplate=(
                    f"Routing strategy: {ZONE_MAP.get(zone, zone)}<br>"
                    f"Interarrival time behavior: {SOURCE_MAP.get(src, src)}<br>"
                    "Mean interarrival time: %{customdata[0]:.2f} sec<br>"
                    f"{hover_label}: %{{y:.2f}} sec<extra></extra>"
                ),
            )
        )

    if show_obs_points and not obs.empty:
        for src in sources:
            o = obs[obs["source"] == src].sort_values(x_col)
            if o.empty or o[observed_value_column].isna().all():
                continue
            decoded_obs = _decode_mean_arrival(o[x_col].to_numpy(dtype=float))
            fig.add_trace(
                go.Scatter(
                    x=o[x_col],
                    y=o[observed_value_column],
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=6,
                        color=colors.get(src, "#666"),
                        line=dict(width=0.5, color="#222"),
                    ),
                    name="Observation",
                    showlegend=False,
                    legendgroup=src,
                    customdata=np.column_stack((decoded_obs,)),
                    hovertemplate=(
                        "Observation<br>"
                        f"Routing strategy: {ZONE_MAP.get(zone, zone)}<br>"
                        f"Interarrival time behavior: {SOURCE_MAP.get(src, src)}<br>"
                        "Mean interarrival time: %{customdata[0]:.2f} sec<br>"
                        f"{hover_label}: %{{y:.2f}} sec<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        height=700,
        width=700,
        margin=dict(l=54, r=40, t=28, b=50),
        legend=dict(
            title=dict(text="Interarrival time behavior", font=dict(size=font_size - 2, color="#000000")),
            font=dict(size=font_size - 2, color="#000000"),
            groupclick="togglegroup",
        ),
        font=dict(size=font_size, color="#000000"),
    )
    fig.update_xaxes(
        title_text=x_title,
        range=[-1.1, 1.1],
        autorange=False,
        tickmode="array",
        tickvals=[-1, -0.5, 0, 0.5, 1],
        ticktext=["10", "15", "20", "25", "30"],
        zeroline=False,
        title_font=dict(size=font_size, color="#000000"),
        tickfont=dict(size=font_size - 2, color="#000000"),
    )
    y_axis = dict(
        title_text=y_title,
        title_font=dict(size=font_size, color="#000000"),
        tickfont=dict(size=font_size - 2, color="#000000"),
        zeroline=False,
    )
    if y_range is not None:
        y_axis["range"] = y_range
        y_axis["autorange"] = False
    fig.update_yaxes(**y_axis)
    return fig


def _available_targets() -> dict[str, dict]:
    available = {}
    for name, cfg in TARGET_CONFIGS.items():
        preds = {
            label: path for label, path in cfg.get("prediction_files", {}).items()
            if path.exists()
        }
        if preds:
            available[name] = {**cfg, "prediction_files": preds}
    return available


def main() -> None:
    st.set_page_config(page_title="LOCUBs (Single Plot)", layout="wide")
    st.sidebar.header("Display")

    available = _available_targets()
    if not available:
        st.error("No QREI datasets found next to this script.")
        st.stop()

    target_labels = list(available.keys())
    target_choice = st.sidebar.radio("Target metric", options=target_labels)
    cfg = available[target_choice]

    st.markdown(f"### {cfg['title']}")

    predictions = cfg["prediction_files"]
    interval_labels = list(predictions.keys())
    if len(interval_labels) == 1:
        chosen_interval = interval_labels[0]
    else:
        chosen_interval = st.sidebar.selectbox("Prediction interval", interval_labels)

    df = load_predictions(predictions[chosen_interval], cfg["prediction_column"])
    observed_df = load_observed(OBSERVED_FILE)

    zones = [z for z in ZONE_ORDER if z in df["zoning"].dropna().unique().tolist()]
    zone = zones[0] if zones else None
    if zone:
        zone = st.sidebar.selectbox(
            "Routing strategy",
            options=zones,
            index=zones.index(zone),
            format_func=lambda z: ZONE_MAP.get(z, z),
        )

    sources_all = [s for s in SOURCE_ORDER if s in df["source"].dropna().unique().tolist()]
    sources = st.sidebar.multiselect(
        "Interarrival time behavior",
        options=sources_all,
        default=sources_all,
        format_func=lambda s: SOURCE_MAP.get(s, s),
    )

    show_obs_points = st.sidebar.checkbox("Show observed values", value=not observed_df.empty)
    if show_obs_points and observed_df.empty:
        st.sidebar.info("No observed values available for this dataset.")

    st.sidebar.markdown("---")
    st.sidebar.caption("Colors")
    col_fix = st.sidebar.color_picker("Fixed", "#D55E00")
    col_no = st.sidebar.color_picker("Normal", "#0072B2")
    col_exp = st.sidebar.color_picker("Exponential", "#009E73")
    colors = {"FIX": col_fix, "NO": col_no, "EXP": col_exp}
    line_width = st.sidebar.slider("Line width", 1, 6, 2, 1)
    font_size = st.sidebar.slider("Base font size", 10, 40, 20, 1)
    plot_size = st.sidebar.slider("Plot size (px)", 400, 900, 700, 10)
    ribbon_alpha = st.sidebar.slider("Ribbon transparency", 0.05, 0.9, 0.18, 0.01)

    if zone and sources:
        fig = build_single_plot(
            df,
            zone,
            sources,
            colors,
            line_width,
            ribbon_alpha,
            font_size,
            cfg["y_title"],
            cfg["hover_label"],
            observed=observed_df,
            show_obs_points=show_obs_points,
            observed_value_column=cfg["observed_column"],
        )
        fig.update_layout(width=plot_size, height=plot_size)
        st.plotly_chart(fig, use_container_width=False)
        if not ({"low_delta", "up_delta"}.issubset(df.columns) or {"low_corr", "up_corr"}.issubset(df.columns)):
            st.warning("No interval bands found in the selected dataset - only the central line is drawn.")
    else:
        st.info("Select at least one routing strategy and one interarrival time pattern.")


if __name__ == "__main__":
    main()
