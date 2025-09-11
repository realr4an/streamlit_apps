# ------------------------------------------------------------
# throughput_single_plot_app.py
# 2D-Plot (encoded system load –1…+1) with selectable prediction intervals
# tuned for: data.throughput.2d_10_30 4.xlsx
# ------------------------------------------------------------
from pathlib import Path
import warnings
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np

# Base directory of this script (needed for file lookups)
BASE_DIR = Path(__file__).resolve().parent

# Path to the observation file
OBS_DATA_FILE = BASE_DIR / "data.xlsx"

# Available throughput prediction files (asymptotic vs finite-sample)
def _available_throughput_files() -> dict[str, Path]:
    mapping = {
        "asymptotic normal": BASE_DIR / "data.throughput_vollesDesign_r2.xlsx",
        "with finite sample correction": BASE_DIR / "data.throughput_vollesDesign_r2 1.xlsx",
    }
    return {k: p for k, p in mapping.items() if p.exists()}

# ----------------------- Mappings ---------------------------
ZONE_MAP = {
    "BU": "Bottom-Up",
    "TD": "Top-Down",
    "RA": "Random",
    "SQ": "Shortest Queue",
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

def _normalize_source_value(val: str) -> str:
    s = str(val).upper().strip()
    s = SOURCE_NORMALIZE.get(s, s)
    if s.startswith("FIX"):
        return "FIX"
    if s == "TA":
        return "FIX"
    if s.startswith("EX") or s.startswith("EXP"):
        return "EXP"
    if s.startswith("NO") or s.startswith("NOR"):
        return "NO"
    return s

# Optional normalization for typos/variants
ZONING_NORMALIZE = {
    "BA": "BU", "BOTTOM-UP": "BU", "BOTTOM UP": "BU",
    "TOP-DOWN": "TD", "TOP DOWN": "TD",
    "RANDOM": "RA",
    "SHORTEST QUEUE": "SQ",
}

# ----------------------- Find file -----------------------
def _find_data_file() -> Path:
    patterns = [
        "data.throughput_vollesDesign_r2.xlsx"
    ]
    for pat in patterns:
        cands = sorted(BASE_DIR.glob(pat))
        if cands:
            return cands[0]
    raise FileNotFoundError("No matching Excel file found.")

# Robust numeric coercion that also handles decimal commas
def _coerce_numeric(series: pd.Series) -> pd.Series:
    s1 = pd.to_numeric(series, errors="coerce")
    if s1.notna().any() and s1.isna().sum() == 0:
        return s1
    # Try replacing comma decimals; keep digits and minus
    s2 = pd.to_numeric(series.astype(str).str.replace(" ", "", regex=False)
                       .str.replace("\u00A0", "", regex=False)
                       .str.replace(".", "", regex=False)  # remove thousand separator if present
                       .str.replace(",", ".", regex=False), errors="coerce")
    # Prefer s1 where valid, otherwise s2
    return s1.where(s1.notna(), s2)

# Robustly resolve the X column used for coded mean arrival time
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
    # fallback: first numeric column
    for c in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
        except Exception:
            continue
    # last resort
    return candidates[0]

# ----------------------- Load & Prepare ----------------
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    warnings.filterwarnings("ignore", message="Workbook contains no default style",
                            category=UserWarning, module="openpyxl")
    df = pd.read_excel(path)

    # Harmonize columns to internal names
    rename_map = {
        # X
        "systemload": "systemload",
        "coded_sourceparameter": "systemload",
        # new dataset variants for coded mean arrival time
        "coded_meanarrivaltime": "systemload",
        "coded_mean_arrival_time": "systemload",
        "coded_mean_interarrival_time": "systemload",
        "coded_mean_interarrival": "systemload",
        # Facet
        "traycontrol": "zoning",
        "distributionstrategy": "zoning",
        "assignment_strategy": "zoning",
        # arrival pattern/source codes
        "arrival_pattern": "source",
        # y
        "prediction": "prediction",
        "predicted_throughput": "prediction",
        "predicted_mopt": "prediction",
        # Intervals (asymptotic / finite-sample corrected)
        "low.delta": "low_delta",
        "up.delta":  "up_delta",
        "low.corr":  "low_corr",
        "up.corr":   "up_corr",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Source: "Source" (uppercase) → "source"
    if "Source" in df.columns and "source" not in df.columns:
        df["source"] = df["Source"]
    if "source" not in df.columns:
        df["source"] = "ALL"

    # Remove unnecessary index column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Ensure numeric types
    for c in ["systemload", "prediction", "low_delta", "up_delta", "low_corr", "up_corr"]:
        if c in df.columns:
            df[c] = _coerce_numeric(df[c])

    # Normalize categories (Upper/Trim + Mapping BA→BU etc.)
    if "zoning" in df.columns:
        z = df["zoning"].astype(str).str.upper().str.strip()
        df["zoning"] = z.replace(ZONING_NORMALIZE)
    if "source" in df.columns:
        s = df["source"].apply(_normalize_source_value)
        df["source"] = s

    return df

@st.cache_data
def load_observed(path: Path) -> pd.DataFrame:
    """Load raw dataset with observed throughput & mopt.
    Also fixes duplicate columns after rename (e.g. distributionstrategy + distributinstrategy → zoning).
    """
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_excel(path)
    rename_map = {
        "coded_sourceparameter": "systemload",
        "coded_meanarrivaltime": "systemload",
        "coded_mean_arrival_time": "systemload",
        "coded_mean_interarrival_time": "systemload",
        "coded_mean_interarrival": "systemload",
        "distributionstrategy": "zoning",
        "assignment_strategy": "zoning",
        "arrival_pattern": "source",
        "distributinstrategy": "zoning",  # Variant according to specification
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    def _collapse_duplicates(frame: pd.DataFrame, col_name: str) -> pd.DataFrame:
        if list(frame.columns).count(col_name) > 1:
            dups = frame.loc[:, frame.columns == col_name]
            merged = dups.bfill(axis=1).iloc[:, 0]
            frame = frame.loc[:, frame.columns != col_name]
            frame[col_name] = merged
        return frame

    for cn in ["zoning", "systemload", "source"]:
        df = _collapse_duplicates(df, cn)

    for c in ["systemload", "throughput", "mopt"]:
        if c in df.columns:
            df[c] = _coerce_numeric(df[c])

    if "zoning" in df.columns:
        zoning_col = df["zoning"]
        if isinstance(zoning_col, pd.DataFrame):
            zoning_col = zoning_col.iloc[:, 0]
        z = zoning_col.astype(str).str.upper().str.strip()
        df["zoning"] = z.replace(ZONING_NORMALIZE)
    if "source" in df.columns:
        source_col = df["source"]
        if isinstance(source_col, pd.DataFrame):
            source_col = source_col.iloc[:, 0]
        df["source"] = source_col.apply(_normalize_source_value)
    return df

def _rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"

# ----------------------- Build plot -------------------------
def build_single_plot(
    df: pd.DataFrame,
    zone: str,
    sources,
    colors: dict,
    line_width: int,
    ribbon_alpha: float = 0.18,
    observed: pd.DataFrame | None = None,
    show_observed: bool = True,
    font_size: int = 18,
) -> go.Figure:
    xcol = _resolve_xcol(df)  # kodiert –1…+1
    xtitle = "Mean arrival time (sec)"

    d = df[df["zoning"] == zone]
    if d.empty:
        st.warning(f"No data for zoning '{zone}'. Available: {sorted(df['zoning'].unique().tolist())}")

    order = [s for s in ["FIX", "NO", "EXP"] if s in sources] + [s for s in sources if s not in ["FIX","NO","EXP"]]
    # Choose interval columns automatically from what's available in the loaded dataset
    if {"low_corr", "up_corr"}.issubset(df.columns):
        low_col, up_col = "low_corr", "up_corr"
    elif {"low_delta", "up_delta"}.issubset(df.columns):
        low_col, up_col = "low_delta", "up_delta"
    else:
        low_col, up_col = None, None
    has_bands = low_col is not None and up_col is not None

    obs = pd.DataFrame()
    if show_observed and observed is not None and not observed.empty:
        obs = observed[(observed["zoning"] == zone) & (observed["source"].isin(sources))].copy()

    # Helper: extend line endpoints to axis limits so the curve reaches the frame border
    def _extend_to_limits(x: np.ndarray, y: np.ndarray, left: float, right: float) -> tuple[np.ndarray, np.ndarray]:
        if len(x) == 0:
            return x, y
        order = np.argsort(x)
        x = np.asarray(x, dtype=float)[order]
        y = np.asarray(y, dtype=float)[order]
        # Left side
        if x[0] > left:
            if len(x) >= 2 and (x[1] - x[0]) != 0:
                slope = (y[1] - y[0]) / (x[1] - x[0])
            else:
                slope = 0.0
            y_left = y[0] + slope * (left - x[0])
            x = np.insert(x, 0, left)
            y = np.insert(y, 0, y_left)
        # Right side
        if x[-1] < right:
            if len(x) >= 2 and (x[-1] - x[-2]) != 0:
                slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
            else:
                slope = 0.0
            y_right = y[-1] + slope * (right - x[-1])
            x = np.append(x, right)
            y = np.append(y, y_right)
        return x, y

    fig = go.Figure()
    # --- Forecast lines + intervals ---
    for src in order:
        s = d[d["source"] == src].sort_values(xcol)
        if s.empty:
            continue

        if has_bands and not s[[low_col, up_col]].isna().all().all():
            # Extend ribbons to the axis limits
            x_left, x_right = -1.1, 1.1
            x_low, y_low = _extend_to_limits(s[xcol].to_numpy(), s[low_col].to_numpy(), x_left, x_right)
            x_up,  y_up  = _extend_to_limits(s[xcol].to_numpy(), s[up_col].to_numpy(),  x_left, x_right)
            fig.add_trace(go.Scatter(
                x=x_low, y=y_low, mode="lines",
                line=dict(width=0), hoverinfo="skip",
                showlegend=False, legendgroup=src
            ))
            fig.add_trace(go.Scatter(
                x=x_up, y=y_up, mode="lines",
                line=dict(width=0), fill="tonexty",
                fillcolor=_rgba(colors.get(src, "#888888"), ribbon_alpha),
                hoverinfo="skip", showlegend=False, legendgroup=src
            ))

        # Extend central prediction to axis limits
        x_left, x_right = -1.1, 1.1
        x_pred, y_pred = _extend_to_limits(s[xcol].to_numpy(), s["prediction"].to_numpy(), x_left, x_right)
        fig.add_trace(go.Scatter(
            x=x_pred, y=y_pred, mode="lines",
            name=SOURCE_MAP.get(src, src),
            line=dict(color=colors.get(src, None), width=line_width),
            legendgroup=src
        ))

    # --- Observed points ---
    if not obs.empty:
        for src in order:
            src_pts = obs[obs["source"] == src]
            if src_pts.empty or src_pts["throughput"].isna().all():
                continue
            fig.add_trace(
                go.Scatter(
                    x=src_pts[xcol] if xcol in src_pts.columns else src_pts.get("systemload", src_pts.iloc[:,0]),
                    y=src_pts["throughput"],
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=7,
                        color=colors.get(src, "#666"),
                        line=dict(width=0.5, color="#222"),
                    ),
                    name="Observation (colored)",
                    legendgroup="obs",
                    showlegend=False,
                    hovertemplate="Observation<br>Mean arrival time: %{x}<br>Throughput: %{y}<extra></extra>",  # changed
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[0], y=[0], mode="markers",
                marker=dict(symbol="circle", size=7, color="#FFFFFF", line=dict(width=0.6, color="#000000")),
                name="Observation",
                legendgroup="obs",
                showlegend=True,
                visible="legendonly",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        height=700, width=700,
        # a touch more outer padding
        margin=dict(l=54, r=40, t=28, b=50),
        legend=dict(
            title=dict(text="Arrival pattern", font=dict(size=font_size-2, color="#000000")),
            font=dict(size=font_size-2, color="#000000")
        ),
        font=dict(size=font_size, color="#000000")
    )
    fig.update_xaxes(
        title_text=xtitle,
        # Match extended line ends so curves reach the frame
        range=[-1.1, 1.1],
        autorange=False,
        tickmode="array",
        tickvals=[-1, -0.5, 0, 0.5, 1],
        ticktext=["10", "15", "20", "25", "30"],  # new labels
        zeroline=False,
        title_font=dict(size=font_size, color="#000000"),
        tickfont=dict(size=font_size-2, color="#000000")
    )
    fig.update_yaxes(
        title_text="Throughput (piece)",
        range=[0, 15000],
        autorange=False,
        tickmode="array",
        tickvals=[0, 5000, 10000, 15000],
        tickformat=".0f",
        showexponent="none",
        zeroline=False,
        title_font=dict(size=font_size, color="#000000"),
        tickfont=dict(size=font_size-2, color="#000000"),
    )
    return fig

# ----------------------- App --------------------------------
def main():
    st.set_page_config(page_title="Single Throughput Plot", layout="wide")
    st.title("LOC of throughput")
    # Sidebar control (instead of central UI)
    st.sidebar.header("Display")

    # Select which dataset file to use for the intervals (under Display)
    avail = _available_throughput_files()
    if avail:
        labels = list(avail.keys())
        default_idx = labels.index("asymptotic normal") if "asymptotic normal" in labels else 0
        chosen_label = st.sidebar.selectbox("Prediction interval", labels, index=default_idx)
        path = avail[chosen_label]
    else:
        path = _find_data_file()
    df = load_data(path)
    try:
        observed_df = load_observed(OBS_DATA_FILE)
    except FileNotFoundError:
        observed_df = pd.DataFrame()
        st.warning("Observation file 'data.xlsx' not found – no points plotted.")

    zones_in_data = sorted(df["zoning"].dropna().unique().tolist())
    preferred_order = [z for z in ["BU","TD","RA","SQ"] if z in zones_in_data]
    zones_available = preferred_order or zones_in_data
    zone = zones_available[0] if zones_available else None  # automatic selection of first zone

    # Offer sources present in either predictions or observations
    sources_all = sorted(set(
        df.get("source", pd.Series(dtype=str)).dropna().unique().tolist()
    ) | set(
        observed_df.get("source", pd.Series(dtype=str)).dropna().unique().tolist()
    ))
    default_sources = [s for s in ["FIX","NO","EXP"] if s in sources_all] or sources_all
    sources = st.sidebar.multiselect(
        "Arrival pattern", options=sources_all, default=default_sources,
        format_func=lambda s: SOURCE_MAP.get(s, s),
        key="sources_select"
    )
    show_observed = st.sidebar.checkbox("Show observed throughput", value=True)  # moved up

    st.sidebar.markdown("---")
    st.sidebar.caption("Colors")  # colors now before sliders
    col_fix = st.sidebar.color_picker("Fixed", "#D55E00")
    col_no = st.sidebar.color_picker("Normal", "#0072B2")
    col_exp = st.sidebar.color_picker("Exponential", "#009E73")
    colors = {"FIX": col_fix, "NO": col_no, "EXP": col_exp}

    # Sliders moved below colors
    line_width = st.sidebar.slider("Line width", 1, 6, 3, 1)
    font_size = st.sidebar.slider("Base font size", 10, 40, 20, 1)
    plot_size = st.sidebar.slider("Plot size (px)", 400, 900, 700, 10)
    ribbon_alpha = st.sidebar.slider("Ribbon transparency", 0.05, 0.9, 0.18, 0.01)

    # Plot in main area
    if zone and sources:
        fig = build_single_plot(
            df, zone, sources, colors, line_width, ribbon_alpha=ribbon_alpha,
            observed=observed_df, show_observed=show_observed, font_size=font_size,
        )
        fig.update_layout(width=plot_size, height=plot_size)
        st.plotly_chart(fig, use_container_width=False)
        # Info if no intervals are present in the loaded dataset
        if not ({"low_delta","up_delta"}.issubset(df.columns) or {"low_corr","up_corr"}.issubset(df.columns)):
            st.warning("No interval bands found in the selected dataset – only the central line is plotted.")
    else:
        st.info("Select at least one arrival pattern.")

if __name__ == "__main__":
    main()
