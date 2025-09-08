# ------------------------------------------------------------
# throughput_single_plot_app.py
# 2D-Plot (encoded system load –1…+1) with delta intervals (if available)
# tuned for: data.throughput.2d_10_30 4.xlsx
# ------------------------------------------------------------
from pathlib import Path
import warnings
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Base directory of this script (needed for file lookups)
BASE_DIR = Path(__file__).resolve().parent

# Path to the observation file
OBS_DATA_FILE = BASE_DIR / "data.xlsx"

# ----------------------- Mappings ---------------------------
ZONE_MAP = {
    "BU": "Bottom-Up",
    "TD": "Top-Down",
    "RA": "Random",
    "SQ": "Shortest Queue",
}
SOURCE_MAP = {"TA": "Tacted", "NO": "Normal", "EX": "Exponential"}

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
        "data.throughput.2d_10_30 6.xlsx"
    ]
    for pat in patterns:
        cands = sorted(BASE_DIR.glob(pat))
        if cands:
            return cands[0]
    raise FileNotFoundError("No matching Excel file found.")

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
        # Facet
        "traycontrol": "zoning",
        "distributionstrategy": "zoning",
        # y
        "prediction": "prediction",
        "predicted_throughput": "prediction",
        "predicted_mopt": "prediction",
        # Intervals (Delta)
        "low.delta": "low_delta",
        "up.delta":  "up_delta",
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
    for c in ["systemload", "prediction", "low_delta", "up_delta"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Normalize categories (Upper/Trim + Mapping BA→BU etc.)
    if "zoning" in df.columns:
        z = df["zoning"].astype(str).str.upper().str.strip()
        df["zoning"] = z.replace(ZONING_NORMALIZE)
    if "source" in df.columns:
        df["source"] = df["source"].astype(str).str.upper().str.strip()

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
        "distributionstrategy": "zoning",
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
            df[c] = pd.to_numeric(df[c], errors="coerce")

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
        df["source"] = source_col.astype(str).str.upper().str.strip()
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
    xcol = "systemload"  # kodiert –1…+1
    xtitle = "Mean arrival time (sec)"

    d = df[df["zoning"] == zone]
    if d.empty:
        st.warning(f"No data for zoning '{zone}'. Available: {sorted(df['zoning'].unique().tolist())}")

    order = [s for s in ["TA", "NO", "EX"] if s in sources] + [s for s in sources if s not in ["TA","NO","EX"]]
    has_delta = {"low_delta", "up_delta"}.issubset(df.columns)

    obs = pd.DataFrame()
    if show_observed and observed is not None and not observed.empty:
        obs = observed[(observed["zoning"] == zone) & (observed["source"].isin(sources))].copy()

    fig = go.Figure()
    # --- Forecast lines + intervals ---
    for src in order:
        s = d[d["source"] == src].sort_values(xcol)
        if s.empty:
            continue

        if has_delta and not s[["low_delta","up_delta"]].isna().all().all():
            fig.add_trace(go.Scatter(
                x=s[xcol], y=s["low_delta"], mode="lines",
                line=dict(width=0), hoverinfo="skip",
                showlegend=False, legendgroup=src
            ))
            fig.add_trace(go.Scatter(
                x=s[xcol], y=s["up_delta"], mode="lines",
                line=dict(width=0), fill="tonexty",
                fillcolor=_rgba(colors.get(src, "#888888"), ribbon_alpha),
                hoverinfo="skip", showlegend=False, legendgroup=src
            ))

        fig.add_trace(go.Scatter(
            x=s[xcol], y=s["prediction"], mode="lines",
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
                    x=src_pts["systemload"],
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
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(
            title=dict(text="Arrival pattern", font=dict(size=font_size-2, color="#000000")),
            font=dict(size=font_size-2, color="#000000")
        ),
        font=dict(size=font_size, color="#000000")
    )
    fig.update_xaxes(
        title_text=xtitle, range=[-1, 1],
        tickmode="array", tickvals=[-1, -0.5, 0, 0.5, 1],
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
        tickfont=dict(size=font_size-2, color="#000000")
    )
    return fig

# ----------------------- App --------------------------------
def main():
    st.set_page_config(page_title="Single Throughput Plot", layout="wide")
    st.title("LOC of throughput")

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

    # Sidebar control (instead of central UI)
    st.sidebar.header("Display")

    sources_all = sorted(df["source"].dropna().unique().tolist())
    default_sources = [s for s in ["TA","NO","EX"] if s in sources_all] or sources_all
    sources = st.sidebar.multiselect(
        "Arrival distribution", options=sources_all, default=default_sources,
        format_func=lambda s: SOURCE_MAP.get(s, s),
        key="sources_select"
    )
    show_observed = st.sidebar.checkbox("Show observed throughput", value=True)  # moved up

    st.sidebar.markdown("---")
    st.sidebar.caption("Colors")  # colors now before sliders
    col_ta = st.sidebar.color_picker("Tacted", "#D55E00")
    col_no = st.sidebar.color_picker("Normal", "#0072B2")
    col_ex = st.sidebar.color_picker("Exponential", "#009E73")
    colors = {"TA": col_ta, "NO": col_no, "EX": col_ex}

    # Sliders moved below colors
    line_width = st.sidebar.slider("Line width", 1, 6, 3, 1)
    font_size = st.sidebar.slider("Base font size", 10, 40, 20, 1)
    plot_size = st.sidebar.slider("Plot size (px)", 400, 900, 700, 10)
    ribbon_alpha = st.sidebar.slider("Ribbon transparency", 0.05, 0.9, 0.18, 0.01)

    # Plot in main area
    if zone and sources:
        fig = build_single_plot(
            df, zone, sources, colors, line_width, ribbon_alpha=ribbon_alpha,
            observed=observed_df, show_observed=show_observed, font_size=font_size
        )
        fig.update_layout(width=plot_size, height=plot_size)
        st.plotly_chart(fig, use_container_width=False)
        if not {"low_delta","up_delta"}.issubset(df.columns):
            st.warning("Delta intervals not found in dataset – only the central line is plotted.")
    else:
        st.info("Select at least one arrival distribution.")

if __name__ == "__main__":
    main()