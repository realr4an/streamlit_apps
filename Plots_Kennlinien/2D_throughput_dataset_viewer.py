# ------------------------------------------------------------
# throughput_2d_delta_viewer.py
# 2-D-Kennlinien-Viewer (nur Delta-Intervalle) — kodierte X-Achse
# ------------------------------------------------------------
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).resolve().parent

ZONE_MAP = {"BU": "Bottom-up","TD": "Top-down","RA": "Random","SQ": "Shortest queue"}
SOURCE_MAP = {"FIX": "Fixed","NO": "Normal","EXP": "Exponential"}
SOURCE_ORDER = ["FIX","NO","EXP"]
# Datei mit beobachteten Rohwerten (mopt)
OBS_DATA_FILE = BASE_DIR / "data.xlsx"

def _find_data_file() -> Path:
    cands = sorted(BASE_DIR.glob("data.mopt_vollesdesign_r2.xlsx"))
    if not cands:
        raise FileNotFoundError("No data.mopt.2d*.xlsx file found in script directory.")
    return cands[0]

def _rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    import warnings
    warnings.filterwarnings("ignore", message="Workbook contains no default style",
                            category=UserWarning, module="openpyxl")
    df = pd.read_excel(path)
    rename_map = {
        "systemload": "systemload",
        "coded_sourceparameter": "systemload",
        # new dataset variants for coded mean arrival time
        "coded_mean_arrival_time": "systemload",
        "coded_mean_interarrival_time": "systemload",
        "coded_mean_interarrival": "systemload",
        "traycontrol": "zoning",
        "distributionstrategy": "zoning",
        "assignment_strategy": "zoning",
        "arrival_pattern": "Source",
        "prediction": "prediction",
        "predicted_throughput": "prediction",
        "predicted_mopt": "prediction",
        "low.delta": "low_delta",
        "up.delta":  "up_delta",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    if "Source" not in df.columns:
        df["Source"] = "ALL"
    for col in ["systemload","prediction","low_delta","up_delta"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "zoning" in df.columns:
        df["zoning"] = df["zoning"].astype(str).str.upper().str.strip()
    if "Source" in df.columns:
        df["Source"] = df["Source"].astype(str).str.upper().str.strip()
    return df

@st.cache_data
def load_observed(path: Path) -> pd.DataFrame:
    """Beobachtete Rohdaten (mopt) laden und harmonisieren.
    Fix: Mehrfach vorkommende Spaltennamen (z.B. distributionstrategy + distributinstrategy → zoning)
    werden zu einer Spalte zusammengeführt (pro Zeile erster nicht-NA Wert).
    """
    import warnings
    if not path.exists():
        return pd.DataFrame()
    warnings.filterwarnings("ignore", message="Workbook contains no default style",
                            category=UserWarning, module="openpyxl")
    df = pd.read_excel(path)
    rename_map = {
        "coded_sourceparameter": "systemload",
        "coded_mean_arrival_time": "systemload",
        "coded_mean_interarrival_time": "systemload",
        "coded_mean_interarrival": "systemload",
        "distributionstrategy": "zoning",
        "assignment_strategy": "zoning",
        "arrival_pattern": "Source",
        "distributinstrategy": "zoning",  # Tippfehler-Variante
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    def _collapse(frame: pd.DataFrame, col: str) -> pd.DataFrame:
        if list(frame.columns).count(col) > 1:
            cols = frame.loc[:, frame.columns == col]
            merged = cols.bfill(axis=1).iloc[:, 0]
            frame = frame.loc[:, frame.columns != col]
            frame[col] = merged
        return frame

    for c in ["zoning", "systemload", "Source"]:
        df = _collapse(df, c)

    # Fallback: falls nur 'source' existiert → in 'Source' überführen
    if "Source" not in df.columns and "source" in df.columns:
        df = _collapse(df, "source")
        df.rename(columns={"source": "Source"}, inplace=True)

    for c in ["systemload", "mopt"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "zoning" in df.columns:
        z = df["zoning"]
        if isinstance(z, pd.DataFrame):
            z = z.iloc[:, 0]
        df["zoning"] = z.astype(str).str.upper().str.strip()
    if "Source" in df.columns:
        s = df["Source"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        df["Source"] = s.astype(str).str.upper().str.strip()

    return df

def _order_sources(sources: list[str]) -> list[str]:
    in_data_ordered = [s for s in SOURCE_ORDER if s in sources]
    leftovers = [s for s in sources if s not in SOURCE_ORDER]
    return in_data_ordered + leftovers

def _nan_safe_min(series: pd.Series, default: float = 0.0) -> float:
    val = series.min(skipna=True)
    return float(val) if pd.notna(val) else default

def _nan_safe_max(series: pd.Series, default: float = 0.0) -> float:
    val = series.max(skipna=True)
    return float(val) if pd.notna(val) else default

def build_facets(df: pd.DataFrame,
                 zones: list[str],
                 sources: list[str],
                 y_lock: bool,
                 y_zero: bool,
                 colors: dict[str, str],
                 ribbon_alpha: float,
                 line_width: int,            # added
                 plot_size: int,             # added (for layout sizing)
                 observed: pd.DataFrame | None = None,
                 show_obs_points: bool = False,
                 font_size: int = 18) -> go.Figure:

    # Immer kodierte X-Achse –1…+1
    x_col   = "systemload"
    x_title = "Mean arrival time (sec)"

    sub = df[df["zoning"].isin(zones) & df["Source"].isin(sources)].copy()
    has_delta = {"low_delta", "up_delta"}.issubset(sub.columns)

    # Beobachtungen subsetten
    obs_sub = pd.DataFrame()
    if show_obs_points and observed is not None and not observed.empty:
        needed_cols = {"zoning", "Source", x_col, "mopt"}
        if needed_cols.issubset(set(observed.columns)):
            obs_sub = observed[(observed["zoning"].isin(zones)) & (observed["Source"].isin(sources))].copy()

    # Fester y-Bereich laut Anforderung: 100 bis 225, Ticks alle 25
    fixed_y_range = [100, 225]
    fixed_y_ticks = list(range(100, 226, 25))

    order = ["BU","TD","RA","SQ"]
    zones = [z for z in order if z in zones]

    titles = [ZONE_MAP.get(z, z) for z in zones]
    while len(titles) < 4:
        titles.append("")
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=titles,
        horizontal_spacing=0.22,
        vertical_spacing=0.18,
    )
    cells = [(1,1),(1,2),(2,1),(2,2)]

    sources_ordered = _order_sources(sources)

    # Placeholder für Observed Mopt wird nach allen Linien (am Ende) hinzugefügt, damit er in der Legende zuletzt steht

    # Helper: extend a line (and ribbons) to axis limits so the curve reaches the frame edges
    def _extend_to_limits(x: np.ndarray, y: np.ndarray, left: float, right: float) -> tuple[np.ndarray, np.ndarray]:
        if len(x) == 0:
            return x, y
        x_sorted_idx = np.argsort(x)
        x = x[x_sorted_idx]
        y = y[x_sorted_idx]
        # Left extrapolation
        if x[0] > left:
            if len(x) >= 2:
                # linear extrapolation using first two points
                slope = (y[1] - y[0]) / (x[1] - x[0]) if (x[1] - x[0]) != 0 else 0.0
                y_left = y[0] + slope * (left - x[0])
            else:
                y_left = y[0]
            x = np.insert(x, 0, left)
            y = np.insert(y, 0, y_left)
        # Right extrapolation
        if x[-1] < right:
            if len(x) >= 2:
                slope = (y[-1] - y[-2]) / (x[-1] - x[-2]) if (x[-1] - x[-2]) != 0 else 0.0
                y_right = y[-1] + slope * (right - x[-1])
            else:
                y_right = y[-1]
            x = np.append(x, right)
            y = np.append(y, y_right)
        return x, y

    # Wider margins on left/right; keep ticks at -1..1 but extend frame so points aren't cramped
    x_left, x_right = -1.1, 1.1

    for idx, z in enumerate(zones):
        r, c = cells[idx]
        d_z = sub[sub["zoning"] == z]
        obs_z = obs_sub[obs_sub["zoning"] == z] if not obs_sub.empty else pd.DataFrame()

        for src in sources_ordered:
            d = d_z[d_z["Source"] == src].sort_values(x_col)
            if d.empty and (obs_z.empty or obs_z[obs_z["Source"] == src].empty):
                continue

            if not d.empty and has_delta:
                # Extend ribbons to the plot edges
                x_low, y_low = _extend_to_limits(d[x_col].to_numpy(dtype=float), d["low_delta"].to_numpy(dtype=float), x_left, x_right)
                x_up,  y_up  = _extend_to_limits(d[x_col].to_numpy(dtype=float), d["up_delta"].to_numpy(dtype=float),  x_left, x_right)
                fig.add_trace(
                    go.Scatter(x=x_low, y=y_low, mode="lines",
                               line=dict(width=0), hoverinfo="skip",
                               showlegend=False, legendgroup=src),
                    row=r, col=c
                )
                fig.add_trace(
                    go.Scatter(x=x_up, y=y_up, mode="lines",
                               line=dict(width=0), fill="tonexty",
                               fillcolor=_rgba(colors.get(src, "#888888"), ribbon_alpha),
                               hoverinfo="skip", showlegend=False, legendgroup=src),
                    row=r, col=c
                )

            if not d.empty:
                # Extend central curve to the plot edges
                x_pred, y_pred = _extend_to_limits(d[x_col].to_numpy(dtype=float), d["prediction"].to_numpy(dtype=float), x_left, x_right)
                fig.add_trace(
                    go.Scatter(x=x_pred, y=y_pred, mode="lines",
                               line=dict(color=colors.get(src, "#444444"), width=line_width),  # use dynamic width
                               name=SOURCE_MAP.get(src, src),
                               legendgroup=src, showlegend=(idx == 0), legendrank=10 + sources_ordered.index(src)),
                    row=r, col=c
                )

            # Beobachtete mopt-Punkte (farbig, ohne Legendeneintrag)
            if show_obs_points and not obs_z.empty:
                o = obs_z[obs_z["Source"] == src].sort_values(x_col)
                if not o.empty and not o["mopt"].isna().all():
                    fig.add_trace(
                        go.Scatter(
                            x=o[x_col], y=o["mopt"], mode="markers",
                            marker=dict(symbol="circle", size=6, color=colors.get(src, "#666"),
                                        line=dict(width=0.5, color="#222")),
                            name="Observed mopt (colored)",
                            legendgroup="obs_mopt",
                            showlegend=False,
                            hovertemplate="Observed mopt<br>Mean arrival time: %{x}<br>mopt: %{y}<extra></extra>",  # translated
                        ),
                        row=r, col=c
                    )

        # X-Achse fix –1…+1, Labels 10,15,20,25,30; extend frame to add space left/right
        fig.update_xaxes(
            title_text=x_title,
            range=[-1.05, 1.05],
            autorange=False,
            tickmode="array",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["10", "15", "20", "25", "30"],
            zeroline=False,
            row=r,
            col=c,
            title_font=dict(size=font_size, color="#000000"),
            tickfont=dict(size=font_size-2, color="#000000"),
        )

        # Y-Achse: fest 100–225 mit 25er Schritten
        fig.update_yaxes(
            title_text="Mean order processing time (sec)",
            range=fixed_y_range, autorange=False,
            tickmode="array", tickvals=fixed_y_ticks,
            row=r, col=c,
            title_font=dict(size=font_size, color="#000000"),
            tickfont=dict(size=font_size-2, color="#000000")
        )

    # Observed Mopt Legendeneintrag zuletzt einfügen (hoher legendrank)
    if show_obs_points and not obs_sub.empty:
        fig.add_trace(
            go.Scatter(
                x=[0], y=[fixed_y_range[0]], mode="markers",
                marker=dict(symbol="circle", size=7, color="#FFFFFF", line=dict(width=0.7, color="#000000")),
                name="Observation",
                legendgroup="obs_mopt",
                showlegend=True,
                hoverinfo="skip",
                visible="legendonly",
                legendrank=100
            ),
            row=1, col=1
        )

    fig.update_layout(
        height=plot_size, width=plot_size,  # dynamic size
        # Titel entfernt auf Wunsch
        font=dict(size=font_size, color="#000000"),
        # Slightly tighter outer margins; internal spacing increased above
        margin=dict(l=6, r=6, t=60, b=6),
        legend=dict(
            orientation="v",
            title=dict(text="Arrival pattern", font=dict(size=font_size-2, color="#000000")),
            x=1.02, xanchor="left", y=1, yanchor="top",
            font=dict(size=font_size-2, color="#000000"),
            bgcolor="rgba(255,255,255,0.0)",
            borderwidth=0
        )
    )
    # Subplot-Titel vergrößern
    if hasattr(fig.layout, 'annotations'):
        for ann in fig.layout.annotations:
            if ann.text in titles:
                ann.font = dict(size=font_size, color="#000000")
    return fig

def main():
    st.set_page_config(page_title="2-D curves (Delta)", layout="wide")  # translated
    st.title("LOC of mean order processing time")

    path = _find_data_file()
    df = load_data(path)
    observed_df = load_observed(OBS_DATA_FILE)

    st.sidebar.header("Display")
    zone_options = ["BU","TD","RA","SQ"]
    zones = st.sidebar.multiselect(
        "Assignment strategy", options=zone_options, default=zone_options,  # renamed
        format_func=lambda z: ZONE_MAP.get(z, z),
    )

    sources_in_data = df["Source"].dropna().unique().tolist()
    source_options = _order_sources(sources_in_data)
    sources = st.sidebar.multiselect(
        "Arrival pattern",
        options=source_options, default=source_options,
        format_func=lambda s: SOURCE_MAP.get(s, s),
    )

    y_lock = st.sidebar.checkbox("Uniform y-range across panels", True)
    y_zero = st.sidebar.checkbox("Force y-axis start at 0", False)
    show_obs_points = st.sidebar.checkbox("Show observed mean order processing time", True)  # moved up

    st.sidebar.markdown("---")
    st.sidebar.caption("Colors")
    col_fix = st.sidebar.color_picker("Fixed", "#D55E00")
    col_no = st.sidebar.color_picker("Normal", "#0072B2")
    col_exp = st.sidebar.color_picker("Exponential", "#009E73")
    colors = {"FIX": col_fix, "NO": col_no, "EXP": col_exp}
    line_width = st.sidebar.slider("Line width", 1, 6, 2, 1)              # moved up
    font_size = st.sidebar.slider("Base font size", 10, 40, 20, 1)        # moved up
    plot_size = st.sidebar.slider("Plot size (px)", 600, 1400, 1000, 50)  # moved up
    ribbon_alpha = st.sidebar.slider("Ribbon transparency", 0.05, 0.9, 0.18, 0.01)  # moved down

    if zones and sources:
        fig = build_facets(
            df, zones, sources, y_lock, y_zero, colors, ribbon_alpha,
            line_width, plot_size,
            observed=observed_df, show_obs_points=show_obs_points, font_size=font_size
        )
        st.plotly_chart(
            fig, use_container_width=False,
            key=f"facets-{y_lock}-{y_zero}-{line_width}-{plot_size}"
        )
        if not {"low_delta","up_delta"}.issubset(df.columns):
            st.warning("Delta intervals not found in dataset – only the central line is drawn.")
    else:
        st.info("Please select at least one assignment strategy and one arrival pattern.")

if __name__ == "__main__":
    main()
