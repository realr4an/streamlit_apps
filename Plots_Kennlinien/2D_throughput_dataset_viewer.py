# ------------------------------------------------------------
# throughput_2d_delta_viewer.py
# 2-D-Kennlinien-Viewer — kodierte X-Achse, wählbare Prognoseintervalle
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
SOURCE_NORMALIZE = {
    "FIXED": "FIX",
    "FIX": "FIX",
    "TA": "FIX",
    "EX": "EXP",
    "EXP": "EXP",
    "EXPONENTIAL": "EXP",
    "NORMAL": "NO",
    "NORM": "NO",
    "NO": "NO",
}
# Datei mit beobachteten Rohwerten (mopt)
DESIGN_CONFIGS: dict[str, dict] = {
    "FFF 36, r=0": {
        "prediction_files": {
            "delta interval": BASE_DIR / "data.mopt_FFF36_mitRand.xlsx",
        },
        "observed_file": BASE_DIR / "data_mitRand.xlsx",
        "default_interval": "delta interval",
        "line_column": "mopt",
        "observed_value_column": "mopt",
    },
}


def _decode_mean_arrival(coded: np.ndarray | pd.Series | float) -> np.ndarray:
    """Convert coded mean arrival time (≈−1…+1) to real seconds (≈10…30)."""
    decoded = 20.0 + 10.0 * np.asarray(coded, dtype=float)
    return np.clip(decoded, 10.0, 30.0)


def _available_designs() -> dict[str, dict]:
    available: dict[str, dict] = {}
    for name, cfg in DESIGN_CONFIGS.items():
        preds = {
            label: path for label, path in cfg.get("prediction_files", {}).items()
            if path.exists()
        }
        if preds:
            available[name] = {**cfg, "prediction_files": preds}
    return available

def _rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"

# Robust numeric coercion that also handles decimal commas
def _coerce_numeric(series: pd.Series) -> pd.Series:
    s1 = pd.to_numeric(series, errors="coerce")
    if s1.notna().any() and s1.isna().sum() == 0:
        return s1
    s2 = pd.to_numeric(series.astype(str).str.replace(" ", "", regex=False)
                       .str.replace("\u00A0", "", regex=False)
                       .str.replace(".", "", regex=False)  # remove thousand separator if present
                       .str.replace(",", ".", regex=False), errors="coerce")
    return s1.where(s1.notna(), s2)

# Resolve the coded mean arrival time column present in the dataframe
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
    return candidates[0]

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
        "coded_meanarrivaltime": "systemload",
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
        # Intervals (asymptotic / finite-sample corrected)
        "low.delta": "low_delta",
        "up.delta":  "up_delta",
        "low.corr":  "low_corr",
        "up.corr":   "up_corr",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    if "Source" not in df.columns:
        df["Source"] = "ALL"
    for col in ["systemload","prediction","low_delta","up_delta","low_corr","up_corr"]:
        if col in df.columns:
            df[col] = _coerce_numeric(df[col])
    if "zoning" in df.columns:
        df["zoning"] = df["zoning"].astype(str).str.upper().str.strip()
    if "Source" in df.columns:
        s = df["Source"].astype(str).str.upper().str.strip()
        df["Source"] = s.replace(SOURCE_NORMALIZE)
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
        "coded_meanarrivaltime": "systemload",
        "coded_mean_arrival_time": "systemload",
        "coded_mean_interarrival_time": "systemload",
        "coded_mean_interarrival": "systemload",
        "distributionstrategy": "zoning",
        "assignment_strategy": "zoning",
        "arrival_pattern": "Source",
        "distributinstrategy": "zoning",  # Tippfehler-Variante
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

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
            df[c] = _coerce_numeric(df[c])

    if "zoning" in df.columns:
        z = df["zoning"]
        if isinstance(z, pd.DataFrame):
            z = z.iloc[:, 0]
        df["zoning"] = z.astype(str).str.upper().str.strip()
    if "Source" in df.columns:
        s = df["Source"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        df["Source"] = s.astype(str).str.upper().str.strip().replace(SOURCE_NORMALIZE)

    return df


def _build_pseudo_observations(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    pseudo_cfg = cfg.get("pseudo_observed")
    if not pseudo_cfg:
        return pd.DataFrame()

    y_from = pseudo_cfg.get("y_from")
    y_to = pseudo_cfg.get("y_to", y_from)
    source_field = pseudo_cfg.get("source_field", "Source")

    required_cols = {"systemload", "zoning", y_from}
    if source_field in df.columns:
        required_cols.add(source_field)
    if not required_cols.issubset(df.columns):
        return pd.DataFrame()

    obs = df.loc[:, list(required_cols)].copy()
    obs = obs.rename(columns={y_from: y_to})
    if source_field not in obs.columns:
        obs[source_field] = "ALL"

    obs[source_field] = obs[source_field].astype(str).str.upper().str.strip().replace(SOURCE_NORMALIZE)
    obs["zoning"] = obs["zoning"].astype(str).str.upper().str.strip()

    keep_cols = ["systemload", "zoning", source_field, y_to]
    obs = obs.loc[:, keep_cols]
    return obs.dropna(subset=["systemload", "zoning", y_to])


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
                 colors: dict[str, str],
                 ribbon_alpha: float,
                 line_width: int,            # added
                 plot_size: int,             # added (for layout sizing)
                 observed: pd.DataFrame | None = None,
                 show_obs_points: bool = False,
                 font_size: int = 18,
                 line_column: str = "prediction",
                 observed_value_column: str = "mopt") -> go.Figure:

    # Immer kodierte X-Achse –1…+1
    x_col   = _resolve_xcol(df)
    x_title = "Mean interarrival time (sec)"

    sub = df[df["zoning"].isin(zones) & df["Source"].isin(sources)].copy()
    # Choose interval columns automatically from what's available in the loaded dataset
    if {"low_corr", "up_corr"}.issubset(sub.columns):
        low_col, up_col = "low_corr", "up_corr"
    elif {"low_delta", "up_delta"}.issubset(sub.columns):
        low_col, up_col = "low_delta", "up_delta"
    else:
        low_col, up_col = None, None
    has_bands = low_col is not None and up_col is not None

    # Beobachtungen subsetten
    obs_sub = pd.DataFrame()
    if show_obs_points and observed is not None and not observed.empty:
        needed_cols = {"zoning", "Source", x_col, observed_value_column}
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

            if not d.empty and has_bands:
                # Extend ribbons to the plot edges
                x_low, y_low = _extend_to_limits(d[x_col].to_numpy(dtype=float), d[low_col].to_numpy(dtype=float), x_left, x_right)
                x_up,  y_up  = _extend_to_limits(d[x_col].to_numpy(dtype=float), d[up_col].to_numpy(dtype=float),  x_left, x_right)
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

            if line_column not in d.columns:
                line_col = "prediction" if "prediction" in d.columns else d.columns[-1]
            else:
                line_col = line_column

            if not d.empty:
                # Extend central curve to the plot edges
                x_pred, y_pred = _extend_to_limits(d[x_col].to_numpy(dtype=float), d[line_col].to_numpy(dtype=float), x_left, x_right)
                decoded_pred = _decode_mean_arrival(x_pred)
                source_label = SOURCE_MAP.get(src, src)
                fig.add_trace(
                    go.Scatter(x=x_pred, y=y_pred, mode="lines",
                               line=dict(color=colors.get(src, "#444444"), width=line_width),  # use dynamic width
                               name=source_label,
                               legendgroup=src, showlegend=(idx == 0), legendrank=10 + sources_ordered.index(src),
                               customdata=np.column_stack((decoded_pred,)),
                               hovertemplate=(
                                   f"Routing strategy: {ZONE_MAP.get(z, z)}<br>"
                                   f"Interarrival time pattern: {source_label}<br>"
                                   "Mean interarrival time: %{customdata[0]:.2f} sec<br>"
                                   "Mean order processing time: %{y:.2f} sec<extra></extra>"
                               )),
                    row=r, col=c
                )

            # Beobachtete mopt-Punkte (farbig, ohne Legendeneintrag)
            if show_obs_points and not obs_z.empty:
                o = obs_z[obs_z["Source"] == src].sort_values(x_col)
                if observed_value_column not in o.columns:
                    continue
                if not o.empty and not o[observed_value_column].isna().all():
                    decoded_obs = _decode_mean_arrival(o[x_col].to_numpy(dtype=float))
                    zone_labels = o["zoning"].map(lambda val: ZONE_MAP.get(val, val))
                    source_label = SOURCE_MAP.get(src, src)
                    source_labels = pd.Series([source_label] * len(o))
                    custom = np.column_stack((decoded_obs, zone_labels.to_numpy(dtype=object), source_labels.to_numpy(dtype=object)))
                    fig.add_trace(
                        go.Scatter(
                            x=o[x_col], y=o[observed_value_column], mode="markers",
                            marker=dict(symbol="circle", size=6, color=colors.get(src, "#666"),
                                        line=dict(width=0.5, color="#222")),
                            name="Observation",
                            legendgroup="obs_mopt",
                            showlegend=False,
                            customdata=custom,
                            hovertemplate=(
                                "Observation<br>"
                                "Routing strategy: %{customdata[1]}<br>"
                                "Interarrival time pattern: %{customdata[2]}<br>"
                                "Mean interarrival time: %{customdata[0]:.2f} sec<br>"
                                "Mean order processing time: %{y:.2f} sec<extra></extra>"
                            ),
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
            title=dict(text="Interarrival time pattern", font=dict(size=font_size-2, color="#000000")),
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
    st.title("LOCs with uncertainty bands of mean order processing time")

    st.sidebar.header("Display")

    available_designs = _available_designs()
    if not available_designs:
        st.error("No mopt datasets found next to this script.")
        st.stop()

    design_labels = list(available_designs.keys())
    if len(design_labels) == 1:
        design_choice = design_labels[0]
    else:
        default_design = design_labels.index("FFF 36, r=0") if "FFF 36, r=0" in design_labels else 0
        design_choice = st.sidebar.selectbox("Design", design_labels, index=default_design)
    design_cfg = available_designs[design_choice]

    predictions = design_cfg["prediction_files"]
    interval_labels = list(predictions.keys())
    default_interval = design_cfg.get("default_interval")
    if default_interval not in interval_labels:
        default_interval = interval_labels[0]
    if len(interval_labels) == 1:
        chosen_interval = interval_labels[0]
    else:
        interval_idx = interval_labels.index(default_interval)
        chosen_interval = st.sidebar.selectbox("Prediction interval", interval_labels, index=interval_idx)

    df = load_data(predictions[chosen_interval])

    observed_df = pd.DataFrame()
    observed_path = design_cfg.get("observed_file")
    if observed_path is not None:
        observed_df = load_observed(observed_path)

    if observed_df.empty:
        pseudo_obs = _build_pseudo_observations(df, design_cfg)
        if not pseudo_obs.empty:
            observed_df = pseudo_obs

    line_column = design_cfg.get("line_column", "prediction")
    obs_value_column = design_cfg.get("observed_value_column", "mopt")

    zone_options = ["BU","TD","RA","SQ"]
    zones = st.sidebar.multiselect(
        "Routing strategy", options=zone_options, default=zone_options,  # renamed
        format_func=lambda z: ZONE_MAP.get(z, z),
    )

    sources_in_data = df["Source"].dropna().unique().tolist()
    source_options = _order_sources(sources_in_data)
    sources = st.sidebar.multiselect(
        "Interarrival time pattern",
        options=source_options, default=source_options,
        format_func=lambda s: SOURCE_MAP.get(s, s),
    )

    y_lock = st.sidebar.checkbox("Uniform y-range across panels", True)
    show_obs_default = not observed_df.empty
    show_obs_points = st.sidebar.checkbox("Show observed mean order processing time", show_obs_default)
    if show_obs_points and observed_df.empty:
        st.sidebar.info("No observed mean order processing time available for this design.")

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
            df, zones, sources, y_lock, colors, ribbon_alpha,
            line_width, plot_size,
            observed=observed_df, show_obs_points=show_obs_points, font_size=font_size,
            line_column=line_column, observed_value_column=obs_value_column,
        )
        st.plotly_chart(
            fig, use_container_width=False,
            key=f"facets-{y_lock}-{line_width}-{plot_size}"
        )
        # Info if no intervals are present in the loaded dataset
        if not ({"low_delta","up_delta"}.issubset(df.columns) or {"low_corr","up_corr"}.issubset(df.columns)):
            st.warning("No interval bands found in the selected dataset – only the central line is drawn.")
    else:
        st.info("Please select at least one routing strategy and one interarrival time pattern.")

if __name__ == "__main__":
    main()
