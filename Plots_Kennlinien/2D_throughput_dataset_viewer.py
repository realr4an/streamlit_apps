# ------------------------------------------------------------
# throughput_2d_delta_viewer.py
# 2-D-Kennlinien-Viewer (nur Delta-Intervalle) — kodierte X-Achse
# ------------------------------------------------------------
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).resolve().parent

ZONE_MAP = {"BU": "Bottom-up","TD": "Top-down","RA": "Random","SQ": "Shortest queue"}
SOURCE_MAP = {"TA": "Tacted","NO": "Normal","EX": "Exponential"}
SOURCE_ORDER = ["TA","NO","EX"]

def _find_data_file() -> Path:
    cands = sorted(BASE_DIR.glob("data.mopt.2d_10_30 2.xlsx"))
    if not cands:
        raise FileNotFoundError("Keine Datei data.mopt.2d*.xlsx im Skriptordner gefunden.")
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
        "traycontrol": "zoning",
        "distributionstrategy": "zoning",
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

def _order_sources(sources: list[str]) -> list[str]:
    in_data_ordered = [s for s in SOURCE_ORDER if s in sources]
    leftovers = [s for s in sources if s not in SOURCE_ORDER]
    return in_data_ordered + leftovers

def build_facets(df: pd.DataFrame,
                 zones: list[str],
                 sources: list[str],
                 y_lock: bool,
                 y_zero: bool,
                 colors: dict[str, str],
                 ribbon_alpha: float) -> go.Figure:

    # Immer kodierte X-Achse –1…+1
    x_col   = "systemload"
    x_title = "Coded Source parameter"

    sub = df[df["zoning"].isin(zones) & df["Source"].isin(sources)].copy()
    has_delta = {"low_delta", "up_delta"}.issubset(sub.columns)

    # Globaler y-Bereich?
    y_range_global = None
    if y_lock and not sub.empty:
        if has_delta:
            ymax = float(sub["up_delta"].max())
        else:
            ymax = float(sub["prediction"].max())
        ymin = 0.0 if y_zero else (float(sub["low_delta"].min()) if has_delta else float(sub["prediction"].min()))
        if y_zero:
            ymin = 0.0
        y_range_global = [ymin, ymax]

    order = ["BU","TD","RA","SQ"]
    zones = [z for z in order if z in zones]

    titles = [ZONE_MAP.get(z, z) for z in zones]
    while len(titles) < 4:
        titles.append("")
    fig = make_subplots(rows=2, cols=2, subplot_titles=titles)
    cells = [(1,1),(1,2),(2,1),(2,2)]

    sources_ordered = _order_sources(sources)

    for idx, z in enumerate(zones):
        r, c = cells[idx]
        d_z = sub[sub["zoning"] == z]

        for src in sources_ordered:
            d = d_z[d_z["Source"] == src].sort_values(x_col)
            if d.empty:
                continue

            if has_delta:
                fig.add_trace(
                    go.Scatter(x=d[x_col], y=d["low_delta"], mode="lines",
                               line=dict(width=0), hoverinfo="skip",
                               showlegend=False, legendgroup=src),
                    row=r, col=c
                )
                fig.add_trace(
                    go.Scatter(x=d[x_col], y=d["up_delta"], mode="lines",
                               line=dict(width=0), fill="tonexty",
                               fillcolor=_rgba(colors.get(src, "#888888"), ribbon_alpha),
                               hoverinfo="skip", showlegend=False, legendgroup=src),
                    row=r, col=c
                )

            fig.add_trace(
                go.Scatter(x=d[x_col], y=d["prediction"], mode="lines",
                           line=dict(color=colors.get(src, "#444444"), width=2),
                           name=SOURCE_MAP.get(src, src),
                           legendgroup=src, showlegend=(idx == 0)),
                row=r, col=c
            )

        # X-Achse fix –1…+1
        fig.update_xaxes(
            title_text=x_title, range=[-1, 1],
            tickmode="array", tickvals=[-1, -0.5, 0, 0.5, 1],
            zeroline=False, row=r, col=c
        )

        # Y-Achse: globaler Bereich oder panel-spezifisch
        if y_range_global is not None:
            fig.update_yaxes(title_text="Mean order processing time",
                             range=y_range_global, row=r, col=c)
        else:
            if y_zero:
                # Panel-spezifisch ab 0, obere Grenze automatisch
                fig.update_yaxes(title_text="Mean order processing time",
                                 rangemode="tozero", row=r, col=c)
            else:
                fig.update_yaxes(title_text="Mean order processing time", row=r, col=c)

    fig.update_layout(
        height=720, width=1200,
        title="Mean order processing time — Delta-Prognoseintervalle",
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.05)
    )
    return fig

def main():
    st.set_page_config(page_title="2-D-Kennlinien (Delta)", layout="wide")
    st.title("2-D-Kennlinien mit Delta-Prognoseintervallen")

    path = _find_data_file()
    df = load_data(path)

    st.sidebar.header("Anzeige")
    zone_options = ["BU","TD","RA","SQ"]
    zones = st.sidebar.multiselect(
        "Zoning", options=zone_options, default=zone_options,
        format_func=lambda z: ZONE_MAP.get(z, z),
    )

    sources_in_data = df["Source"].dropna().unique().tolist()
    source_options = _order_sources(sources_in_data)
    sources = st.sidebar.multiselect(
        "Source", options=source_options, default=source_options,
        format_func=lambda s: SOURCE_MAP.get(s, s),
    )

    y_lock = st.sidebar.checkbox("Einheitlicher y-Bereich über Panels", True)
    y_zero = st.sidebar.checkbox("Y-Achse bei 0 beginnen lassen", False)

    st.sidebar.markdown("---")
    st.sidebar.caption("Farben")
    col_ta = st.sidebar.color_picker("Tacted", "#D55E00")
    col_no = st.sidebar.color_picker("Normal", "#0072B2")
    col_ex = st.sidebar.color_picker("Exponential", "#009E73")
    ribbon_alpha = st.sidebar.slider("Ribbon-Transparenz", 0.05, 0.9, 0.18, 0.01)
    colors = {"TA": col_ta, "NO": col_no, "EX": col_ex}

    if zones and sources:
        fig = build_facets(df, zones, sources, y_lock, y_zero, colors, ribbon_alpha)
        st.plotly_chart(fig, use_container_width=True)
        if not {"low_delta","up_delta"}.issubset(df.columns):
            st.warning("Delta-Intervalle nicht im Datensatz gefunden – es wird nur die Mittellinie gezeichnet.")
    else:
        st.info("Bitte mindestens eine Zoning-Strategie und eine Quelle wählen.")

if __name__ == "__main__":
    main()
