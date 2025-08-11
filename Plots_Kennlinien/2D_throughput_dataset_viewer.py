# ------------------------------------------------------------
# throughput_2d_viewer.py
#
# 2-D-Kennlinien-Viewer für data.throughput.2d*.xlsx
# ------------------------------------------------------------
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------
# Pfade
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

def _find_data_file() -> Path:
    cand = sorted(BASE_DIR.glob("data.throughput.2d.xlsx"))
    if not cand:
        raise FileNotFoundError("Keine Datei data.throughput.2d*.xlsx im Skriptordner gefunden.")
    return cand[0]

DATA_FILE = _find_data_file()

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    # Aufräumen/umbenennen, falls sich Spaltennamen ändern
    df = df.rename(columns={
        "low.bootstrap": "lower",
        "up.bootstrap":  "upper",
        "traycontrol":   "zoning"
    })
    # Zusatz: reale Systemlast (54–62) aus kodiertem Wert zurückrechnen
    MID, HALF_RANGE = 58, 4
    df["systemload_raw"] = df["systemload"] * HALF_RANGE + MID
    return df

def build_facets(df: pd.DataFrame,
                 zones: list[str],
                 sources: list[str],
                 show_ribbon: bool,
                 use_raw_x: bool,
                 y_lock: bool,
                 colors: dict[str, str],
                 ribbon_alpha: float) -> go.Figure:

    x_col   = "systemload_raw" if use_raw_x else "systemload"
    x_title = "System load (raw)" if use_raw_x else "System load"

    df = df[df["zoning"].isin(zones) & df["source"].isin(sources)].copy()

    # Gemeinsamer y-Bereich über alle Facets (optional)
    y_range = None
    if y_lock and not df.empty:
        y_range = [float(df["lower"].min()), float(df["upper"].max())]

    # 2x2-Grid
    order = ["BU", "TD", "RA", "SQ"]
    zones  = [z for z in order if z in zones]
    fig = make_subplots(rows=2, cols=2, subplot_titles=zones)
    cell = [(1,1),(1,2),(2,1),(2,2)]

    for idx, z in enumerate(zones):
        r, c = cell[idx]
        sub = df[df["zoning"] == z]

        for src in sources:
            ssub = sub[sub["source"] == src].sort_values(x_col)
            if ssub.empty:
                continue

            # Ribbon (Bootstrap-Intervall)
            if show_ribbon:
                fig.add_trace(
                    go.Scatter(
                        x=ssub[x_col], y=ssub["lower"],
                        mode="lines", line=dict(width=0),
                        hoverinfo="skip", showlegend=False, legendgroup=src
                    ), row=r, col=c
                )
                fig.add_trace(
                    go.Scatter(
                        x=ssub[x_col], y=ssub["upper"],
                        mode="lines", line=dict(width=0),
                        fill="tonexty",
                        fillcolor=hex_to_rgba(colors[src], ribbon_alpha),
                        hoverinfo="skip", showlegend=False, legendgroup=src
                    ), row=r, col=c
                )

            # Mittellinie
            fig.add_trace(
                go.Scatter(
                    x=ssub[x_col], y=ssub["prediction"],
                    mode="lines",
                    line=dict(color=colors[src], width=2),
                    name=src, legendgroup=src,
                    showlegend=(idx == 0)  # Legende nur einmal oben einblenden
                ), row=r, col=c
            )

        fig.update_xaxes(title_text=x_title, row=r, col=c)
        fig.update_yaxes(title_text="Throughput", row=r, col=c, range=y_range)

    fig.update_layout(
        height=720, width=1200,
        title="Throughput vs. System load (facettiert nach Zoning)",
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.05)
    )
    return fig

# ------------------------------------------------------------
# App
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="2-D-Kennlinien (Running Example)", layout="wide")
    st.title("2-D-Kennlinien mit Bootstrap-Intervall")

    df = load_data(DATA_FILE)

    # Sidebar – Optionen
    st.sidebar.header("Anzeige")
    zones   = st.sidebar.multiselect("Zoning-Strategien", ["BU","TD","RA","SQ"], default=["BU","TD","RA","SQ"])
    sources = st.sidebar.multiselect("Quellen (source)", sorted(df["source"].unique()), default=sorted(df["source"].unique()))
    show_ribbon = st.sidebar.checkbox("Konfidenzband anzeigen (Bootstrap)", True)
    use_raw_x   = st.sidebar.checkbox("X-Achse: reale Systemlast (54–62) statt kodiert (−1…+1)", False)
    y_lock      = st.sidebar.checkbox("Einheitlicher y-Bereich für alle Panels", True)

    st.sidebar.markdown("---")
    st.sidebar.caption("Farben")
    col_ta = st.sidebar.color_picker("TA", "#D55E00")
    col_no = st.sidebar.color_picker("NO", "#0072B2")
    col_ex = st.sidebar.color_picker("EX", "#009E73")
    ribbon_alpha = st.sidebar.slider("Ribbon-Transparenz", 0.05, 0.9, 0.18, 0.01)

    color_map = {"TA": col_ta, "NO": col_no, "EX": col_ex}

    # Plot
    if zones and sources:
        fig = build_facets(
            df, zones, sources,
            show_ribbon=show_ribbon,
            use_raw_x=use_raw_x,
            y_lock=y_lock,
            colors=color_map,
            ribbon_alpha=ribbon_alpha,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Bitte mindestens eine Zoning-Strategie und eine Quelle auswählen.")

if __name__ == "__main__":
    main()
