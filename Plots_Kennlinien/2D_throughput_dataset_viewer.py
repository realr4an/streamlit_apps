# ------------------------------------------------------------
# throughput_2d_delta_viewer.py
# 2-D-Kennlinien-Viewer (nur Delta-Intervalle)
# kompatibel mit data.throughput.2d*.xlsx
# ------------------------------------------------------------
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).resolve().parent

def _find_data_file() -> Path:
    cands = sorted(BASE_DIR.glob("data.throughput.2d*.xlsx"))
    if not cands:
        raise FileNotFoundError("Keine Datei data.throughput.2d*.xlsx im Skriptordner gefunden.")
    return cands[0]

def _rgba(hex_str: str, a: float) -> str:
    h = hex_str.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)

    # Spalten harmonisieren (unterstützt alte & neue Schemata)
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
        # Intervalle (nur Delta)
        "low.delta": "low_delta",
        "up.delta":  "up_delta",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Fallback, falls 'source' fehlt -> eine Gruppe "ALL"
    if "source" not in df.columns:
        df["source"] = "ALL"

    # numerische Typen absichern
    for col in ["systemload","prediction","low_delta","up_delta"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # reale Systemlast (54–62) zusätzlich
    MID, HALF_RANGE = 58, 4
    if "systemload" in df.columns:
        df["systemload_raw"] = df["systemload"] * HALF_RANGE + MID

    return df

def build_facets(df: pd.DataFrame,
                 zones: list[str],
                 sources: list[str],
                 use_raw_x: bool,
                 y_lock: bool,
                 colors: dict[str, str],
                 ribbon_alpha: float) -> go.Figure:

    x_col   = "systemload_raw" if use_raw_x else "systemload"
    x_title = "System load (raw)" if use_raw_x else "System load"

    sub = df[df["zoning"].isin(zones) & df["source"].isin(sources)].copy()

    # Gemeinsamer y-Bereich über alle Panels (nur Delta)
    y_range = None
    if y_lock and {"low_delta","up_delta"}.issubset(sub.columns) and not sub.empty:
        y_range = [float(sub["low_delta"].min()), float(sub["up_delta"].max())]

    order = ["BU","TD","RA","SQ"]
    zones = [z for z in order if z in zones]
    fig = make_subplots(rows=2, cols=2, subplot_titles=zones)
    cells = [(1,1),(1,2),(2,1),(2,2)]

    for idx, z in enumerate(zones):
        r, c = cells[idx]
        d_z = sub[sub["zoning"] == z]

        for src in sources:
            d = d_z[d_z["source"] == src].sort_values(x_col)
            if d.empty:
                continue

            # Delta-Ribbon (falls vorhanden)
            if {"low_delta","up_delta"}.issubset(d.columns):
                fig.add_trace(
                    go.Scatter(x=d[x_col], y=d["low_delta"],
                               mode="lines", line=dict(width=0),
                               hoverinfo="skip", showlegend=False,
                               legendgroup=src),
                    row=r, col=c
                )
                fig.add_trace(
                    go.Scatter(x=d[x_col], y=d["up_delta"],
                               mode="lines", line=dict(width=0),
                               fill="tonexty",
                               fillcolor=_rgba(colors.get(src, "#888888"), ribbon_alpha),
                               hoverinfo="skip", showlegend=False,
                               legendgroup=src),
                    row=r, col=c
                )

            # Mittellinie
            fig.add_trace(
                go.Scatter(x=d[x_col], y=d["prediction"],
                           mode="lines",
                           line=dict(color=colors.get(src, "#444444"), width=2),
                           name=src, legendgroup=src,
                           showlegend=(idx == 0)),
                row=r, col=c
            )

        fig.update_xaxes(title_text=x_title, row=r, col=c)
        fig.update_yaxes(title_text="Throughput", row=r, col=c, range=y_range)

    fig.update_layout(
        height=720, width=1200,
        title="Throughput vs. System load — Delta-Prognoseintervalle",
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
    zones   = st.sidebar.multiselect("Zoning", ["BU","TD","RA","SQ"], default=["BU","TD","RA","SQ"])
    sources = st.sidebar.multiselect("source", sorted(df["source"].unique()),
                                     default=sorted(df["source"].unique()))
    use_raw_x = st.sidebar.checkbox("X-Achse: reale Systemlast (54–62)", True)
    y_lock    = st.sidebar.checkbox("Einheitlicher y-Bereich über Panels", True)

    st.sidebar.markdown("---")
    st.sidebar.caption("Farben")
    col_ta = st.sidebar.color_picker("TA", "#D55E00")
    col_no = st.sidebar.color_picker("NO", "#0072B2")
    col_ex = st.sidebar.color_picker("EX", "#009E73")
    ribbon_alpha = st.sidebar.slider("Ribbon-Transparenz", 0.05, 0.9, 0.18, 0.01)

    # Farbzuordnung (Fallback für unbekannte Quellen)
    colors = {"TA": col_ta, "NO": col_no, "EX": col_ex}

    if zones and sources:
        fig = build_facets(df, zones, sources, use_raw_x, y_lock, colors, ribbon_alpha)
        st.plotly_chart(fig, use_container_width=True)
        if not {"low_delta","up_delta"}.issubset(df.columns):
            st.warning("Delta-Intervalle nicht im Datensatz gefunden – es wird nur die Mittellinie gezeichnet.")
    else:
        st.info("Bitte mindestens eine Zoning-Strategie und eine Quelle auswählen.")

if __name__ == "__main__":
    main()
