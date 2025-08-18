# ------------------------------------------------------------
# throughput_2d_delta_viewer.py
# 2-D-Kennlinien-Viewer (nur Delta-Intervalle)
# kompatibel mit data.mopt.2d*.xlsx
# ------------------------------------------------------------
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).resolve().parent

# --------- Anzeigenamen (Mappings) ----------
ZONE_MAP = {
    "BU": "Bottom-up",
    "TD": "Top-down",
    "RA": "Random",
    "SQ": "Shortest queue",
}

SOURCE_MAP = {
    "TA": "Tacted",
    "NO": "Normal",
    "EX": "Exponential",
}

SOURCE_ORDER = ["TA", "NO", "EX"]  # gewünschte Reihenfolge

def _find_data_file() -> Path:
    # neue Datei: data.mopt.2d_10_30.xlsx (ggf. weitere Varianten)
    cands = sorted(BASE_DIR.glob("data.mopt.2d_10_30 1.xlsx"))
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
    warnings.filterwarnings(
        "ignore",
        message="Workbook contains no default style",
        category=UserWarning,
        module="openpyxl",
    )

    df = pd.read_excel(path)

    # Spalten harmonisieren (MOPT-Datensatz)
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
        "predicted_mopt": "prediction",          # MOPT
        # Intervalle (nur Delta)
        "low.delta": "low_delta",
        "up.delta":  "up_delta",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Unnötige Indexspalte ggf. droppen
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Fallback, falls 'source' fehlt -> eine Gruppe "ALL"
    if "source" not in df.columns:
        df["source"] = "ALL"

    # numerische Typen absichern
    for col in ["systemload", "prediction", "low_delta", "up_delta"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Kategorien normalisieren, damit Mapping sicher greift
    if "zoning" in df.columns:
        df["zoning"] = df["zoning"].astype(str).str.upper().str.strip()
    if "source" in df.columns:
        df["source"] = df["source"].astype(str).str.upper().str.strip()

    return df

def _order_sources(sources: list[str]) -> list[str]:
    in_data_ordered = [s for s in SOURCE_ORDER if s in sources]
    leftovers = [s for s in sources if s not in SOURCE_ORDER]
    return in_data_ordered + leftovers

def build_facets(df: pd.DataFrame,
                 zones: list[str],
                 sources: list[str],
                 y_lock: bool,
                 colors: dict[str, str],
                 ribbon_alpha: float) -> go.Figure:

    # Immer kodierte Achse –1…+1
    x_col   = "systemload"
    x_title = "Coded source parameter"

    sub = df[df["zoning"].isin(zones) & df["source"].isin(sources)].copy()
    has_delta = {"low_delta", "up_delta"}.issubset(sub.columns)

    # Gemeinsamer y-Bereich über alle Panels (nur Delta)
    y_range = None
    if y_lock and has_delta and not sub.empty:
        y_range = [float(sub["low_delta"].min()), float(sub["up_delta"].max())]

    # feste Reihenfolge der Panels
    order = ["BU", "TD", "RA", "SQ"]
    zones = [z for z in order if z in zones]

    # Subplot-Titel (ausgeschrieben) und auf 4 Felder auffüllen
    titles = [ZONE_MAP.get(z, z) for z in zones]
    while len(titles) < 4:
        titles.append("")

    fig = make_subplots(rows=2, cols=2, subplot_titles=titles)
    cells = [(1,1),(1,2),(2,1),(2,2)]

    # Reihenfolge der Serien im Plot/Legende
    sources_ordered = _order_sources(sources)

    for idx, z in enumerate(zones):
        r, c = cells[idx]
        d_z = sub[sub["zoning"] == z]

        for src in sources_ordered:
            d = d_z[d_z["source"] == src].sort_values(x_col)
            if d.empty:
                continue

            # Delta-Ribbon (falls vorhanden)
            if has_delta:
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

            # Mittellinie (Legende mit ausgeschriebenem Namen)
            fig.add_trace(
                go.Scatter(x=d[x_col], y=d["prediction"],
                           mode="lines",
                           line=dict(color=colors.get(src, "#444444"), width=2),
                           name=SOURCE_MAP.get(src, src),
                           legendgroup=src,
                           showlegend=(idx == 0)),
                row=r, col=c
            )

        fig.update_xaxes(
            title_text=x_title,
            range=[-1, 1],
            tickmode="array",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            zeroline=False,
            row=r, col=c
        )
        fig.update_yaxes(title_text="Mean order processing time", row=r, col=c, range=y_range)

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

    # Multiselect: Zonen in fixer Reihenfolge, Anzeige ausgeschrieben
    zone_options = ["BU","TD","RA","SQ"]
    zones = st.sidebar.multiselect(
        "Zoning",
        options=zone_options,
        default=zone_options,
        format_func=lambda z: ZONE_MAP.get(z, z),
    )

    # Sources in gewünschter Reihenfolge, Anzeige ausgeschrieben
    sources_in_data = df["source"].dropna().unique().tolist()
    source_options = _order_sources(sources_in_data)
    sources = st.sidebar.multiselect(
        "Source",
        options=source_options,
        default=source_options,
        format_func=lambda s: SOURCE_MAP.get(s, s),
    )

    y_lock = st.sidebar.checkbox("Einheitlicher y-Bereich über Panels", True)

    st.sidebar.markdown("---")
    st.sidebar.caption("Farben")
    # Labels ausgeschrieben, intern weiter TA/NO/EX
    col_ta = st.sidebar.color_picker("Tacted", "#D55E00")
    col_no = st.sidebar.color_picker("Normal", "#0072B2")
    col_ex = st.sidebar.color_picker("Exponential", "#009E73")
    ribbon_alpha = st.sidebar.slider("Ribbon-Transparenz", 0.05, 0.9, 0.18, 0.01)

    colors = {"TA": col_ta, "NO": col_no, "EX": col_ex}

    if zones and sources:
        fig = build_facets(df, zones, sources, y_lock, colors, ribbon_alpha)
        st.plotly_chart(fig, use_container_width=True)
        if not {"low_delta","up_delta"}.issubset(df.columns):
            st.warning("Delta-Intervalle nicht im Datensatz gefunden – es wird nur die Mittellinie gezeichnet.")
    else:
        st.info("Bitte mindestens eine Zoning-Strategie und eine Quelle wählen.")

if __name__ == "__main__":
    main()
