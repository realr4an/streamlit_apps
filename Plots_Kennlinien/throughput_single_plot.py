# ------------------------------------------------------------
# throughput_single_plot_app.py
# 2D-Plot (kodierte Systemlast –1…+1) mit Delta-Intervallen (falls vorhanden)
# abgestimmt auf: data.throughput.2d_10_30 4.xlsx
# ------------------------------------------------------------
from pathlib import Path
import warnings
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent

# ----------------------- Mappings ---------------------------
ZONE_MAP = {
    "BU": "Bottom-Up",
    "TD": "Top-Down",
    "RA": "Random",
    "SQ": "Shortest Queue",
}
SOURCE_MAP = {"TA": "Tacted", "NO": "Normal", "EX": "Exponential"}

# Optionale Normalisierung für Tippfehler/Varianten
ZONING_NORMALIZE = {
    "BA": "BU", "BOTTOM-UP": "BU", "BOTTOM UP": "BU",
    "TOP-DOWN": "TD", "TOP DOWN": "TD",
    "RANDOM": "RA",
    "SHORTEST QUEUE": "SQ",
}

# ----------------------- Datei finden -----------------------
def _find_data_file() -> Path:
    patterns = [
        "data.throughput.2d_10_30 4.xlsx",  # dein File
        "data.throughput.2d*.xlsx",
        "data.mopt.2d*.xlsx",
    ]
    for pat in patterns:
        cands = sorted(BASE_DIR.glob(pat))
        if cands:
            return cands[0]
    raise FileNotFoundError("Keine passende Excel-Datei gefunden.")

# ----------------------- Laden & Aufbereiten ----------------
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    warnings.filterwarnings("ignore", message="Workbook contains no default style",
                            category=UserWarning, module="openpyxl")
    df = pd.read_excel(path)

    # Spalten harmonisieren auf interne Namen
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
        # Intervalle (Delta)
        "low.delta": "low_delta",
        "up.delta":  "up_delta",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Quelle: "Source" (groß) → "source"
    if "Source" in df.columns and "source" not in df.columns:
        df["source"] = df["Source"]
    if "source" not in df.columns:
        df["source"] = "ALL"

    # Unnötige Indexspalte entfernen
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Numerik absichern
    for c in ["systemload", "prediction", "low_delta", "up_delta"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Kategorien normalisieren (Upper/Trim + Mapping BA→BU etc.)
    if "zoning" in df.columns:
        z = df["zoning"].astype(str).str.upper().str.strip()
        df["zoning"] = z.replace(ZONING_NORMALIZE)
    if "source" in df.columns:
        df["source"] = df["source"].astype(str).str.upper().str.strip()

    return df

def _rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"

# ----------------------- Plot bauen -------------------------
def build_single_plot(
    df: pd.DataFrame,
    zone: str,
    sources,
    colors: dict,
    line_width: int,
    y_zero: bool,
    ribbon_alpha: float = 0.18
) -> go.Figure:
    xcol = "systemload"  # kodiert –1…+1
    xtitle = "Coded source parameter"

    d = df[df["zoning"] == zone]
    if d.empty:
        st.warning(f"Keine Daten für Zoning '{zone}'. Verfügbare Zonen: {sorted(df['zoning'].unique().tolist())}")
    # gewünschte Source-Reihenfolge
    order = [s for s in ["TA", "NO", "EX"] if s in sources] + [s for s in sources if s not in ["TA","NO","EX"]]

    has_delta = {"low_delta", "up_delta"}.issubset(df.columns)

    fig = go.Figure()
    for src in order:
        s = d[d["source"] == src].sort_values(xcol)
        if s.empty:
            continue

        # Delta-Ribbon (falls vorhanden)
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

        # Mittellinie
        fig.add_trace(go.Scatter(
            x=s[xcol], y=s["prediction"], mode="lines",
            name=SOURCE_MAP.get(src, src),
            line=dict(color=colors.get(src, None), width=line_width),
            legendgroup=src
        ))

    fig.update_layout(
        height=700, width=700,
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(title="Source"),
    )
    fig.update_xaxes(
        title_text=xtitle, range=[-1, 1],
        tickmode="array", tickvals=[-1, -0.5, 0, 0.5, 1],
        zeroline=False
    )
    fig.update_yaxes(
        title_text="Throughput",
        rangemode=("tozero" if y_zero else None),
        zeroline=False
    )
    return fig

# ----------------------- App --------------------------------
def main():
    st.set_page_config(page_title="Single Throughput Plot", layout="centered")
    st.title("Durchsatz vs. Systemlast (Einzelplot) — Delta-Intervalle")

    path = _find_data_file()
    df = load_data(path)

    zones_in_data = sorted(df["zoning"].dropna().unique().tolist())
    preferred_order = [z for z in ["BU","TD","RA","SQ"] if z in zones_in_data]
    zones_available = preferred_order or zones_in_data

    zone = st.selectbox("Zoning", zones_available, index=0,
                        format_func=lambda z: ZONE_MAP.get(z, z))

    sources_all = sorted(df["source"].dropna().unique().tolist())
    default_sources = [s for s in ["TA","NO","EX"] if s in sources_all] or sources_all
    sources = st.multiselect(
        "Source", options=sources_all, default=default_sources,
        format_func=lambda s: SOURCE_MAP.get(s, s),
    )

    line_width = st.slider("Linienbreite", 1, 6, 3, 1)
    y_zero = st.checkbox("Y-Achse bei 0 beginnen lassen", value=False)

    st.markdown("**Farben**")
    col1, col2, col3 = st.columns(3)
    with col1: col_ta = st.color_picker("Tacted", "#D55E00")
    with col2: col_no = st.color_picker("Normal", "#7A88C2")
    with col3: col_ex = st.color_picker("Exponential", "#7CC68E")
    colors = {"TA": col_ta, "NO": col_no, "EX": col_ex}

    if zone and sources:
        fig = build_single_plot(df, zone, sources, colors, line_width, y_zero, ribbon_alpha=0.18)
        st.plotly_chart(fig, use_container_width=False)
        st.caption(f"Zoning: {ZONE_MAP.get(zone, zone)}")
        if not {"low_delta","up_delta"}.issubset(df.columns):
            st.warning("Delta-Intervalle nicht im Datensatz gefunden – es wird nur die Mittellinie gezeichnet.")
    else:
        st.info("Bitte eine Zoning-Strategie und mindestens eine Quelle wählen.")

if __name__ == "__main__":
    main()