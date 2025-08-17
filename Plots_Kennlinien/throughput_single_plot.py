# ------------------------------------------------------------
# throughput_single_plot_app.py
# Ein einzelner 2D-Plot (Systemlast vs. Throughput) in Streamlit
# ------------------------------------------------------------
from pathlib import Path
import warnings
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent

# ----------------------- Mappings ---------------------------
# Zonen: Kürzel → Anzeigename
ZONE_MAP = {
    "BU": "Bottom-Up",       # ggf. "Bottem-Up" schreiben
    "TD": "Top-Down",
    "RA": "Random",
    "SQ": "Shortest Queue",
}

# Quellen: Kürzel → Anzeigename
SOURCE_MAP = {
    "TA": "Tacted",
    "NO": "Normal",
    "EX": "Exponential",
}

# ----------------------- Datei finden -----------------------
def _find_data_file() -> Path:
    """
    Sucht typische Dateinamen im selben Verzeichnis wie dieses Script.
    Nimmt die erste gefundene Datei.
    """
    patterns = [
        "data.throughput.2d*.xlsx",
        "data.mopt.2d*.xlsx",
        "data.throughput.2d_10_30 3.xlsx",  # falls exakt so benannt
    ]
    for pat in patterns:
        cands = sorted(BASE_DIR.glob(pat))
        if cands:
            return cands[0]
    raise FileNotFoundError(
        "Keine passende Excel-Datei gefunden (z. B. data.throughput.2d*.xlsx oder data.mopt.2d*.xlsx)."
    )

# ----------------------- Laden & Aufbereiten ----------------
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    # Openpyxl-Warnung unterdrücken
    warnings.filterwarnings(
        "ignore",
        message="Workbook contains no default style",
        category=UserWarning,
        module="openpyxl",
    )

    df = pd.read_excel(path)

    # Spalten harmonisieren (alte & neue Datensätze)
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
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Falls 'source' fehlt, eine Sammelkategorie setzen
    if "source" not in df.columns:
        df["source"] = "ALL"

    # Numerik erzwingen
    for c in ["systemload", "prediction"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Reale Systemlast (54–62) aus kodierter (–1…+1) zurückrechnen
    if "systemload" in df.columns:
        MID, HALF_RANGE = 58, 4  # (54 ↔ –1) … (62 ↔ +1)
        df["systemload_raw"] = df["systemload"] * HALF_RANGE + MID

    # Zoning normalisieren  (wichtig: .str.upper(), nicht .upper())
    if "zoning" in df.columns:
        df["zoning"] = df["zoning"].astype(str).str.upper().str.strip()

    # Source normalisieren
    if "source" in df.columns:
        df["source"] = df["source"].astype(str).str.upper().str.strip()

    return df

# ----------------------- Plot bauen -------------------------
def build_single_plot(df: pd.DataFrame, zone: str, sources, use_raw_x: bool, colors: dict, line_width: int) -> go.Figure:
    xcol = "systemload_raw" if use_raw_x else "systemload"
    xtitle = "Coded source parameter" if use_raw_x else "source parameter"

    d = df[df["zoning"] == zone]
    # feste Reihenfolge für die bekannten Quellen
    order = [s for s in ["TA", "NO", "EX"] if s in sources] + [s for s in sources if s not in ["TA", "NO", "EX"]]

    fig = go.Figure()
    for src in order:
        s = d[d["source"] == src].sort_values(xcol)
        if s.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=s[xcol],
                y=s["prediction"],
                mode="lines",
                name=SOURCE_MAP.get(src, src),  # Legende: ausgeschriebener Name
                line=dict(color=colors.get(src, None), width=line_width),
            )
        )

    # Achsen/Optik
    fig.update_layout(
        height=700,
        width=700,
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(title="Source"),
    )
    fig.update_xaxes(title_text=xtitle, range=[-1, 1] if not use_raw_x else None, zeroline=False)
    fig.update_yaxes(title_text="Throughput", zeroline=False)
    return fig

# ----------------------- App --------------------------------
def main():
    st.set_page_config(page_title="Single Throughput Plot", layout="centered")
    st.title("Durchsatz vs. Systemlast (Einzelplot)")

    path = _find_data_file()
    df = load_data(path)

    # UI: Zoning-Auswahl (zeigt lange Namen, liefert intern Kürzel)
    zones_in_data = sorted(df["zoning"].dropna().unique().tolist())
    preferred_order = [z for z in ["BU", "TD", "RA", "SQ"] if z in zones_in_data]
    zones_available = preferred_order or zones_in_data

    zone = st.selectbox(
        "Zoning",
        zones_available,
        index=0,
        format_func=lambda z: ZONE_MAP.get(z, z),  # Anzeige: langer Name
    )

    # Quellen-Auswahl (zeigt lange Namen, liefert intern Kürzel)
    sources_all = sorted(df["source"].dropna().unique().tolist())
    default_sources = [s for s in ["TA", "NO", "EX"] if s in sources_all] or sources_all
    sources = st.multiselect(
        "Source",
        options=sources_all,
        default=default_sources,
        format_func=lambda s: SOURCE_MAP.get(s, s),  # Anzeige: langer Name
    )

    use_raw_x = st.checkbox("X-Achse: reale Systemlast (54–62)", value=True)
    line_width = st.slider("Linienbreite", 1, 6, 3, 1)

    st.markdown("**Farben**")
    col1, col2, col3 = st.columns(3)
    with col1:
        col_ta = st.color_picker("Tacted", "#D55E00")
    with col2:
        col_no = st.color_picker("Normal", "#7A88C2")
    with col3:
        col_ex = st.color_picker("Exponential", "#7CC68E")
    # Farben bleiben intern auf Kürzel gemappt
    colors = {"TA": col_ta, "NO": col_no, "EX": col_ex}

    # Plot
    if zone and sources:
        fig = build_single_plot(df, zone, sources, use_raw_x, colors, line_width)
        st.plotly_chart(fig, use_container_width=False)
        st.caption(f"Zoning: {ZONE_MAP.get(zone, zone)}")
    else:
        st.info("Bitte eine Zoning-Strategie und mindestens eine Quelle wählen.")

if __name__ == "__main__":
    main()
