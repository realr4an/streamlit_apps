# ------------------------------------------------------------
# throughput_single_plot_app.py
# Ein einzelner 2D-Plot (kodierte Systemlast vs. Throughput) in Streamlit
# ------------------------------------------------------------
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent

# ----------------------- Datei finden -----------------------
def _find_data_file() -> Path:
    # versucht beide Namensschemata; nimmt die erste gefundene Datei
    patterns = ["data.throughput.2d*.xlsx", "data.mopt.2d*.xlsx"]
    for pat in patterns:
        cands = sorted(BASE_DIR.glob(pat))
        if cands:
            return cands[0]
    raise FileNotFoundError("Keine passende Excel-Datei gefunden (data.throughput.2d*.xlsx oder data.mopt.2d*.xlsx).")

# ----------------------- Laden & Aufbereiten ----------------
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    # Warnung von openpyxl unterdrücken
    import warnings
    warnings.filterwarnings("ignore", message="Workbook contains no default style", category=UserWarning, module="openpyxl")

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

    if "source" not in df.columns:
        df["source"] = "ALL"

    for c in ["systemload", "prediction"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # für optionalen Raw-View (54–62)
    if "systemload" in df.columns:
        MID, HALF_RANGE = 58, 4
        df["systemload_raw"] = df["systemload"] * HALF_RANGE + MID

    return df

# ----------------------- Plot bauen -------------------------
def build_single_plot(df: pd.DataFrame, zone: str, sources: list[str], use_raw_x: bool, colors: dict, line_width: int) -> go.Figure:
    xcol = "systemload_raw" if use_raw_x else "systemload"
    xtitle = "System load (raw)" if use_raw_x else "Coded system load"

    d = df[df["zoning"] == zone]
    # feste Reihenfolge wie im Screenshot
    order = [s for s in ["TA", "NO", "EX"] if s in sources] + [s for s in sources if s not in ["TA","NO","EX"]]

    fig = go.Figure()
    for src in order:
        s = d[d["source"] == src].sort_values(xcol)
        if s.empty: 
            continue
        fig.add_trace(go.Scatter(
            x=s[xcol],
            y=s["prediction"],
            mode="lines",
            name=src,
            line=dict(color=colors.get(src, None), width=line_width),
        ))

    # Achsen/Optik
    fig.update_layout(
        height=700, width=700,
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(title="source"),
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

    # UI
    zones_available = [z for z in ["BU","TD","RA","SQ"] if z in df["zoning"].unique().tolist()] or sorted(df["zoning"].unique().tolist())
    zone = st.selectbox("Zoning", zones_available, index=0)

    sources_all = sorted(df["source"].unique().tolist())
    sources = st.multiselect("source", sources_all, default=[s for s in ["TA","NO","EX"] if s in sources_all] or sources_all)

    use_raw_x = st.checkbox("X-Achse: reale Systemlast (54–62)", value=False)  # wie im Bild: kodiert
    line_width = st.slider("Linienbreite", 1, 6, 3, 1)

    st.markdown("**Farben**")
    col1, col2, col3 = st.columns(3)
    with col1: col_ta = st.color_picker("TA", "#D55E00")
    with col2: col_no = st.color_picker("NO", "#7A88C2")  # leicht ggplot-ähnlich
    with col3: col_ex = st.color_picker("EX", "#7CC68E")
    colors = {"TA": col_ta, "NO": col_no, "EX": col_ex}

    # Plot
    if zone and sources:
        fig = build_single_plot(df, zone, sources, use_raw_x, colors, line_width)
        st.plotly_chart(fig, use_container_width=False)
    else:
        st.info("Bitte eine Zoning-Strategie und mindestens eine Quelle wählen.")

if __name__ == "__main__":
    main()
