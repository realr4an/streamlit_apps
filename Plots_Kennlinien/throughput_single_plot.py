# ------------------------------------------------------------
# throughput_single_plot_app.py
# Ein einzelner 2D-Plot (kodierte Systemlast –1…+1 vs. Throughput)
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

SOURCE_MAP = {
    "TA": "Tacted",
    "NO": "Normal",
    "EX": "Exponential",
}

# ----------------------- Datei finden -----------------------
def _find_data_file() -> Path:
    patterns = [
        "data.throughput.2d_10_30 4.xlsx",
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
    warnings.filterwarnings(
        "ignore",
        message="Workbook contains no default style",
        category=UserWarning,
        module="openpyxl",
    )
    df = pd.read_excel(path)

    rename_map = {
        "systemload": "systemload",
        "coded_sourceparameter": "systemload",
        "traycontrol": "zoning",
        "distributionstrategy": "zoning",
        "prediction": "prediction",
        "predicted_throughput": "prediction",
        "predicted_mopt": "prediction",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "Source" not in df.columns:
        df["Source"] = "ALL"

    for c in ["systemload", "prediction"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "zoning" in df.columns:
        df["zoning"] = df["zoning"].astype(str).str.upper().str.strip()
    if "Source" in df.columns:
        df["Source"] = df["Source"].astype(str).str.upper().str.strip()
    return df

# ----------------------- Plot bauen -------------------------
def build_single_plot(df: pd.DataFrame, zone: str, sources, colors: dict, line_width: int, y_zero: bool) -> go.Figure:
    xcol = "systemload"  # immer kodiert –1…+1
    xtitle = "Coded source parameter"

    d = df[df["zoning"] == zone]
    order = [s for s in ["TA", "NO", "EX"] if s in sources] + [s for s in sources if s not in ["TA", "NO", "EX"]]

    fig = go.Figure()
    for src in order:
        s = d[d["Source"] == src].sort_values(xcol)
        if s.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=s[xcol],
                y=s["prediction"],
                mode="lines",
                name=SOURCE_MAP.get(src, src),
                line=dict(color=colors.get(src, None), width=line_width),
            )
        )

    fig.update_layout(
        height=700,
        width=700,
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(title="Source"),
    )
    fig.update_xaxes(
        title_text=xtitle,
        range=[-1, 1],
        tickmode="array",
        tickvals=[-1, -0.5, 0, 0.5, 1],
        zeroline=False
    )
    if y_zero:
        # Untere Grenze bei 0, obere automatisch
        fig.update_yaxes(title_text="Throughput", rangemode="tozero", zeroline=False)
    else:
        fig.update_yaxes(title_text="Throughput", zeroline=False)
    return fig

# ----------------------- App --------------------------------
def main():
    st.set_page_config(page_title="Single Throughput Plot", layout="centered")
    st.title("Durchsatz vs. Systemlast (Einzelplot)")

    path = _find_data_file()
    df = load_data(path)

    zones_in_data = sorted(df["zoning"].dropna().unique().tolist())
    preferred_order = [z for z in ["BU", "TD", "RA", "SQ"] if z in zones_in_data]
    zones_available = preferred_order or zones_in_data

    zone = st.selectbox(
        "Zoning",
        zones_available,
        index=0,
        format_func=lambda z: ZONE_MAP.get(z, z),
    )

    sources_all = sorted(df["Source"].dropna().unique().tolist())
    default_sources = [s for s in ["TA", "NO", "EX"] if s in sources_all] or sources_all
    sources = st.multiselect(
        "Source",
        options=sources_all,
        default=default_sources,
        format_func=lambda s: SOURCE_MAP.get(s, s),
    )

    line_width = st.slider("Linienbreite", 1, 6, 3, 1)
    y_zero = st.checkbox("Y-Achse bei 0 beginnen lassen", value=False)

    st.markdown("**Farben**")
    col1, col2, col3 = st.columns(3)
    with col1:
        col_ta = st.color_picker("Tacted", "#D55E00")
    with col2:
        col_no = st.color_picker("Normal", "#7A88C2")
    with col3:
        col_ex = st.color_picker("Exponential", "#7CC68E")
    colors = {"TA": col_ta, "NO": col_no, "EX": col_ex}

    if zone and sources:
        fig = build_single_plot(df, zone, sources, colors, line_width, y_zero)
        st.plotly_chart(fig, use_container_width=False)
        st.caption(f"Zoning: {ZONE_MAP.get(zone, zone)}")
    else:
        st.info("Bitte eine Zoning-Strategie und mindestens eine Quelle wählen.")

if __name__ == "__main__":
    main()
