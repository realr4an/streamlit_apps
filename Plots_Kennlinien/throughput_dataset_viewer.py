# ------------------------------------------------------------
# throughput_dataset_viewer.py
#
# 3-D-Kennlinien-Viewer auf Basis des Datensatzes
#   data.throughput.3d.csv
# ------------------------------------------------------------
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------
# Konstanten
# ------------------------------------------------------------
BASE_DIR  = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data.throughput.3d.csv"

# ------------------------------------------------------------
# Daten laden
# ------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, sep=";", decimal=",").rename(
        columns={
            "coded_systemload": "systemload",      # codiert  –1 … +1
            "low.bootstrap":    "lower_boot",
            "up.bootstrap":     "upper_boot",
            "low.analytical":   "lower",
            "up.analytical":    "upper",
        }
    )
    # --- FIX: Spalte explizit in numerischen Typ umwandeln
    df["systemload"] = pd.to_numeric(df["systemload"])
    
    # --- NEW:  codierten Wert in die reale Systemlast (54-62) zurückrechnen
    MID, HALF_RANGE = 58, 4                       #  (54 ↔ –1) … (62 ↔ +1)
    df["systemload_raw"] = df["systemload"] * HALF_RANGE + MID

    return df



# ------------------------------------------------------------
# Helper: HEX → rgba-String
# ------------------------------------------------------------
def _hex_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ------------------------------------------------------------
# 3-D-Figure bauen
# ------------------------------------------------------------
def build_figure(sub: pd.DataFrame, cfg: dict, interval: str = "Bootstrap") -> go.Figure:
    # Raster für Surface-Plot
    P = sub.pivot(index="loadunitcap", columns="systemload_raw", values="prediction").values

    if interval == "Bootstrap":
        low_col, up_col = "lower_boot", "upper_boot"
    else:
        low_col, up_col = "lower", "upper"

    L = sub.pivot(index="loadunitcap", columns="systemload_raw", values=low_col).values
    U = sub.pivot(index="loadunitcap", columns="systemload_raw", values=up_col).values

    X, Y = np.meshgrid(
        sorted(sub["systemload_raw"].unique()),
        sorted(sub["loadunitcap"].unique()),
    )

    fig = go.Figure()
    hovertemplate = (
        "<b>Systemlast: %{x:.2f}</b><br>"
        "<b>Losgröße: %{y:.2f}</b><br>"
        "Prognose: %{customdata[0]:.2f}<br>"
        "Untere Grenze: %{customdata[1]:.2f}<br>"
        "Obere Grenze: %{customdata[2]:.2f}<extra></extra>"
    )
    customdata = np.stack([P, L, U], axis=-1)

    # Mittelfläche
    fig.add_surface(
        x=X, y=Y, z=P,
        customdata=customdata,
        surfacecolor=np.zeros_like(P),
        colorscale=[[0, cfg["pred_color"]], [1, cfg["pred_color"]]],
        showscale=False,
        hovertemplate=hovertemplate,
    )

    # Volumen-Füllung (optional)
    if cfg["show_fill"]:
        rgba = _hex_rgba(cfg["fill_color"], cfg["fill_alpha"])
        for t in np.linspace(0, 1, cfg["layers"] + 2)[1:-1]:
            Z = L + t * (U - L)
            fig.add_surface(
                x=X, y=Y, z=Z,
                surfacecolor=np.zeros_like(Z),
                colorscale=[[0, rgba], [1, rgba]],
                showscale=False,
                opacity=cfg["fill_alpha"],
                hoverinfo="none",
            )

    # Deckel
    for Z, col in [(U, cfg["upper_color"]), (L, cfg["lower_color"])]:
        fig.add_surface(
            x=X, y=Y, z=Z,
            customdata=customdata,
            surfacecolor=np.zeros_like(Z),
            colorscale=[[0, col], [1, col]],
            showscale=False,
            opacity=cfg["deckel_alpha"],
            hovertemplate=hovertemplate,
        )

    # Seitenwände (optional)
    if cfg["show_walls"]:
        rgba = _hex_rgba(cfg["fill_color"], cfg["wall_alpha"])
        for i in [0, -1]:  # vorne / hinten
            fig.add_surface(
                x=np.vstack([X[i], X[i]]),
                y=np.vstack([Y[i], Y[i]]),
                z=np.vstack([L[i], U[i]]),
                surfacecolor=np.zeros_like(np.vstack([L[i], U[i]])),
                colorscale=[[0, rgba], [1, rgba]],
                showscale=False,
                opacity=cfg["wall_alpha"],
                hoverinfo="none",
            )
        for j in [0, -1]:  # links / rechts
            fig.add_surface(
                x=np.column_stack([X[:, j], X[:, j]]),
                y=np.column_stack([Y[:, j], Y[:, j]]),
                z=np.column_stack([L[:, j], U[:, j]]),
                surfacecolor=np.zeros_like(np.column_stack([L[:, j], U[:, j]])),
                colorscale=[[0, rgba], [1, rgba]],
                showscale=False,
                opacity=cfg["wall_alpha"],
                hoverinfo="none",
            )

    fig.update_layout(
        margin=dict(l=5, r=5, t=35, b=5),
        scene=dict(
            xaxis_title="Systemlast (kodiert)",
            yaxis_title="Losgröße",
            zaxis_title="Durchsatz",
            aspectratio=dict(x=1, y=1, z=0.6),
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.85)),
            zaxis=dict(range=cfg.get("z_range")),
        ),
    )
    return fig


# ------------------------------------------------------------
# Streamlit-Frontend
# ------------------------------------------------------------
def main():
    df = load_data()

    st.set_page_config(page_title="3-D-Kennlinien (Datensatz)", layout="wide")
    st.title("3-D-Kennlinien mit 90 %-Intervall (aus Datei)")

    # --- Sidebar: Anzeige-Einstellungen -----------------------
    st.sidebar.header("Anzeige-Einstellungen")
    cfg = dict(
        pred_color   = st.sidebar.color_picker("Farbe Mittelfläche",  "#FFB400"),
        upper_color  = st.sidebar.color_picker("Farbe oberer Deckel", "#282828"),
        lower_color  = st.sidebar.color_picker("Farbe unterer Deckel", "#282828"),
        fill_color   = st.sidebar.color_picker("Farbe Volumen",       "#3C3C3C"),
        fill_alpha   = st.sidebar.slider("Volumen-Transparenz", 0.05, 0.9,  0.18, 0.01),
        deckel_alpha = st.sidebar.slider("Deckel-Transparenz",  0.1,  1.0,  0.45, 0.05),
        show_fill    = st.sidebar.checkbox("Volumenfüllung anzeigen", False),
        show_walls   = st.sidebar.checkbox("Seitliche Wände anzeigen", False),
        wall_alpha   = st.sidebar.slider("Wand-Transparenz",    0.05, 1.0,  0.25, 0.05),
        layers       = st.sidebar.slider("Füll-Schichten (Dichte)", 4, 40, 20, 2),
        interval_type = st.sidebar.selectbox("Intervall-Variante", ["Bootstrap", "Analytical"]),
    )

    # --- Zoning-Auswahl --------------------------------------
    zones     = df["zoning"].unique().tolist()
    sel_zones = st.multiselect("Zoning-Strategien:", zones, default=zones)

    # Globalen z-Achsen-Bereich für Vergleichbarkeit
    if sel_zones:
        plot_data = df[df["zoning"].isin(sel_zones)]
        if not plot_data.empty:
            if cfg["interval_type"] == "Bootstrap":
                z_min = plot_data["lower_boot"].min()
                z_max = plot_data["upper_boot"].max()
            else:
                z_min = plot_data["lower"].min()
                z_max = plot_data["upper"].max()
            cfg["z_range"] = [z_min, z_max]

    # --- Darstellung -----------------------------------------
    cols = st.columns(max(len(sel_zones), 1))
    for col, z in zip(cols, sel_zones):
        fig = build_figure(
            df[df["zoning"] == z],
            cfg,
            interval=cfg["interval_type"]
        )
        col.subheader(f"{z} – 90 % Intervall")
        col.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
