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
# Pfad zur Beobachtungsdatei
OBS_DATA_FILE = BASE_DIR / "data.xlsx"

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
        "data.throughput.2d_10_30 6.xlsx"
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

@st.cache_data
def load_observed(path: Path) -> pd.DataFrame:
    """Rohdatensatz mit beobachteten throughput & mopt laden.
    Behebt auch doppelte Spalten nach dem Rename (z.B. distributionstrategy + distributinstrategy → zoning).
    """
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_excel(path)
    rename_map = {
        "coded_sourceparameter": "systemload",
        "distributionstrategy": "zoning",
        "distributinstrategy": "zoning",  # Schreibvariante laut Angabe
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    def _collapse_duplicates(frame: pd.DataFrame, col_name: str) -> pd.DataFrame:
        if list(frame.columns).count(col_name) > 1:
            dups = frame.loc[:, frame.columns == col_name]
            # Nimm je Zeile den ersten nicht‑NA Wert
            merged = dups.bfill(axis=1).iloc[:, 0]
            # Entferne alle Duplikat-Spalten
            frame = frame.loc[:, frame.columns != col_name]
            frame[col_name] = merged
        return frame

    for cn in ["zoning", "systemload", "source"]:
        df = _collapse_duplicates(df, cn)

    for c in ["systemload", "throughput", "mopt"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "zoning" in df.columns:
        zoning_col = df["zoning"]
        if isinstance(zoning_col, pd.DataFrame):  # Fallback (sollte nach Collapse nicht mehr passieren)
            zoning_col = zoning_col.iloc[:, 0]
        z = zoning_col.astype(str).str.upper().str.strip()
        df["zoning"] = z.replace(ZONING_NORMALIZE)
    if "source" in df.columns:
        source_col = df["source"]
        if isinstance(source_col, pd.DataFrame):
            source_col = source_col.iloc[:, 0]
        df["source"] = source_col.astype(str).str.upper().str.strip()
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
    ribbon_alpha: float = 0.18,
    observed: pd.DataFrame | None = None,
    show_observed: bool = True,
    font_size: int = 18,
) -> go.Figure:
    xcol = "systemload"  # kodiert –1…+1
    xtitle = "Coded source parameter"

    d = df[df["zoning"] == zone]
    if d.empty:
        st.warning(f"Keine Daten für Zoning '{zone}'. Verfügbare Zonen: {sorted(df['zoning'].unique().tolist())}")
    # gewünschte Source-Reihenfolge
    order = [s for s in ["TA", "NO", "EX"] if s in sources] + [s for s in sources if s not in ["TA","NO","EX"]]

    has_delta = {"low_delta", "up_delta"}.issubset(df.columns)

    # Beobachtungen für dieselbe Zone / Sources
    obs = pd.DataFrame()
    if show_observed and observed is not None and not observed.empty:
        obs = observed[(observed["zoning"] == zone) & (observed["source"].isin(sources))].copy()

    fig = go.Figure()
    # --- Vorhersage-Linien + Intervalle ---
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

    # --- Beobachtete Punkte (throughput) ---
    if not obs.empty:
        # Farbige Punkte je Source (ohne eigene Legendeneinträge)
        for src in order:
            src_pts = obs[obs["source"] == src]
            if src_pts.empty or src_pts["throughput"].isna().all():
                continue
            fig.add_trace(
                go.Scatter(
                    x=src_pts["systemload"],
                    y=src_pts["throughput"],
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=7,
                        color=colors.get(src, "#666"),
                        line=dict(width=0.5, color="#222"),
                    ),
                    name="Observed throughput (colored)",  # Platzhalter, nicht in Legende
                    legendgroup="obs",
                    showlegend=False,
                    hovertemplate="Observed throughput<br>Systemload: %{x}<br>Throughput: %{y}<extra></extra>",
                )
            )
        # Ein einzelner weißer Legendeneintrag (nur in Legende sichtbar)
        fig.add_trace(
            go.Scatter(
                x=[0], y=[0], mode="markers",
                marker=dict(symbol="circle", size=7, color="#FFFFFF", line=dict(width=0.6, color="#000000")),
                name="Observed throughput",
                legendgroup="obs",
                showlegend=True,
                visible="legendonly",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        height=700, width=700,
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(
            title=dict(text="Source", font=dict(size=font_size-2, color="#000000")),
            font=dict(size=font_size-2, color="#000000")
        ),
        font=dict(size=font_size, color="#000000")
    )
    fig.update_xaxes(
        title_text=xtitle, range=[-1, 1],
        tickmode="array", tickvals=[-1, -0.5, 0, 0.5, 1],
        zeroline=False,
        title_font=dict(size=font_size, color="#000000"),
        tickfont=dict(size=font_size-2, color="#000000")
    )
    # Fixe Y-Achse 0–15000, volle Zahlen ohne k-Abkürzung
    fig.update_yaxes(
        title_text="Throughput",
        range=[0, 15000], autorange=False,
        tickformat=".0f",  # keine wissenschaftl. Notation, keine k-Suffixe
        showexponent="none",
        zeroline=False,
        title_font=dict(size=font_size, color="#000000"),
        tickfont=dict(size=font_size-2, color="#000000")
    )
    return fig

# ----------------------- App --------------------------------
def main():
    st.set_page_config(page_title="Single Throughput Plot", layout="centered")
    st.title("Durchsatz vs. Systemlast (Einzelplot) — Delta-Intervalle")

    path = _find_data_file()
    df = load_data(path)
    try:
        observed_df = load_observed(OBS_DATA_FILE)
    except FileNotFoundError:
        observed_df = pd.DataFrame()
        st.warning("Beobachtungsdatei 'data.xlsx' nicht gefunden – keine Punkte geplottet.")

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
    show_observed = st.checkbox("Beobachtete throughput anzeigen", value=True)
    font_size = st.slider("Grund-Schriftgröße", 10, 40, 20, 1)

    st.markdown("**Farben**")
    col1, col2, col3 = st.columns(3)
    with col1: col_ta = st.color_picker("Tacted", "#D55E00")
    with col2: col_no = st.color_picker("Normal", "#7A88C2")
    with col3: col_ex = st.color_picker("Exponential", "#7CC68E")
    colors = {"TA": col_ta, "NO": col_no, "EX": col_ex}

    if zone and sources:
        fig = build_single_plot(
            df, zone, sources, colors, line_width, y_zero, ribbon_alpha=0.18,
            observed=observed_df, show_observed=show_observed, font_size=font_size
        )
        st.plotly_chart(fig, use_container_width=False)
        st.caption(f"Zoning: {ZONE_MAP.get(zone, zone)}")
        if not {"low_delta","up_delta"}.issubset(df.columns):
            st.warning("Delta-Intervalle nicht im Datensatz gefunden – es wird nur die Mittellinie gezeichnet.")
    else:
        st.info("Bitte eine Zoning-Strategie und mindestens eine Quelle wählen.")

if __name__ == "__main__":
    main()