# app_pred_interval.py  ───────────────────────────────────────────
# Streamlit-App mit *Prädiktions- (Prognose-)intervallen* für ein
# Poisson-GLM.  Erwartungswert-, Konfidenz- und Prädiktions­intervalle
# werden analytisch berechnet (Delta-Methode + Poisson-Varianz).

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as st  # für z-Wert
import statsmodels.api as sm
from statsmodels.formula.api import glm
import streamlit as stlit  # Alias, damit st* ≠ st (scipy.stats)

# ------------------------------------------------------------------
# FESTE DATEI-PFAD-PARAMETER
# ------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "Daten&Programme"
DATA_FILE  = "Kennlinien_Betriebskenngroessen_MitLosgroesse12468.xlsx"
DESIGN_FILE = "CCDVariante2.xlsx"
GRID_RES   = 60
ALPHA      = 0.05
Z_975      = st.norm.ppf(1 - ALPHA / 2)      # ≈ 1.95996

# ------------------------------------------------------------------
# HILFSFUNKTIONEN
# ------------------------------------------------------------------
def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """#RRGGBB  →  'rgba(r,g,b,alpha)'"""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def fill_volume(fig, X, Y, Z_lo, Z_hi, color_hex, layers, alpha):
    rgba = hex_to_rgba(color_hex, alpha)
    for t in np.linspace(0, 1, layers + 2)[1:-1]:
        Z = Z_lo + t * (Z_hi - Z_lo)
        fig.add_surface(
            x=X,
            y=Y,
            z=Z,
            surfacecolor=np.zeros_like(Z),
            colorscale=[[0, rgba], [1, rgba]],
            showscale=False,
            opacity=alpha,
        )


def add_side_walls(fig, X, Y, Z_lo, Z_hi, color_hex, alpha):
    rgba = hex_to_rgba(color_hex, alpha)
    # Front & Back
    for i in [0, -1]:
        fig.add_surface(
            x=np.vstack([X[i], X[i]]),
            y=np.vstack([Y[i], Y[i]]),
            z=np.vstack([Z_lo[i], Z_hi[i]]),
            colorscale=[[0, rgba], [1, rgba]],
            showscale=False,
            opacity=alpha,
        )
    # Left & Right
    for j in [0, -1]:
        fig.add_surface(
            x=np.column_stack([X[:, j], X[:, j]]),
            y=np.column_stack([Y[:, j], Y[:, j]]),
            z=np.column_stack([Z_lo[:, j], Z_hi[:, j]]),
            colorscale=[[0, rgba], [1, rgba]],
            showscale=False,
            opacity=alpha,
        )


def build_figure(sub: pd.DataFrame, cfg: dict):
    """Plot 3-D-Kennlinie mit Prognoseintervallen."""
    # Matrix-Form
    P = sub.pivot(index="loadunitcap", columns="systemload", values="prediction").values
    L = sub.pivot(index="loadunitcap", columns="systemload", values="lower").values
    U = sub.pivot(index="loadunitcap", columns="systemload", values="upper").values
    X, Y = np.meshgrid(sorted(sub["systemload"].unique()), sorted(sub["loadunitcap"].unique()))

    fig = go.Figure()

    # Mittel­fläche λ̂
    fig.add_surface(
        x=X,
        y=Y,
        z=P,
        surfacecolor=np.zeros_like(P),
        colorscale=[[0, cfg["pred_color"]], [1, cfg["pred_color"]]],
        showscale=False,
        opacity=1,
    )

    # Volumen (Prädiktionsintervall)
    if cfg["show_fill"]:
        fill_volume(fig, X, Y, L, U, cfg["fill_color"], cfg["layers"], cfg["fill_alpha"])

    # Deckel
    for Z, col in [(U, cfg["upper_color"]), (L, cfg["lower_color"])]:
        fig.add_surface(
            x=X,
            y=Y,
            z=Z,
            surfacecolor=np.zeros_like(Z),
            colorscale=[[0, col], [1, col]],
            showscale=False,
            opacity=cfg["deckel_alpha"],
        )

    # Wände
    if cfg["show_walls"]:
        add_side_walls(fig, X, Y, L, U, cfg["fill_color"], cfg["wall_alpha"])

    # Layout
    fig.update_layout(
        margin=dict(l=5, r=5, t=35, b=5),
        scene=dict(aspectratio=dict(x=1, y=1, z=0.6), camera=dict(eye=dict(x=1.4, y=1.4, z=0.85))),
    )
    return fig


# ------------------------------------------------------------------
# STREAMLIT-APP
# ------------------------------------------------------------------
stlit.set_page_config(page_title="Kennlinien-Viewer", layout="wide")
stlit.title("3-D-Kennlinien mit Prognoseintervallen")

# ---------- Sidebar ----------
stlit.sidebar.header("Anzeige-Einstellungen")
pred_color = stlit.sidebar.color_picker("Farbe Mittel­fläche", "#FFB400")
upper_color = stlit.sidebar.color_picker("Farbe oberer Deckel", "#282828")
lower_color = stlit.sidebar.color_picker("Farbe unterer Deckel", "#282828")
fill_color = stlit.sidebar.color_picker("Farbe Volumen", "#3C3C3C")

fill_alpha = stlit.sidebar.slider("Volumen-Transparenz", 0.05, 0.9, 0.18, 0.01)
deckel_alpha = stlit.sidebar.slider("Deckel-Transparenz", 0.1, 1.0, 0.45, 0.05)

show_fill = stlit.sidebar.checkbox("Volumenfüllung anzeigen", True)
show_walls = stlit.sidebar.checkbox("Seitliche Wände anzeigen", True)
wall_alpha = stlit.sidebar.slider("Wand-Transparenz", 0.05, 1.0, 0.25, 0.05)

layers = stlit.sidebar.slider("Füll-Schichten (Dichte)", 4, 40, 20, 2)

cfg = dict(
    pred_color=pred_color,
    upper_color=upper_color,
    lower_color=lower_color,
    fill_color=fill_color,
    fill_alpha=fill_alpha,
    deckel_alpha=deckel_alpha,
    layers=layers,
    show_fill=show_fill,
    show_walls=show_walls,
    wall_alpha=wall_alpha,
)

# ---------- Daten & Modell (Cache) ----------
@stlit.cache_data
def load_and_model():
    # ---------------- Daten einlesen ----------------
    data = pd.read_excel(DATA_DIR / DATA_FILE).rename(columns=str.strip)
    design = pd.read_excel(DATA_DIR / DESIGN_FILE, usecols=[0, 1, 2]).rename(columns=str.strip)
    design.rename(columns={"Vorfahrtstrategie": "Vorfahrtsstrategie"}, inplace=True)

    rows = []
    for _, r in design.iterrows():
        rows.append(
            data.loc[
                (data["Systemlast"] == r["Systemlast"])
                & (data["Losgroesse"] == r["Losgroesse"])
                & (data["Vorfahrtsstrategie"] == r["Vorfahrtsstrategie"])
            ]
        )
    df = pd.concat(rows, ignore_index=True)
    df = (
        df[df["Deadlock"] == "-"]
        .loc[:, ["Durchsatz", "Vorfahrtsstrategie", "Systemlast", "Losgroesse"]]
        .rename(
            columns={
                "Durchsatz": "throughput",
                "Vorfahrtsstrategie": "zoning",
                "Systemlast": "systemload",
                "Losgroesse": "loadunitcap",
            }
        )
    )
    df["zoning"] = df["zoning"].astype("category")

    # ---------------- Poisson-Modell (bestes nach AIC) ---------------
    terms = ["systemload", "I(systemload**2)", "zoning", "systemload:zoning"]
    best_aic, best_model = np.inf, None
    for k in range(1, len(terms) + 1):
        for combo in itertools.combinations(terms, k):
            m = glm("throughput ~ " + " + ".join(combo), data=df, family=sm.families.Poisson()).fit()
            if m.aic < best_aic:
                best_aic, best_model = m.aic, m

    # ---------------- Vorhersage-Grid ----------------
    grid = pd.DataFrame(
        list(
            itertools.product(
                df["zoning"].cat.categories,
                np.linspace(df["systemload"].min(), df["systemload"].max(), GRID_RES),
                np.linspace(df["loadunitcap"].min(), df["loadunitcap"].max(), GRID_RES),
            )
        ),
        columns=["zoning", "systemload", "loadunitcap"],
    )

    # Erwartungswert λ̂
    pred_res = best_model.get_prediction(grid)
    grid["prediction"] = pred_res.predicted_mean

    # ----------- Konfidenzintervall des Mittelwerts -----------------
    ci_mean = pred_res.summary_frame(alpha=ALPHA)
    grid[["mean_lower", "mean_upper"]] = ci_mean[["mean_ci_lower", "mean_ci_upper"]]
    se_mean = ci_mean["mean_se"]

    # ----------- Prädiktionsintervall -------------------------------
    # Var_total = Var_mean + Var_poisson(λ)   => SE_pred = sqrt(Var_total)
    var_mean = se_mean**2
    var_pred = var_mean + grid["prediction"]
    se_pred = np.sqrt(var_pred)

    grid["lower"] = (grid["prediction"] - Z_975 * se_pred).clip(lower=0)
    grid["upper"] = grid["prediction"] + Z_975 * se_pred

    return df, grid


df, grid = load_and_model()

# ---------- Strategien wählen & nebeneinander plotten -------------
zones = df["zoning"].cat.categories.tolist()
sel_zones = stlit.multiselect("Zoning-Strategien:", zones, default=zones)

cols = stlit.columns(len(sel_zones) or 1)
for col, z in zip(cols, sel_zones):
    fig = build_figure(grid[grid["zoning"] == z], cfg)
    col.subheader(f"{z} – 95 % Prognoseintervall")
    col.plotly_chart(fig, use_container_width=True)
