# -*- coding: utf-8 -*-
"""
Kennlinien-Analyse mit farbigem Prognoseintervall
=================================================
✓ Excel-Import + Design-Sampling
✓ Poisson-GLM + AIC-Suche
✓ Vorhersagefläche (kräftig, einfarbig)
✓ Obere / untere Intervallgrenze (einfarbig)
✓ Volumen dazwischen blass eingefärbt
"""

# ---------------------------------------------------------------------------
# 1. EINSTELLUNGEN  ----------------------------------------------------------
# ---------------------------------------------------------------------------
DATA_DIR    = "Daten&Programme"       # Ordner mit den Excel-Dateien
DATA_FILE   = "Kennlinien_Betriebskenngroessen_MitLosgroesse12468.xlsx"
DESIGN_FILE = "CCDVariante2.xlsx"

GRID_RES    = 60                      # ng  (kleiner → schneller / grober)
PRED_COLOR  = "rgb(255,180,0)"        # kräftige Mittel­fläche
UPPER_COLOR = "rgb(40,40,40)"         # obere Deckelfläche
LOWER_COLOR = "rgb(40,40,40)"         # untere Deckelfläche
FILL_COLOR  = "rgba(60,60,60,0.18)"   # Volumenfarbe (blass)
LAYERS_FILL = 20                      # „Schichtzahl“ zum Füllen des Volumens
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2. BIBLIOTHEKEN  -----------------------------------------------------------
# ---------------------------------------------------------------------------
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.formula.api import glm


# ---------------------------------------------------------------------------
# 3. HILFSFUNKTIONEN  --------------------------------------------------------
# ---------------------------------------------------------------------------
def fill_interval_volume(fig, X, Y, Z_lo, Z_hi,
                         n_layers: int = LAYERS_FILL,
                         color: str = FILL_COLOR) -> None:
    """füllt den Raum zwischen Z_lo und Z_hi mit vielen feinen Flächen"""
    alpha = float(color.split(",")[-1].rstrip(")"))  # Alpha aus rgba
    for t in np.linspace(0, 1, n_layers + 2)[1:-1]:  # ohne 0 & 1
        Z_mid = Z_lo + t * (Z_hi - Z_lo)
        fig.add_surface(
            x=X, y=Y, z=Z_mid,
            showscale=False,
            surfacecolor=np.zeros_like(Z_mid),
            colorscale=[[0, color], [1, color]],
            opacity=alpha
        )


# ---------------------------------------------------------------------------
# 4. DATEN EINLESEN & AUFBEREITEN  ------------------------------------------
# ---------------------------------------------------------------------------
data_path   = Path(DATA_DIR) / DATA_FILE
design_path = Path(DATA_DIR) / DESIGN_FILE

data   = pd.read_excel(data_path).rename(columns=str.strip)
design = pd.read_excel(design_path, usecols=[0, 1, 2]).rename(columns=str.strip)

# Design-Spaltennamen an Daten-Spalten angleichen (Robustheit gegen Tippfehler)
design.rename(columns={
    "Vorfahrtstrategie": "Vorfahrtsstrategie",   # häufigster Schreibfehler
    "Losgröße": "Losgroesse"                     # falls deutsches ß
}, inplace=True)

# Stichprobe gemäß Design
rows = []
for _, row in design.iterrows():
    sel = (
        (data["Systemlast"] == row["Systemlast"]) &
        (data["Losgroesse"] == row["Losgroesse"]) &
        (data["Vorfahrtsstrategie"] == row["Vorfahrtsstrategie"])
    )
    rows.append(data.loc[sel])
data_small = pd.concat(rows, ignore_index=True)

# Relevante Spalten + Umbenennen
df = (
    data_small[data_small["Deadlock"] == "-"]
    .loc[:, ["Durchsatz", "Vorfahrtsstrategie", "Systemlast", "Losgroesse"]]
    .rename(columns={
        "Durchsatz": "throughput",
        "Vorfahrtsstrategie": "zoning",
        "Systemlast": "systemload",
        "Losgroesse": "loadunitcap"
    })
)
df["zoning"] = df["zoning"].astype("category")

# Kodierung [-1, +1] (optional, für spätere Experimente)
for col in ["systemload", "loadunitcap"]:
    mid = df[col].mean()
    rng = df[col].max() - df[col].min()
    df[f"coded_{col}"] = (df[col] - mid) / (rng / 2)

print(f"Daten geladen: n = {len(df)} Zeilen")


# ---------------------------------------------------------------------------
# 5. MODELLANPASSUNG (Poisson-GLM, AIC-Suche)  -------------------------------
# ---------------------------------------------------------------------------
base_terms = ["systemload", "I(systemload**2)", "zoning", "systemload:zoning"]
best_aic, best_formula, best_model = np.inf, None, None

for k in range(1, len(base_terms) + 1):
    for combo in itertools.combinations(base_terms, k):
        formula = "throughput ~ " + " + ".join(combo)
        model = glm(formula, data=df,
                    family=sm.families.Poisson()).fit()
        if model.aic < best_aic:
            best_aic, best_formula, best_model = model.aic, formula, model

print(f"Bestes Modell: {best_formula}  (AIC {best_aic:0.1f})")


# ---------------------------------------------------------------------------
# 6. VORHERSAGE-GRID & KONFIDENZINTERVALLE  ----------------------------------
# ---------------------------------------------------------------------------
grid = pd.DataFrame(
    list(itertools.product(
        df["zoning"].cat.categories,
        np.linspace(df["systemload"].min(),  df["systemload"].max(),  GRID_RES),
        np.linspace(df["loadunitcap"].min(), df["loadunitcap"].max(), GRID_RES)
    )),
    columns=["zoning", "systemload", "loadunitcap"]
)

grid["prediction"] = best_model.predict(grid)
ci = best_model.get_prediction(grid).summary_frame(alpha=0.05)
grid[["lower", "upper"]] = ci[["mean_ci_lower", "mean_ci_upper"]]


# ---------------------------------------------------------------------------
# 7. PLOTTEN  – Mittel­fläche, Volumen-Fill + vertikale Seitenwände
# ---------------------------------------------------------------------------
def add_side_walls(fig, X, Y, Z_lo, Z_hi, color=FILL_COLOR, alpha=0.5):
    """fügt vier senkrechte Wände (Front, Back, Left, Right) ein"""
    rgba_wall = color.rsplit(",", 1)[0] + f",{alpha})"   # Alpha auf Wunschhöhe

    # → 2-Zeilen-Grids für jede Wand
    # Vorderseite (min Y)
    y_front = Y.min(axis=0)     # konstantes Y
    x_front = X[0, :]
    fig.add_surface(
        x=np.vstack([x_front, x_front]),
        y=np.vstack([y_front, y_front]),
        z=np.vstack([Z_lo[0, :], Z_hi[0, :]]),
        showscale=False,
        colorscale=[[0, rgba_wall], [1, rgba_wall]],
        opacity=alpha
    )
    # Rückseite (max Y)
    y_back = Y.max(axis=0)
    x_back = X[-1, :]
    fig.add_surface(
        x=np.vstack([x_back, x_back]),
        y=np.vstack([y_back, y_back]),
        z=np.vstack([Z_lo[-1, :], Z_hi[-1, :]]),
        showscale=False,
        colorscale=[[0, rgba_wall], [1, rgba_wall]],
        opacity=alpha
    )
    # Linke Seite (min X)
    x_left = X[:, 0]
    y_left = Y[:, 0]
    fig.add_surface(
        x=np.column_stack([x_left, x_left]),
        y=np.column_stack([y_left, y_left]),
        z=np.column_stack([Z_lo[:, 0], Z_hi[:, 0]]),
        showscale=False,
        colorscale=[[0, rgba_wall], [1, rgba_wall]],
        opacity=alpha
    )
    # Rechte Seite (max X)
    x_right = X[:, -1]
    y_right = Y[:, -1]
    fig.add_surface(
        x=np.column_stack([x_right, x_right]),
        y=np.column_stack([y_right, y_right]),
        z=np.column_stack([Z_lo[:, -1], Z_hi[:, -1]]),
        showscale=False,
        colorscale=[[0, rgba_wall], [1, rgba_wall]],
        opacity=alpha
    )


for z in df["zoning"].cat.categories:
    sub = grid[grid["zoning"] == z]

    # --- Matrix-Form --------------------------------------------------------
    P = sub.pivot(index="loadunitcap", columns="systemload", values="prediction").values
    L = sub.pivot(index="loadunitcap", columns="systemload", values="lower").values
    U = sub.pivot(index="loadunitcap", columns="systemload", values="upper").values
    X, Y = np.meshgrid(sorted(sub["systemload"].unique()),
                       sorted(sub["loadunitcap"].unique()))

    fig = go.Figure()

    # 1) kräftige Mittel­fläche
    fig.add_surface(
        x=X, y=Y, z=P,
        surfacecolor=np.zeros_like(P),
        colorscale=[[0, PRED_COLOR], [1, PRED_COLOR]],
        showscale=False, opacity=1
    )

    # 2) Volumen zwischen L & U
    fill_interval_volume(fig, X, Y, L, U,
                         n_layers=LAYERS_FILL,
                         color=FILL_COLOR)

    # 3) Deckel oben/unten
    for Z, col in [(U, UPPER_COLOR), (L, LOWER_COLOR)]:
        fig.add_surface(
            x=X, y=Y, z=Z,
            surfacecolor=np.zeros_like(Z),
            colorscale=[[0, col], [1, col]],
            showscale=False, opacity=0.45
        )

    # 4) → neue Seiten­wände
    add_side_walls(fig, X, Y, L, U, color=FILL_COLOR, alpha=0.50)

    # Layout
    fig.update_layout(
        title=f"Zoning: {z}",
        scene=dict(
            xaxis=dict(title="System load",  backgroundcolor="#EAF2F8"),
            yaxis=dict(title="Load unit capacity", backgroundcolor="#EAF2F8"),
            zaxis=dict(title="Throughput"),
            aspectratio=dict(x=1, y=1, z=0.6),
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
        ),
        paper_bgcolor="white",
        width=950, height=700,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    fig.show()
