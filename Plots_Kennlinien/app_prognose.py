"""
app_prognoseintervall.py
Streamlit‑App, die den Workflow des bereitgestellten R‑Skripts Schritt für Schritt repliziert
(Data‑Import → Modellselektion → Vorhersage → 3‑D‑Plot mit 95‑%‑Prognoseintervall).

Autor: (2025‑07‑23 Rewrite)
"""
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import glm
import streamlit as st


# ------------------------------------------------------------------
# Konstanten / feste Parameter (wie im R‑Skript)
# ------------------------------------------------------------------
BASE_DIR     = Path(__file__).resolve().parent
DATA_DIR     = BASE_DIR / "Daten&Programme"
DATA_FILE    = "Kennlinien_Betriebskenngroessen_MitLosgroesse12468.xlsx"
DESIGN_FILE  = "CCDVariante2.xlsx"
ALPHA        = 0.05
Z_975        = stats.norm.ppf(1 - ALPHA / 2)   # ≈ 1.960


class ThroughputApp:
    """Kapselt den kompletten Berechnung‑ & Visualisierungs‑Workflow."""

    # --------------------------------------------------------------
    # Initialisierung → Daten laden, Modell selektieren, Vorhersage
    # --------------------------------------------------------------
    def __init__(self):
        self.df          = None   # gereinigter Datensatz
        self.model       = None   # bestes Poisson‑GLM
        self.grid        = None   # Vorhersageraster mit PI
        self._load_data()
        self._select_model()
        self._build_grid()
        self._calc_prediction_interval()
        self._calc_se_mean()

    # --------------------------------------------------------------
    # 1 – Daten laden & aufbereiten  (≈ R‑Abschnitt “load data”)
    # --------------------------------------------------------------
    def _load_data(self):
        data   = pd.read_excel(DATA_DIR / DATA_FILE).rename(columns=str.strip)
        design = (
            pd.read_excel(DATA_DIR / DESIGN_FILE, usecols=[0, 1, 2])
            .rename(columns=str.strip)
            .rename(columns={"Vorfahrtstrategie": "Vorfahrtsstrategie"})
        )

        # CCD‑Design anwenden
        rows = []
        for _, r in design.iterrows():
            sel = (
                (data["Systemlast"] == r["Systemlast"])
                & (data["Losgroesse"] == r["Losgroesse"])
                & (data["Vorfahrtsstrategie"] == r["Vorfahrtsstrategie"])
            )
            rows.append(data.loc[sel])
        df = pd.concat(rows, ignore_index=True)

        # Deadlocks entfernen und Spalten umbenennen
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

        # Kodierte Variablen (‑1 … 1) wie im R‑Code
        for col in ["systemload", "loadunitcap"]:
            mid         = (df[col].max() + df[col].min()) / 2
            half_range  = (df[col].max() - df[col].min()) / 2
            df[f"coded_{col}"] = (df[col] - mid) / half_range

        self.df = df

    # --------------------------------------------------------------
    # 2 – Modellselektion (AIC‐basiert, exhaustive Search)
    # --------------------------------------------------------------
    def _select_model(self):
        terms = [
            # Haupteffekte
            "systemload",
            "loadunitcap",
            "zoning",
            # Quadratische Terme
            "I(systemload**2)",
            "I(loadunitcap**2)",
            # Zweiwege‑Interaktionen
            "systemload:zoning",
            "loadunitcap:zoning",
            "systemload:loadunitcap",
        ]
        best_aic   = np.inf
        best_model = None
        for k in range(1, len(terms) + 1):
            for combo in itertools.combinations(terms, k):
                formula = "throughput ~ " + " + ".join(combo)
                m = glm(formula, data=self.df, family=sm.families.Poisson()).fit()
                if m.aic < best_aic:
                    best_aic, best_model = m.aic, m
        self.model = best_model

    # --------------------------------------------------------------
    # 3 – Vorhersageraster wie im R‑Code (Größe = n Beob.)
    # --------------------------------------------------------------
    def _build_grid(self):
        n = len(self.df)
        self.grid = pd.DataFrame(
            list(
                itertools.product(
                    self.df["zoning"].cat.categories,
                    np.linspace(self.df["systemload"].min(),  self.df["systemload"].max(),  n),
                    np.linspace(self.df["loadunitcap"].min(), self.df["loadunitcap"].max(), n),
                )
            ),
            columns=["zoning", "systemload", "loadunitcap"],
        )

    # --------------------------------------------------------------
    # 4 – Vorhersage + 95 % Prognoseintervall (Delta‑Methode)
    # --------------------------------------------------------------
    def _calc_prediction_interval(self):
        pred = self.model.get_prediction(self.grid)
        self.grid["prediction"] = pred.predicted_mean
        se_mean = pred.se_mean                      # Varianz der Erwartungswerte
        var_tot = se_mean**2 + self.grid["prediction"]   # + Poisson‑Varianz
        se_pred = np.sqrt(var_tot)
        self.grid["lower"] = (self.grid["prediction"] - Z_975 * se_pred).clip(lower=0)
        self.grid["upper"] =  self.grid["prediction"] + Z_975 * se_pred

    # --------------------------------------------------------------
    # 4b – Standardfehler des Erwartungswerts für FDS / VDG
    # --------------------------------------------------------------
    def _calc_se_mean(self):
        """Speichert den SE der Erwartungswerte im Grid (Design‑Space‑Varianz)."""
        pred = self.model.get_prediction(self.grid)
        self.grid["se_mean"] = pred.se_mean

    # --------------------------------------------------------------
    # Helfer: FDS‑Kurve  (sorted SE vs. Fraction)
    # --------------------------------------------------------------
    def build_fds_curve(self) -> go.Figure:
        v_sorted = np.sort(self.grid["se_mean"].values)
        frac = np.linspace(0, 1, len(v_sorted))
        fig = go.Figure(
            go.Scatter(x=v_sorted, y=frac, mode="lines", line=dict(width=2))
        )
        fig.update_layout(
            xaxis_title="Standardfehler des Erwartungswerts",
            yaxis_title="Fraction of Design Space",
            title="FDS‑Plot",
            margin=dict(l=40, r=20, t=40, b=40),
        )
        return fig

    # --------------------------------------------------------------
    # Helfer: VDG‑Heatmap (SE über Design‑Space)
    # --------------------------------------------------------------
    def build_vdg_heatmap(self) -> go.Figure:
        # Mehrere Zoning‑Kategorien erzeugen für jede (systemload, loadunitcap)‑Kombination
        # Duplikate.  → vorher mitteln, damit Pivot ohne Fehler funktioniert.
        df_vdg = (
            self.grid
            .groupby(["loadunitcap", "systemload"], as_index=False)["se_mean"]
            .mean()
        )
        hm = df_vdg.pivot(index="loadunitcap", columns="systemload", values="se_mean")
        fig = px.imshow(
            hm.values,
            x=hm.columns,
            y=hm.index,
            color_continuous_scale="Viridis",
            origin="lower",
            labels=dict(color="SE(mean)", x="Systemlast", y="Losgröße"),
        )
        fig.update_layout(
            title="Variance Dispersion Graph (VDG)",
            margin=dict(l=40, r=20, t=40, b=40),
        )
        return fig

    # --------------------------------------------------------------
    # Helfer: Verteilungs‑ & Diagnose‑Plots (Histogramm, Boxplot, Residuen)
    # --------------------------------------------------------------
    def build_distribution_plots(self):
        hist = px.histogram(self.df, x="throughput", nbins=20, title="Histogramm Throughput")
        box  = px.box(self.df, y="throughput", points="all", title="Boxplot Throughput")
        resid = self.model.resid_response
        fitted = self.model.fittedvalues
        resid_fig = px.scatter(
            x=fitted, y=resid, trendline="lowess",
            labels=dict(x="Fitted", y="Residual"),
            title="Residuals vs. Fitted"
        )
        return hist, box, resid_fig

    # --------------------------------------------------------------
    # Helfer: Korrelations‑Plot (Throughput vs. aopt, falls vorhanden)
    # --------------------------------------------------------------
    def build_corr_plot(self):
        if "aopt" not in self.df.columns:
            return None
        corr = self.df[["throughput", "aopt"]].corr()
        fig = px.imshow(
            corr.values,
            x=corr.columns, y=corr.columns,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1,
            title="Korrelationsmatrix"
        )
        return fig

    # --------------------------------------------------------------
    # 5 – Plot‑Funktionen (3‑D Surface wie plot_3D im R‑Code)
    # --------------------------------------------------------------
    @staticmethod
    def _hex_rgba(hex_color: str, alpha: float) -> str:
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"

    def build_figure(self, sub: pd.DataFrame, cfg: dict) -> go.Figure:
        # Raster in Matrixform
        P = sub.pivot(index="loadunitcap", columns="systemload", values="prediction").values
        L = sub.pivot(index="loadunitcap", columns="systemload", values="lower").values
        U = sub.pivot(index="loadunitcap", columns="systemload", values="upper").values
        X, Y = np.meshgrid(
            sorted(sub["systemload"].unique()),
            sorted(sub["loadunitcap"].unique()),
        )

        fig = go.Figure()

        # Mittelfläche
        fig.add_surface(
            x=X, y=Y, z=P,
            surfacecolor=np.zeros_like(P),
            colorscale=[[0, cfg["pred_color"]], [1, cfg["pred_color"]]],
            showscale=False,
            opacity=1,
        )

        # Volumen (PI)
        if cfg["show_fill"]:
            rgba = self._hex_rgba(cfg["fill_color"], cfg["fill_alpha"])
            for t in np.linspace(0, 1, cfg["layers"] + 2)[1:-1]:
                Z = L + t * (U - L)
                fig.add_surface(
                    x=X, y=Y, z=Z,
                    surfacecolor=np.zeros_like(Z),
                    colorscale=[[0, rgba], [1, rgba]],
                    showscale=False,
                    opacity=cfg["fill_alpha"],
                )

        # Deckel
        for Z, col in [(U, cfg["upper_color"]), (L, cfg["lower_color"])]:
            fig.add_surface(
                x=X, y=Y, z=Z,
                surfacecolor=np.zeros_like(Z),
                colorscale=[[0, col], [1, col]],
                showscale=False,
                opacity=cfg["deckel_alpha"],
            )

        # Seitenwände
        if cfg["show_walls"]:
            rgba = self._hex_rgba(cfg["fill_color"], cfg["wall_alpha"])
            for i in [0, -1]:  # vorne / hinten
                fig.add_surface(
                    x=np.vstack([X[i], X[i]]),
                    y=np.vstack([Y[i], Y[i]]),
                    z=np.vstack([L[i], U[i]]),
                    surfacecolor=np.zeros_like(np.vstack([L[i], U[i]])),
                    colorscale=[[0, rgba], [1, rgba]],
                    showscale=False,
                    opacity=cfg["wall_alpha"],
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
                )

        # --- Optional: einzelne Vorhersagepunkte & Linien verbinden -------------
        if cfg.get("show_points"):
            fig.add_trace(
                go.Scatter3d(
                    x=sub["systemload"],
                    y=sub["loadunitcap"],
                    z=sub["prediction"],
                    mode="markers",
                    marker=dict(size=3, color=cfg.get("points_color", "#FFFFFF")),
                    name="Predictions",
                    showlegend=False,
                )
            )

        if cfg.get("show_lines"):
            # Verbinde Punkte gleichen loadunitcap-Werts der Reihe nach über systemload
            for lu in sorted(sub["loadunitcap"].unique()):
                dat = sub[sub["loadunitcap"] == lu].sort_values("systemload")
                fig.add_trace(
                    go.Scatter3d(
                        x=dat["systemload"],
                        y=dat["loadunitcap"],
                        z=dat["prediction"],
                        mode="lines",
                        line=dict(color=cfg.get("lines_color", "#FFFFFF"), width=2),
                        showlegend=False,
                    )
                )

        # Achsen & Layout
        fig.update_layout(
            margin=dict(l=5, r=5, t=35, b=5),
            scene=dict(
                xaxis_title="Systemlast",
                yaxis_title="Losgröße",
                zaxis_title="Durchsatz",
                aspectratio=dict(x=1, y=1, z=0.6),
                camera=dict(eye=dict(x=1.4, y=1.4, z=0.85)),
            ),
        )
        return fig


# ------------------------------------------------------------------
# Streamlit UI (entspricht der R‑Visualisierung “Graphics”)
# ------------------------------------------------------------------
def main():
    app = ThroughputApp()

    st.set_page_config(page_title="Kennlinien‑Viewer", layout="wide")
    st.title("3‑D‑Kennlinien mit 95 % Prognoseintervall")

    # Sidebar – Anzeigeoptionen
    st.sidebar.header("Anzeige‑Einstellungen")
    cfg = dict(
        pred_color   = st.sidebar.color_picker("Farbe Mittelfläche",  "#FFB400"),
        upper_color  = st.sidebar.color_picker("Farbe oberer Deckel", "#282828"),
        lower_color  = st.sidebar.color_picker("Farbe unterer Deckel", "#282828"),
        fill_color   = st.sidebar.color_picker("Farbe Volumen",       "#3C3C3C"),
        fill_alpha   = st.sidebar.slider("Volumen‑Transparenz", 0.05, 0.9,  0.18, 0.01),
        deckel_alpha = st.sidebar.slider("Deckel‑Transparenz",  0.1,  1.0,  0.45, 0.05),
        show_fill    = st.sidebar.checkbox("Volumenfüllung anzeigen", True),
        show_walls   = st.sidebar.checkbox("Seitliche Wände anzeigen", True),
        wall_alpha   = st.sidebar.slider("Wand‑Transparenz",    0.05, 1.0,  0.25, 0.05),
        layers       = st.sidebar.slider("Füll‑Schichten (Dichte)", 4, 40, 20, 2),
        show_points  = st.sidebar.checkbox("Vorhersage‑Punkte anzeigen", False),
        points_color = st.sidebar.color_picker("Farbe Punkte", "#FFFFFF"),
        show_lines   = st.sidebar.checkbox("Vorhersage‑Linien verbinden", False),
        lines_color  = st.sidebar.color_picker("Farbe Linien", "#FFFFFF"),
    )

    # Auswahl der Zoning‑Strategien
    zones      = app.df["zoning"].cat.categories.tolist()
    sel_zones  = st.multiselect("Zoning‑Strategien:", zones, default=zones)

    tab_plot, tab_design, tab_diag = st.tabs(["3‑D‑Plot", "Design‑Space FDS/VDG", "Diagnostik"])

    # --- Tab 1: 3‑D‑Plots --------------------------------------------------
    with tab_plot:
        cols = st.columns(max(len(sel_zones), 1))
        for col, z in zip(cols, sel_zones):
            fig = app.build_figure(app.grid[app.grid["zoning"] == z], cfg)
            col.subheader(f"{z} – 95 % Prognoseintervall")
            col.plotly_chart(fig, use_container_width=True)

    # --- Tab 2: Design‑Space‑Analyse (FDS & VDG) ---------------------------
    with tab_design:
        st.plotly_chart(app.build_fds_curve(), use_container_width=True)
        st.plotly_chart(app.build_vdg_heatmap(), use_container_width=True)

    # --- Tab 3: Verteilungs‑ & Modelldiag. ---------------------------------
    with tab_diag:
        hist_fig, box_fig, resid_fig = app.build_distribution_plots()
        st.plotly_chart(hist_fig, use_container_width=True)
        st.plotly_chart(box_fig,  use_container_width=True)
        st.plotly_chart(resid_fig, use_container_width=True)
        corr_fig = app.build_corr_plot()
        if corr_fig:
            st.plotly_chart(corr_fig, use_container_width=True)


if __name__ == "__main__":
    main()
