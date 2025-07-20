# 📊 Streamlit Apps – Quick-Deploy Repository

Dieses Repository dient als Sammelstelle für **kleine, eigenständige Streamlit-Apps**, die du ohne Aufwand lokal testen oder direkt über **Streamlit Community Cloud** veröffentlichen kannst.

---

## 🗂️ Ordnerstruktur

```
streamlit-apps/
├─ app1/
│  ├─ app.py            # Haupt-Script
│  └─ requirements.txt   # (optional) spez. Abh.
├─ app2/
│  └─ app.py
└─ README.md
```

* Jede Untermappe enthält **eine** App (`app.py`).
* Eigene Python-Libs, Daten­dateien etc. einfach daneben legen.
* Benötigt eine App zusätzliche Packages, lege dort eine `requirements.txt` ab; sonst greift das globale.

---

## 🚀 Lokal starten

```bash
# Repo klonen
git clone https://github.com/<user>/streamlit-apps.git
cd streamlit-apps/app1          # gewünschte App

# Abhängigkeiten installieren
pip install -r ../requirements.txt    # global
pip install -r requirements.txt       # (falls vorhanden)

# App ausführen
streamlit run app.py
```

---

## ☁️ Deploy auf Streamlit Cloud

1. Repository forken oder in ein eigenes Repo kopieren.  
2. Auf **https://streamlit.io/cloud** ein neues Projekt anlegen.  
3. Als **Main file path** z. B. `app1/app.py` angeben.  
4. Optional: unter *Advanced settings* eine app‑spezifische `requirements.txt` wählen.

> **Tipp:** Wenn du mehrere Apps deployen willst, leg für jede App ein separates Cloud‑Projekt an und setze dort den entsprechenden Unterpfad.

---

## 📦 Globale Abhängigkeiten

Alle Apps teilen sich diese Packages (Versionen bei Bedarf anpassen):

```text
streamlit>=1.34
plotly>=5.20
pandas>=2.0
numpy>=1.25
statsmodels>=0.14
openpyxl>=3.1
```

Apps mit speziellen Libraries legen eine eigene `requirements.txt` in ihrem Ordner ab – diese wird von Streamlit Cloud automatisch erkannt und zusammen mit den globalen Paketen installiert.

---

## 🤝 Beitragen

1. **Branch** erstellen:  
   ```bash
   git checkout -b feature/my-new-app
   ```
2. Neuen Ordner `my-new-app/` anlegen, `app.py` + ggf. Daten & `requirements.txt` hinzufügen.  
3. Pull‑Request öffnen – fertig!

---

Happy Streamliting 🚀
