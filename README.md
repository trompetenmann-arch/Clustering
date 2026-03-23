# Prompt Profile Analyzer

Lokale, browserbasierte Streamlit-App zur Analyse von Prompt-Daten aus CSV-Dateien.

## Features

- CSV-Upload (Dateiauswahl) mit Datenvorschau
- Flexible Spaltenzuordnung (`user_id`, `timestamp`, `prompt_text` + optionale Felder)
- Datenbereinigung (Whitespace, optional lower-case, Mindestlänge)
- Semantisches Clustering einzelner Prompts mit:
  - `sentence-transformers` Embeddings
  - `UMAP` Reduktion
  - `HDBSCAN` Clustering
  - optional BERTopic-Workflow mit Fallback
- User-Profiling auf Basis von Clusterverteilungen:
  - absolute/relative Clusterhäufigkeiten
  - dominante Cluster
  - Entropie/Diversität
  - durchschnittliche Promptlänge
- Optionales zweites Clustering auf User-Ebene (User-Typen)
- Interaktive Visualisierungen mit Plotly
- Manuelles Labeling der Cluster
- Export als CSV und JSON

## Projektstruktur

- `app.py` – Streamlit Anwendung
- `requirements.txt` – Python-Abhängigkeiten
- `sample_data.csv` – Beispieldaten

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Start

```bash
streamlit run app.py
```

Die App läuft dann lokal (meist unter `http://localhost:8501`).

## Datenformat

Mindestens diese Spalten werden benötigt:

- `user_id`
- `timestamp`
- `prompt_text`

Optional:

- `session`
- `treatment_arm`
- `grade`
- `problem_id`
- `response_text`
- `conversation_id`

## Methodik in Kurzform

1. Einzelprompts werden bereinigt und eingebettet.
2. Prompts werden mit UMAP+HDBSCAN semantisch geclustert (optional BERTopic).
3. Aus Prompt-Clustern wird pro User eine Clusterverteilung abgeleitet.
4. Daraus entstehen datengetriebene User-Profile und optional User-Typen.

## Hinweise

- Für deutschsprachige Daten ist standardmäßig ein multilinguales Modell voreingestellt.
- Bei sehr großen Datensätzen kann die Berechnung mehrere Minuten dauern.
- Falls BERTopic in der Laufzeitumgebung nicht korrekt verfügbar ist, nutzt die App automatisch den Fallback-Workflow.
