# Message Topic Analyzer

Lokale Streamlit-App zur **BERT-basierten Clusteranalyse von User-Prompts**.

## Fokus der App

Die App ist jetzt bewusst reduziert auf CSV-Dateien mit nur zwei relevanten Feldern:

- `user_id`
- `prompt_text`

Alle Prompts eines Users werden berücksichtigt (keine Problem-/Session-Filterung erforderlich).

## Workflow

1. Bereinigung der Prompt-Texte (Whitespace, optional lowercase, optional Entfernen leerer Prompts)
2. Clustering über **alle Prompts** mit Sentence-BERT + UMAP + HDBSCAN (optional BERTopic)
3. c-TF-IDF-Auswertung pro Cluster mit Top-Termen und Top-3 repräsentativen Nachrichten
4. Optionales Meta-Clustering der Cluster-Repräsentanten
5. Manuelles Labeling der Meta-Cluster (`message_type`, `superficial_flag`)
6. CSV/JSON-Export

## Features

- CSV-Upload + einfache Spaltenzuordnung nur für `user_id` und `prompt_text`
- Interaktive UMAP-Visualisierungen für Cluster und Meta-Cluster
- Repräsentative Nachrichten pro Cluster
- Editierbares Meta-Labeling
- Export aller zentralen Ergebnistabellen

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

Danach läuft die App lokal (typisch `http://localhost:8501`).

## Hinweis

Falls BERTopic in der Umgebung nicht verfügbar ist, läuft automatisch der UMAP+HDBSCAN-Workflow weiter.
