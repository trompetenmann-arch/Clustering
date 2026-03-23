# Message Topic Analyzer

Lokale Streamlit-App zur **zweistufigen BERT-basierten Topic-Clustering-Analyse** von Schülernachrichten.

## Was wurde umgesetzt?

Die App implementiert jetzt das beschriebene Vorgehen:

1. **First-message Selektion je Schüler und Problem**
   - Pro Kombination aus `session`, `treatment_arm`, `grade`, `problem_id`, `user_id` wird nur die **erste Nachricht** verwendet.
2. **Problem-spezifisches Clustering**
   - Sentence-BERT Embeddings
   - UMAP zur Dimensionsreduktion
   - HDBSCAN (mind. 2 Nachrichten pro Cluster)
   - optional BERTopic (falls verfügbar)
3. **c-TF-IDF je Cluster**
   - Jede Cluster-Nachrichtenmenge wird als ein Dokument behandelt.
   - Top-Terme und **Top-3 repräsentative Nachrichten** werden extrahiert.
4. **Meta-Clustering**
   - Repräsentative Cluster-Nachrichten werden problemübergreifend erneut geclustert.
   - Mindestgröße: 5 Cluster pro Meta-Cluster.
   - c-TF-IDF und **Top-1 Meta-repräsentative Nachricht**.
5. **Manuelles Labeling**
   - Meta-Cluster können im UI mit `message_type` gelabelt werden.
   - Zusätzlich `superficial_flag` (z. B. superficial / nicht-superficial).
6. **Session-Visualisierung**
   - Anteil erster Nachrichten nach superficial-Kategorie je Session/Treatment.

## Features

- CSV-Upload + flexible Spaltenzuordnung
- Interaktive Plotly-Visualisierungen (Problem-Cluster und Meta-Cluster)
- Editierbares Meta-Cluster-Labeling
- CSV- und JSON-Export aller Stufen

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

## Datenformat

Erforderlich:

- `user_id`
- `timestamp`
- `prompt_text`

Optional (empfohlen für die beschriebene Methodik):

- `session`
- `treatment_arm`
- `grade`
- `problem_id`
- `response_text`
- `conversation_id`

## Hinweis

Falls BERTopic in der Umgebung nicht verfügbar ist, läuft automatisch der UMAP+HDBSCAN-Workflow weiter.
