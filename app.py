import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import hdbscan
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import umap
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

try:
    from bertopic import BERTopic
except Exception:  # BERTopic optional fallback
    BERTopic = None

st.set_page_config(page_title="Message Topic Analyzer", layout="wide")

REQUIRED_COLS = ["user_id", "prompt_text"]


@dataclass
class AnalysisConfig:
    embedding_model: str
    umap_neighbors: int
    umap_components: int
    lower_case: bool
    remove_empty: bool
    use_bertopic: bool


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    return SentenceTransformer(model_name)


def normalize_whitespace(text: str) -> str:
    return " ".join(str(text).split())


def ensure_columns(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    renamed = df.rename(columns={v: k for k, v in mapping.items() if v})
    for col in REQUIRED_COLS:
        if col not in renamed.columns:
            raise ValueError(f"Missing required mapped column: {col}")
    return renamed


def clean_dataframe(df: pd.DataFrame, prompt_col: str, lower_case: bool, remove_empty: bool) -> pd.DataFrame:
    out = df.copy()
    out["prompt_text_original"] = out[prompt_col].astype(str)
    out["prompt_text_clean"] = out[prompt_col].fillna("").astype(str).map(normalize_whitespace)
    if lower_case:
        out["prompt_text_clean"] = out["prompt_text_clean"].str.lower()
    if remove_empty:
        out = out[out["prompt_text_clean"].str.len() > 0]
    return out.reset_index(drop=True)


def compute_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    model = load_model(model_name)
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def cluster_texts(texts: List[str], config: AnalysisConfig, min_cluster_size: int) -> Tuple[np.ndarray, np.ndarray]:
    emb = compute_embeddings(texts, config.embedding_model)

    reducer = umap.UMAP(
        n_neighbors=min(config.umap_neighbors, max(2, len(texts) - 1)),
        n_components=min(config.umap_components, 2),
        metric="cosine",
        random_state=42,
    )
    emb_umap = reducer.fit_transform(emb)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="euclidean",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(emb_umap)

    if config.use_bertopic and BERTopic is not None:
        try:
            topic_model = BERTopic(
                embedding_model=load_model(config.embedding_model),
                umap_model=reducer,
                hdbscan_model=clusterer,
                language="multilingual",
                calculate_probabilities=False,
                verbose=False,
            )
            bt_labels, _ = topic_model.fit_transform(texts)
            labels = np.array(bt_labels)
        except Exception:
            pass

    return labels, emb_umap


def cluster_all_prompts(df: pd.DataFrame, config: AnalysisConfig, min_cluster_size: int = 2) -> pd.DataFrame:
    out = df.copy()
    if len(out) < min_cluster_size:
        out["cluster_id"] = -1
        out["umap_x"] = 0.0
        out["umap_y"] = 0.0
        return out

    labels, emb_umap = cluster_texts(out["prompt_text_clean"].tolist(), config, min_cluster_size=min_cluster_size)
    out["cluster_id"] = labels
    out["umap_x"] = emb_umap[:, 0]
    out["umap_y"] = emb_umap[:, 1] if emb_umap.shape[1] > 1 else 0.0
    return out


def build_ctfidf_representatives(
    df: pd.DataFrame,
    cluster_col: str,
    text_col: str,
    top_messages: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    valid = df[df[cluster_col] >= 0].copy()
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame()

    docs = valid.groupby(cluster_col)[text_col].apply(lambda x: " ".join(x.tolist())).sort_index()
    vectorizer = CountVectorizer(max_features=4000)
    counts = vectorizer.fit_transform(docs.values)
    tfidf = TfidfTransformer(norm="l2", use_idf=True).fit_transform(counts)

    terms = np.array(vectorizer.get_feature_names_out())
    topic_rows = []
    rep_rows = []

    valid = valid.reset_index(drop=True)
    msg_counts = vectorizer.transform(valid[text_col])
    cluster_to_idx = {cid: i for i, cid in enumerate(docs.index.tolist())}

    for cid, doc_text in docs.items():
        row_idx = cluster_to_idx[cid]
        weights = tfidf[row_idx].toarray().ravel()
        top_term_idx = weights.argsort()[::-1][:10]
        top_terms = ", ".join(terms[top_term_idx])

        sub = valid[valid[cluster_col] == cid].copy()
        sub_pos = sub.index.to_numpy()
        sub_counts = msg_counts[sub_pos]
        term_weights = weights.reshape(-1, 1)
        scores = (sub_counts @ term_weights).ravel()
        sub["rep_score"] = scores
        top_msgs = sub.sort_values("rep_score", ascending=False).head(top_messages)

        topic_rows.append(
            {
                cluster_col: int(cid),
                "n_messages": int(len(sub)),
                "top_terms": top_terms,
                "cluster_document": doc_text,
            }
        )

        for rank, r in enumerate(top_msgs.itertuples(), start=1):
            rep_rows.append(
                {
                    cluster_col: int(cid),
                    "rank": rank,
                    "representative_message": r.prompt_text_original,
                    "representative_message_clean": r.prompt_text_clean,
                    "rep_score": float(r.rep_score),
                    "user_id": r.user_id,
                }
            )

    return pd.DataFrame(topic_rows), pd.DataFrame(rep_rows)


def meta_cluster_representatives(reps: pd.DataFrame, config: AnalysisConfig, min_cluster_size: int = 5) -> pd.DataFrame:
    if reps.empty:
        return reps.assign(meta_cluster=-1, meta_umap_x=0.0, meta_umap_y=0.0)

    cluster_level = (
        reps.sort_values("rank")
        .groupby("cluster_id", as_index=False)
        .first()[["cluster_id", "representative_message_clean", "representative_message"]]
    )

    if len(cluster_level) < min_cluster_size:
        cluster_level["meta_cluster"] = -1
        cluster_level["meta_umap_x"] = 0.0
        cluster_level["meta_umap_y"] = 0.0
        return cluster_level

    labels, emb_umap = cluster_texts(
        cluster_level["representative_message_clean"].tolist(),
        config,
        min_cluster_size=min_cluster_size,
    )
    cluster_level["meta_cluster"] = labels
    cluster_level["meta_umap_x"] = emb_umap[:, 0]
    cluster_level["meta_umap_y"] = emb_umap[:, 1] if emb_umap.shape[1] > 1 else 0.0
    return cluster_level


def export_df_button(df: pd.DataFrame, label: str, filename: str):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv")


st.title("Message Topic Analyzer")
st.caption("BERT-basierte Clusteranalyse aller User-Prompts (inkl. Meta-Cluster)")

with st.sidebar:
    st.header("Konfiguration")
    embedding_model = st.text_input("Embedding-Modell", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    umap_neighbors = st.slider("UMAP n_neighbors", 2, 100, 15)
    umap_components = st.slider("UMAP n_components", 2, 10, 2)
    lower_case = st.checkbox("Kleinschreibung", value=False)
    remove_empty = st.checkbox("Leere Nachrichten entfernen", value=True)
    use_bertopic = st.checkbox("BERTopic verwenden (wenn verfügbar)", value=True)

config = AnalysisConfig(
    embedding_model=embedding_model,
    umap_neighbors=umap_neighbors,
    umap_components=umap_components,
    lower_case=lower_case,
    remove_empty=remove_empty,
    use_bertopic=use_bertopic,
)

st.header("1) Daten hochladen")
uploaded = st.file_uploader("CSV auswählen", type=["csv"])

if uploaded is not None:
    raw = pd.read_csv(uploaded)
    st.subheader("2) Daten prüfen & Spalten zuordnen")
    st.dataframe(raw.head(20), use_container_width=True)

    mapping = {}
    cols = [""] + raw.columns.tolist()
    for required in REQUIRED_COLS:
        default = required if required in raw.columns else ""
        mapping[required] = st.selectbox(f"Spalte für {required}", cols, index=cols.index(default) if default in cols else 0)

    if st.button("Analyse starten", type="primary"):
        try:
            df = ensure_columns(raw, mapping)
            clean = clean_dataframe(df, "prompt_text", config.lower_case, config.remove_empty)
            if clean.empty:
                st.error("Nach Bereinigung sind keine Nachrichten mehr übrig.")
                st.stop()

            clustered = cluster_all_prompts(clean, config, min_cluster_size=2)

            cluster_summary, reps = build_ctfidf_representatives(
                clustered,
                cluster_col="cluster_id",
                text_col="prompt_text_clean",
                top_messages=3,
            )

            meta_clusters = meta_cluster_representatives(reps, config, min_cluster_size=5)
            reps_meta = reps.merge(meta_clusters[["cluster_id", "meta_cluster", "meta_umap_x", "meta_umap_y"]], on="cluster_id", how="left")
            reps_meta["meta_cluster"] = reps_meta["meta_cluster"].fillna(-1).astype(int)

            meta_summary, meta_reps = build_ctfidf_representatives(
                reps_meta.rename(columns={"representative_message_clean": "prompt_text_clean", "representative_message": "prompt_text_original"}),
                cluster_col="meta_cluster",
                text_col="prompt_text_clean",
                top_messages=1,
            )

            st.session_state["clean"] = clean
            st.session_state["first_messages"] = first_messages
            st.session_state["clustered"] = clustered
            st.session_state["cluster_summary"] = cluster_summary
            st.session_state["reps"] = reps_meta
            st.session_state["meta_summary"] = meta_summary
            st.session_state["meta_reps"] = meta_reps
            st.session_state["meta_labels"] = {
                int(r.meta_cluster): f"Meta-Cluster {int(r.meta_cluster)}" for r in meta_summary.itertuples()
            }
            st.session_state["superficial_labels"] = {
                int(r.meta_cluster): "nicht-superficial" for r in meta_summary.itertuples()
            }
            st.success("Analyse abgeschlossen.")
        except Exception as e:
            st.exception(e)

if "clustered" in st.session_state:
    clustered = st.session_state["clustered"]
    cluster_summary = st.session_state["cluster_summary"]
    reps = st.session_state["reps"]
    meta_summary = st.session_state["meta_summary"]
    meta_reps = st.session_state["meta_reps"]

    st.header("3) Cluster (alle Prompts)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Anzahl User", int(clustered["user_id"].nunique()))
    c2.metric("Anzahl Prompts", int(len(clustered)))
    c3.metric("Anzahl Cluster", int((cluster_summary["cluster_id"] >= 0).sum()) if not cluster_summary.empty else 0)

    fig_cluster = px.scatter(
        clustered,
        x="umap_x",
        y="umap_y",
        color=clustered["cluster_id"].astype(str),
        hover_data=["user_id", "prompt_text_original", "cluster_id"],
        title="UMAP-Cluster aller Prompts",
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.dataframe(cluster_summary.sort_values("n_messages", ascending=False), use_container_width=True)
    st.subheader("Top-3 repräsentative Nachrichten je Cluster")
    st.dataframe(reps[["cluster_id", "meta_cluster", "rank", "representative_message", "user_id"]], use_container_width=True)

    st.header("4) Meta-Cluster")
    fig_meta = px.scatter(
        reps.drop_duplicates("cluster_id"),
        x="meta_umap_x",
        y="meta_umap_y",
        color=reps.drop_duplicates("cluster_id")["meta_cluster"].astype(str),
        hover_data=["cluster_id", "representative_message"],
        title="Meta-Clustering auf Cluster-Repräsentanten",
    )
    st.plotly_chart(fig_meta, use_container_width=True)

    editable_meta = meta_summary.copy()
    meta_labels = st.session_state.get("meta_labels", {})
    superficial_labels = st.session_state.get("superficial_labels", {})
    editable_meta["message_type"] = editable_meta["meta_cluster"].map(lambda x: meta_labels.get(int(x), f"Meta-Cluster {int(x)}"))
    editable_meta["superficial_flag"] = editable_meta["meta_cluster"].map(lambda x: superficial_labels.get(int(x), "nicht-superficial"))
    edited_meta = st.data_editor(editable_meta, use_container_width=True, num_rows="fixed")

    st.session_state["meta_labels"] = {int(r["meta_cluster"]): r["message_type"] for _, r in edited_meta.iterrows()}
    st.session_state["superficial_labels"] = {int(r["meta_cluster"]): r["superficial_flag"] for _, r in edited_meta.iterrows()}

    meta_top = meta_reps.copy()
    meta_top["message_type"] = meta_top["meta_cluster"].map(st.session_state["meta_labels"])
    meta_top["superficial_flag"] = meta_top["meta_cluster"].map(st.session_state["superficial_labels"])
    st.dataframe(meta_top[["meta_cluster", "representative_message", "message_type", "superficial_flag"]], use_container_width=True)

    st.header("5) Export")
    export_df_button(clustered, "Prompts mit Clusterzuordnung (CSV)", "prompts_with_clusters.csv")
    export_df_button(cluster_summary, "Cluster-Summary (CSV)", "cluster_summary.csv")
    export_df_button(reps, "Cluster-Repräsentanten (CSV)", "cluster_representatives.csv")
    export_df_button(meta_summary, "Meta-Cluster-Summary (CSV)", "meta_cluster_summary.csv")
    export_df_button(meta_top, "Meta-Repräsentanten (CSV)", "meta_representatives.csv")

    export_payload = {
        "cluster_summary": cluster_summary.to_dict(orient="records"),
        "cluster_representatives": reps.to_dict(orient="records"),
        "meta_cluster_summary": meta_summary.to_dict(orient="records"),
        "meta_representatives": meta_top.to_dict(orient="records"),
        "meta_labels": st.session_state["meta_labels"],
        "superficial_labels": st.session_state["superficial_labels"],
    }
    st.download_button(
        "Gesamtergebnis als JSON",
        data=json.dumps(export_payload, ensure_ascii=False, indent=2),
        file_name="message_topic_analysis.json",
        mime="application/json",
    )
else:
    st.info("Bitte eine CSV hochladen und die Analyse starten.")
