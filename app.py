import io
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import umap
import hdbscan

try:
    from bertopic import BERTopic
except Exception:  # BERTopic optional fallback
    BERTopic = None


st.set_page_config(page_title="Prompt Profile Analyzer", layout="wide")

REQUIRED_COLS = ["user_id", "timestamp", "prompt_text"]
OPTIONAL_COLS = ["session", "treatment_arm", "grade", "problem_id", "response_text", "conversation_id"]


@dataclass
class AnalysisConfig:
    embedding_model: str
    umap_neighbors: int
    umap_components: int
    hdbscan_min_cluster_size: int
    min_prompt_length: int
    lower_case: bool
    remove_short: bool
    use_bertopic: bool
    aggregation_mode: str


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    return SentenceTransformer(model_name)


def normalize_whitespace(text: str) -> str:
    return " ".join(str(text).split())


def clean_dataframe(
    df: pd.DataFrame,
    prompt_col: str,
    lower_case: bool,
    min_len: int,
    remove_short: bool,
) -> pd.DataFrame:
    out = df.copy()
    out["prompt_text_original"] = out[prompt_col].astype(str)
    out["prompt_text_clean"] = out[prompt_col].fillna("").astype(str).map(normalize_whitespace)
    if lower_case:
        out["prompt_text_clean"] = out["prompt_text_clean"].str.lower()
    out = out[out["prompt_text_clean"].str.len() > 0]
    if remove_short:
        out = out[out["prompt_text_clean"].str.len() >= min_len]
    return out.reset_index(drop=True)


def aggregate_docs(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "user":
        group_cols = ["user_id"]
    elif mode == "user+session" and "session" in df.columns:
        group_cols = ["user_id", "session"]
    elif mode == "user+problem" and "problem_id" in df.columns:
        group_cols = ["user_id", "problem_id"]
    else:
        group_cols = ["user_id"]

    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            prompt_count=("prompt_text_clean", "size"),
            agg_text=("prompt_text_clean", lambda x: " ".join(x.tolist())),
            avg_prompt_len=("prompt_text_clean", lambda x: float(np.mean([len(i) for i in x]))),
        )
        .reset_index()
    )
    return agg


def compute_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    model = load_model(model_name)
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def prompt_clustering(
    df: pd.DataFrame,
    config: AnalysisConfig,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Optional[object]]:
    texts = df["prompt_text_clean"].tolist()
    emb = compute_embeddings(texts, config.embedding_model)

    reducer = umap.UMAP(
        n_neighbors=config.umap_neighbors,
        n_components=config.umap_components,
        metric="cosine",
        random_state=42,
    )
    emb_umap = reducer.fit_transform(emb)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.hdbscan_min_cluster_size,
        metric="euclidean",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(emb_umap)

    topic_model = None
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
            topic_model = None

    out = df.copy()
    out["cluster_id"] = labels
    out["umap_x"] = emb_umap[:, 0]
    out["umap_y"] = emb_umap[:, 1] if emb_umap.shape[1] > 1 else 0.0
    return out, emb, emb_umap, topic_model


def extract_top_terms(cluster_df: pd.DataFrame, n_terms: int = 8) -> str:
    if cluster_df.empty:
        return ""
    vectorizer = CountVectorizer(stop_words=None, max_features=2000)
    X = vectorizer.fit_transform(cluster_df["prompt_text_clean"])
    counts = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vectorizer.get_feature_names_out())
    top_idx = counts.argsort()[::-1][:n_terms]
    return ", ".join(terms[top_idx])


def build_cluster_summary(df_clustered: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, grp in df_clustered.groupby("cluster_id"):
        examples = grp["prompt_text_original"].head(3).tolist()
        rows.append(
            {
                "cluster_id": int(cid),
                "n_prompts": int(len(grp)),
                "top_terms": extract_top_terms(grp),
                "examples": " || ".join(examples),
            }
        )
    return pd.DataFrame(rows).sort_values("n_prompts", ascending=False)


def derive_user_profiles(df_clustered: pd.DataFrame, labels_map: Dict[int, str]) -> pd.DataFrame:
    valid = df_clustered[df_clustered["cluster_id"] >= 0].copy()
    counts = (
        valid.groupby(["user_id", "cluster_id"]).size().rename("count").reset_index()
    )
    totals = valid.groupby("user_id").size().rename("total_prompts")
    profile = counts.merge(totals, on="user_id")
    profile["share"] = profile["count"] / profile["total_prompts"]

    entropy_df = profile.groupby("user_id")["share"].apply(lambda x: float(-(x * np.log2(x + 1e-12)).sum()))
    diversity_df = counts.groupby("user_id")["cluster_id"].nunique().rename("n_cluster_types")
    avg_len = (
        df_clustered.groupby("user_id")["prompt_text_clean"]
        .apply(lambda x: float(np.mean([len(t) for t in x])))
        .rename("avg_prompt_len")
    )

    pivot = profile.pivot_table(index="user_id", columns="cluster_id", values="share", fill_value=0.0)
    dom = pivot.idxmax(axis=1).rename("dominant_cluster")

    out = pivot.reset_index().merge(dom, on="user_id").merge(entropy_df.rename("entropy"), on="user_id")
    out = out.merge(diversity_df, on="user_id").merge(avg_len, on="user_id")
    out["dominant_cluster_label"] = out["dominant_cluster"].map(lambda x: labels_map.get(int(x), f"Cluster {x}"))
    return out


def cluster_users(user_profiles: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    feature_cols = [c for c in user_profiles.columns if isinstance(c, (int, np.integer))]
    if len(feature_cols) == 0 or len(user_profiles) < 3:
        user_profiles["user_type_id"] = -1
        user_profiles["user_type_label"] = "Insufficient users"
        return user_profiles

    X = user_profiles[feature_cols].values
    Xs = StandardScaler().fit_transform(X)
    k = min(6, max(2, int(np.sqrt(len(user_profiles)))))
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = km.fit_predict(Xs)
    out = user_profiles.copy()
    out["user_type_id"] = labels

    names = {}
    for t, grp in out.groupby("user_type_id"):
        center = grp[feature_cols].mean().sort_values(ascending=False)
        topc = int(center.index[0]) if len(center) else -1
        names[t] = f"Type {t}: dominant Cluster {topc}"
    out["user_type_label"] = out["user_type_id"].map(names)
    return out


def ensure_columns(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    renamed = df.rename(columns={v: k for k, v in mapping.items() if v})
    for col in REQUIRED_COLS:
        if col not in renamed.columns:
            raise ValueError(f"Missing required mapped column: {col}")
    return renamed


def export_df_button(df: pd.DataFrame, label: str, filename: str):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv")


st.title("Prompt Profile Analyzer")
st.caption("CSV-basierte semantische Prompt-Analyse, Clusterbildung und User-Profiling")

with st.sidebar:
    st.header("Clustering konfigurieren")
    embedding_model = st.text_input("Embedding-Modell", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    umap_neighbors = st.slider("UMAP n_neighbors", 2, 100, 15)
    umap_components = st.slider("UMAP n_components", 2, 10, 2)
    hdbscan_min_cluster_size = st.slider("HDBSCAN min_cluster_size", 2, 100, 8)
    min_prompt_length = st.slider("Mindestlänge Prompt", 0, 500, 5)
    lower_case = st.checkbox("Kleinschreibung", value=False)
    remove_short = st.checkbox("Sehr kurze Prompts entfernen", value=True)
    use_bertopic = st.checkbox("BERTopic verwenden (wenn verfügbar)", value=False)
    aggregation_mode = st.selectbox("Aggregationsebene", ["user", "user+session", "user+problem"])

config = AnalysisConfig(
    embedding_model=embedding_model,
    umap_neighbors=umap_neighbors,
    umap_components=umap_components,
    hdbscan_min_cluster_size=hdbscan_min_cluster_size,
    min_prompt_length=min_prompt_length,
    lower_case=lower_case,
    remove_short=remove_short,
    use_bertopic=use_bertopic,
    aggregation_mode=aggregation_mode,
)

st.header("1) Daten hochladen")
uploaded = st.file_uploader("CSV auswählen", type=["csv"])

if uploaded is not None:
    raw = pd.read_csv(uploaded)
    st.subheader("2) Daten prüfen & Spalten zuordnen")
    st.dataframe(raw.head(20), use_container_width=True)

    mapping = {}
    cols = [""] + raw.columns.tolist()
    for required in REQUIRED_COLS + OPTIONAL_COLS:
        default = required if required in raw.columns else ""
        mapping[required] = st.selectbox(f"Spalte für {required}", cols, index=cols.index(default) if default in cols else 0)

    if st.button("Analyse starten", type="primary"):
        try:
            df = ensure_columns(raw, mapping)
            for c in OPTIONAL_COLS:
                if c not in df.columns:
                    df[c] = np.nan

            clean = clean_dataframe(df, "prompt_text", config.lower_case, config.min_prompt_length, config.remove_short)
            if clean.empty:
                st.error("Nach Bereinigung sind keine Prompts mehr übrig.")
                st.stop()

            clustered, emb, emb_umap, topic_model = prompt_clustering(clean, config)
            summary = build_cluster_summary(clustered)

            st.session_state["clustered"] = clustered
            st.session_state["cluster_summary"] = summary
            st.session_state["manual_labels"] = {int(r.cluster_id): f"Cluster {int(r.cluster_id)}" for r in summary.itertuples()}
            st.success("Analyse abgeschlossen.")
        except Exception as e:
            st.exception(e)

if "clustered" in st.session_state:
    clustered = st.session_state["clustered"]
    summary = st.session_state["cluster_summary"]
    manual_labels = st.session_state.get("manual_labels", {})

    st.header("3) Prompt-Cluster analysieren")

    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        min_user_prompts = st.number_input("Mindestanzahl Prompts pro User", min_value=1, value=1)
    with colf2:
        min_cluster_size_filter = st.number_input("Mindestclustergröße", min_value=1, value=1)
    with colf3:
        selected_cluster_filter = st.selectbox("Cluster-Filter", ["Alle"] + sorted(clustered["cluster_id"].astype(str).unique().tolist()))

    dfv = clustered.copy()
    user_counts = dfv.groupby("user_id").size()
    keep_users = user_counts[user_counts >= min_user_prompts].index
    dfv = dfv[dfv["user_id"].isin(keep_users)]
    cluster_counts = dfv["cluster_id"].value_counts()
    keep_clusters = cluster_counts[cluster_counts >= min_cluster_size_filter].index
    dfv = dfv[dfv["cluster_id"].isin(keep_clusters)]
    if selected_cluster_filter != "Alle":
        dfv = dfv[dfv["cluster_id"].astype(str) == selected_cluster_filter]

    fig_scatter = px.scatter(
        dfv,
        x="umap_x",
        y="umap_y",
        color=dfv["cluster_id"].astype(str),
        hover_data=["user_id", "prompt_text_original", "cluster_id", "session", "treatment_arm", "grade", "problem_id"],
        title="UMAP-Scatter der Einzelprompts nach Cluster",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    fig_bar = px.bar(summary, x="cluster_id", y="n_prompts", title="Clustergrößen")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Clusterübersicht")
    editable = summary.copy()
    editable["manual_label"] = editable["cluster_id"].map(lambda x: manual_labels.get(int(x), f"Cluster {int(x)}"))
    edited = st.data_editor(editable, use_container_width=True, num_rows="fixed")

    new_labels = {int(r["cluster_id"]): r["manual_label"] for _, r in edited.iterrows()}
    st.session_state["manual_labels"] = new_labels

    st.header("4) User-Profile analysieren")
    profiles = derive_user_profiles(clustered, new_labels)
    profiles = cluster_users(profiles)
    st.session_state["user_profiles"] = profiles

    st.dataframe(profiles, use_container_width=True)

    user_list = sorted(profiles["user_id"].astype(str).tolist())
    selected_user = st.selectbox("User auswählen", user_list)
    if selected_user:
        uid = selected_user
        sub = clustered[clustered["user_id"].astype(str) == uid]
        dist = sub["cluster_id"].value_counts().reset_index()
        dist.columns = ["cluster_id", "count"]
        dist["label"] = dist["cluster_id"].map(lambda x: new_labels.get(int(x), f"Cluster {int(x)}"))
        fig_user = px.bar(dist, x="label", y="count", title=f"Clusterprofil User {uid}")
        st.plotly_chart(fig_user, use_container_width=True)
        st.write("Typische User-Prompts:")
        st.dataframe(sub[["timestamp", "prompt_text_original", "cluster_id"]].head(20), use_container_width=True)

    st.header("5) User-Typen (zweites Clustering)")
    type_counts = profiles["user_type_label"].value_counts().reset_index()
    type_counts.columns = ["user_type", "n_users"]
    fig_types = px.bar(type_counts, x="user_type", y="n_users", title="Verteilung der User-Typen")
    st.plotly_chart(fig_types, use_container_width=True)

    st.header("6) Export")
    c1, c2 = st.columns(2)
    with c1:
        export_df_button(clustered, "Prompts mit Clusterzuordnung (CSV)", "prompts_with_clusters.csv")
        export_df_button(summary, "Clusterübersicht (CSV)", "cluster_overview.csv")
        export_df_button(profiles, "User-Profile inkl. User-Typen (CSV)", "user_profiles.csv")
    with c2:
        labels_df = pd.DataFrame([{"cluster_id": k, "manual_label": v} for k, v in new_labels.items()])
        export_df_button(labels_df, "Manuelle Clusterlabels (CSV)", "cluster_labels.csv")

        export_payload = {
            "cluster_labels": new_labels,
            "cluster_summary": summary.to_dict(orient="records"),
            "user_profiles": profiles.to_dict(orient="records"),
        }
        st.download_button(
            "Gesamtergebnis als JSON",
            data=json.dumps(export_payload, ensure_ascii=False, indent=2),
            file_name="prompt_profile_analysis.json",
            mime="application/json",
        )

else:
    st.info("Bitte eine CSV hochladen und die Analyse starten.")
