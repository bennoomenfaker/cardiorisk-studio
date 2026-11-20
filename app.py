import json
import base64
import io
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Configuration générale (SaaS-like UI)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioRisk Studio — Aide à la décision",
    page_icon="❤",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "heart_disease_uci.csv"
NOTEBOOK_PATH = "heart_disease_project.ipynb"
ASSETS_DIR = Path("assets")
RANDOM_STATE = 42

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
      .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #334155 100%);
        color: #f8fafc; padding: 1.2rem 1.4rem; border-radius: 16px; margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
      }
      .hero h1 {margin: 0; font-size: 1.8rem; color: #fff;}
      .hero p {margin: .35rem 0 0 0; color: #cbd5e1;}
      .card {
        background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px; padding: 1rem 1.1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03); margin-bottom: .75rem; color: #0f172a;
      }
      .card b { color: #0f172a; }
      .card code {
        color: #0f172a;
        background: #f1f5f9;
        padding: 0.1rem 0.35rem;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
      }
      .step {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: .8rem .9rem;
        text-align: center; min-height: 110px;
      }
      .step h4 {margin: 0 0 .25rem 0; font-size: 1rem; color:#0f172a;}
      .step p {margin: 0; font-size: .85rem; color:#475569;}
      .arrow {text-align:center; font-size: 1.6rem; color:#64748b; padding-top: 1.8rem;}
      .muted {color: #64748b; font-size: 0.92rem;}
      .small-note {font-size: 0.85rem; color:#475569;}
      .kpi-label {font-size:.85rem; color:#64748b; margin-bottom:.2rem;}
      .kpi-value {font-size:1.6rem; font-weight:700; color:#0f172a; line-height:1.1;}
      .kpi-help {font-size:.8rem; color:#475569; margin-top:.2rem;}
      .img-caption {font-size: 0.85rem; color:#475569; margin-top: -0.2rem; margin-bottom: .6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Métadonnées / descriptions pédagogiques
# ---------------------------------------------------------------------------
TEAM = "Faker BEN NOOMEN · Haythem BEN BOUBAKER · Mariem DRIDI"

ATTR_DESCRIPTIONS = {
    "id": "Identifiant unique du patient (exclu de la prédiction).",
    "age": "Âge du patient en années.",
    "sex": "Sexe biologique (Male/Female).",
    "dataset": "Centre d'origine (Cleveland, Hungary, Switzerland, VA Long Beach). Métadonnée.",
    "cp": "Type de douleur thoracique : typical angina, atypical angina, non-anginal, asymptomatic.",
    "trestbps": "Pression artérielle au repos (mm Hg) à l'admission.",
    "chol": "Cholestérol sérique (mg/dl).",
    "fbs": "Glycémie à jeun > 120 mg/dl (True/False).",
    "restecg": "ECG au repos : normal, st-t abnormality, lv hypertrophy.",
    "thalch": "Fréquence cardiaque maximale atteinte à l'effort (bpm).",
    "exang": "Angine induite par l'effort (True/False).",
    "oldpeak": "Dépression du segment ST induite par l'effort (ischémie).",
    "slope": "Pente du segment ST à l'effort : upsloping, flat, downsloping.",
    "ca": "Nombre de vaisseaux majeurs (0-3) colorés par fluoroscopie.",
    "thal": "Résultat du test de thalassémie/perfusion : normal, fixed defect, reversable defect.",
    "num": "Diagnostic original (0 = absence, 1-4 = présence/sévérité). Sert à construire la cible.",
}

JARGON_GLOSSARY = [
    ("cp (douleur thoracique)", "Type de douleur : angine typique/atypique, douleur non angineuse, ou asymptomatique."),
    ("trestbps", "Pression artérielle au repos, mesurée en mm Hg."),
    ("chol", "Cholestérol sérique (mg/dl) dans le sang."),
    ("fbs", "Glycémie à jeun élevée (> 120 mg/dl) : Oui/Non."),
    ("restecg", "Résultat de l'électrocardiogramme au repos."),
    ("thalch", "Fréquence cardiaque maximale atteinte pendant l'épreuve d'effort."),
    ("exang", "Angine (douleur) provoquée par l'effort : Oui/Non."),
    ("oldpeak", "Dépression du segment ST à l'effort par rapport au repos (marqueur possible d'ischémie)."),
    ("slope", "Pente du segment ST à l'effort : montante, plate ou descendante."),
    ("ca", "Nombre de vaisseaux majeurs (0 à 3) visibles à la fluoroscopie."),
    ("thal", "Résultat du test de perfusion/thalassémie : normal, défaut fixe, défaut réversible."),
]

CONTEXTE_NOTEBOOK = r"""
### A. Présentation et compréhension (contexte, idée, problématique, données)

**Contexte métier et idée**  
Les maladies cardiovasculaires sont une cause majeure de morbidité et de mortalité. À partir de données cliniques de routine (symptômes, mesures, ECG, épreuve d’effort), l’idée est de construire un outil **d’aide à la décision** pour estimer le risque de présence d’une maladie cardiaque et aider au dépistage / à la priorisation des examens.

**Question / problématique métier**  
*Peut-on prédire la présence d’une maladie cardiaque à partir de variables cliniques de routine, afin d’aider au dépistage et à la priorisation des examens ?*

**Comment décider « maladie cardiaque : Oui ou Non » ? (cible / target)**  
- **Variable originale :** `num` (diagnostic) — `0` = absence, `1-4` = présence/sévérité de maladie.  
- **Cible binaire créée :** `target = 1` (**Oui**, maladie présente) si `num > 0`, sinon `target = 0` (**Non**, absence).  
- Cette formulation répond directement à la question métier de dépistage (présence/absence).

**Source des données**  
- **Jeu de données :** Heart Disease (UCI Machine Learning Repository), version combinée multi-centres.  
- **Fichier utilisé :** `heart_disease_uci.csv` — **Centres :** Cleveland, Hungary, Switzerland, VA Long Beach.  
- **Liens :** https://archive.ics.uci.edu/dataset/45/heart+disease | https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data  
- **Licence/utilisation :** données publiques anonymisées, usage pédagogique.

**Variables et jargon (explication simple)**  
- **Âge/mesures :** `age`, `trestbps` (pression au repos, mm Hg), `chol` (cholestérol, mg/dl), `thalch` (fréquence cardiaque max à l’effort), `oldpeak` (dépression du segment ST à l’effort).  
- **Symptômes/examens :** `cp` (type de douleur thoracique), `exang` (angine à l’effort oui/non), `fbs` (glycémie à jeun > 120 mg/dl), `restecg` (ECG au repos), `slope` (pente du segment ST), `ca` (nb de vaisseaux majeurs colorés 0–3), `thal` (test thalassémie/perfusion).  
- **Métadonnées/exclus :** `id`, `dataset` ; `num` sert uniquement à construire `target`.
"""

# ---------------------------------------------------------------------------
# Helpers: images, dataset, notebook
# ---------------------------------------------------------------------------
def show_image_if_exists(filename, caption, use_container_width=True):
    p = ASSETS_DIR / filename
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=use_container_width)
        return True
    return False


def render_notebook_cell_output(output):
    otype = output.get("output_type")
    if otype == "stream":
        text = "".join(output.get("text", []))
        if text.strip():
            st.code(text.rstrip(), language="text")
    elif otype in ("execute_result", "display_data"):
        data = output.get("data", {})
        if "image/png" in data:
            b64 = data["image/png"]
            if isinstance(b64, list):
                b64 = "".join(b64)
            try:
                st.image(base64.b64decode(b64), use_container_width=True)
            except Exception:
                st.warning("Impossible d'afficher une image de sortie du notebook.")
        elif "image/jpeg" in data:
            b64 = data["image/jpeg"]
            if isinstance(b64, list):
                b64 = "".join(b64)
            try:
                st.image(base64.b64decode(b64), use_container_width=True)
            except Exception:
                pass
        elif "text/html" in data:
            html = "".join(data.get("text/html", []))
            components.html(html, height=300, scrolling=True)
        elif "text/plain" in data:
            text = "".join(data.get("text/plain", []))
            if text.strip():
                st.code(text.rstrip(), language="text")
    elif otype == "error":
        tb = "\n".join(output.get("traceback", []))
        if tb:
            st.code(tb, language="text")


@st.cache_data
def load_notebook(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_notebook(nb):
    cells = nb.get("cells", [])
    st.caption(f"Notebook chargé : {len(cells)} cellules (affichage en lecture seule dans Streamlit).")
    show_code = st.checkbox("Afficher le code des cellules", value=False)
    show_outputs = st.checkbox("Afficher les sorties (outputs)", value=True)
    max_cells = st.slider("Nombre max de cellules à afficher", 10, len(cells), min(len(cells), 60), step=5)
    for i, cell in enumerate(cells[:max_cells]):
        ctype = cell.get("cell_type")
        src = "".join(cell.get("source", []))
        if ctype == "markdown":
            if src.strip():
                st.markdown(src)
        elif ctype == "code":
            if show_code and src.strip():
                st.markdown(f"**Code (cellule {i+1})**")
                st.code(src, language="python")
            if show_outputs:
                for out in cell.get("outputs", []):
                    render_notebook_cell_output(out)
        st.divider()
    if max_cells < len(cells):
        st.info(f"Affichage limité aux {max_cells} premières cellules. Augmentez le curseur pour voir la suite.")


# ---------------------------------------------------------------------------
# Chargement et préparation des données
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage de base : copie, correction 0 invalides, création target."""
    data = df.copy()
    data["chol"] = data["chol"].replace(0, np.nan)
    data["trestbps"] = data["trestbps"].replace(0, np.nan)
    data["target"] = (data["num"] > 0).astype(int)  # 1 = Oui, 0 = Non
    return data


@st.cache_data
def data_quality_report(df_raw: pd.DataFrame):
    """Diagnostics qualité : doublons, manquants, invalides, avant/après."""
    duplicates = int(df_raw.duplicated().sum())
    missing_before = df_raw.isna().sum().sort_values(ascending=False)
    chol_zero = int((df_raw["chol"] == 0).sum())
    trestbps_zero = int((df_raw["trestbps"] == 0).sum())
    df_clean = clean_data(df_raw)
    missing_after = df_clean.isna().sum().sort_values(ascending=False)
    compare = pd.DataFrame({
        "manquants_avant": df_raw.isna().sum(),
        "manquants_apres": df_clean.isna().sum(),
    })
    compare["delta"] = compare["manquants_apres"] - compare["manquants_avant"]
    return {
        "duplicates": duplicates,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "chol_zero": chol_zero,
        "trestbps_zero": trestbps_zero,
        "compare": compare.sort_values("manquants_apres", ascending=False),
        "df_clean": df_clean,
    }


def build_pipeline(model_name: str, num_features, cat_features):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features),
    ])

    if model_name == "Régression Logistique":
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    elif model_name == "KNN (k=7)":
        model = KNeighborsClassifier(n_neighbors=7)
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)

    return Pipeline([("preprocess", pre), ("model", model)])


@st.cache_resource
def train_models(df: pd.DataFrame):
    data = clean_data(df)
    features = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalch", "exang", "oldpeak", "slope", "ca", "thal",
    ]
    X = data[features]
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    num_features = X_train.select_dtypes(include=["int64", "float64", "number"]).columns.tolist()
    cat_features = [c for c in X_train.columns if c not in num_features]

    model_names = ["Baseline (classe majoritaire)", "Régression Logistique", "Random Forest", "KNN (k=7)"]
    results = {}
    for name in model_names:
        if name == "Baseline (classe majoritaire)":
            baseline = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
            pipe = Pipeline([("preprocess", "passthrough"), ("model", baseline)])
            pipe.fit(X_train, y_train)
        else:
            pipe = build_pipeline(name, num_features, cat_features)
            pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else y_pred.astype(float)
        results[name] = {
            "pipeline": pipe,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

    return data, features, results, num_features, cat_features


def compute_metrics(yt, yp, yprob):
    mse = mean_squared_error(yt, yprob)
    metrics = {
        "Accuracy": accuracy_score(yt, yp),
        "Précision": precision_score(yt, yp, zero_division=0),
        "Rappel": recall_score(yt, yp, zero_division=0),
        "F1": f1_score(yt, yp, zero_division=0),
        "MAE": mean_absolute_error(yt, yprob),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(yt, yprob),
    }
    try:
        metrics["ROC-AUC"] = roc_auc_score(yt, yprob)
    except Exception:
        metrics["ROC-AUC"] = np.nan
    return metrics


def show_target_explainer():
    st.markdown(
        """
        <div class="card">
          <b>Comment on décide « Maladie cardiaque : Oui ou Non » ? (cible / target)</b><br/>
          • Variable originale <code>num</code> : <code>0</code> = absence, <code>1-4</code> = présence/sévérité.<br/>
          • Cible créée <code>target</code> : <code>1 (Oui)</code> si <code>num &gt; 0</code>, sinon <code>0 (Non)</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )


def process_schema():
    st.markdown("#### Schéma explicatif du processus Machine Learning (pipeline du projet)")
    c1, a1, c2, a2, c3, a3, c4, a4, c5, a5, c6 = st.columns([1.2, 0.25, 1.2, 0.25, 1.2, 0.25, 1.2, 0.25, 1.2, 0.25, 1.2])
    with c1:
        st.markdown('<div class="step"><h4>1. Données</h4><p>Collecte & compréhension<br/>(CSV UCI, head/info/describe)</p></div>', unsafe_allow_html=True)
    with a1:
        st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="step"><h4>2. Qualité & Nettoyage</h4><p>Doublons, manquants, incohérences<br/>avant/après + justifications</p></div>', unsafe_allow_html=True)
    with a2:
        st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="step"><h4>3. Préparation</h4><p>Features, nouvelle cible <code>target</code><br/>encodage, imputation, scaling</p></div>', unsafe_allow_html=True)
    with a3:
        st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="step"><h4>4. Split & Modèles</h4><p>Train/Test stratifié (80/20)<br/>baseline + modèles (fit sur train)</p></div>', unsafe_allow_html=True)
    with a4:
        st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="step"><h4>5. Évaluation</h4><p>Accuracy, Précision, Rappel, F1<br/>matrice de confusion, ROC-AUC</p></div>', unsafe_allow_html=True)
    with a5:
        st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)
    with c6:
        st.markdown('<div class="step"><h4>6. Prédiction & Décision</h4><p>Réel vs prédit, scénario Oui/Non<br/>risque, alerte, interprétation</p></div>', unsafe_allow_html=True)
    st.caption("Principe clé : le prétraitement est ajusté sur le train uniquement (pipeline sklearn) pour éviter la fuite de données. `random_state=42`.")


# ---------------------------------------------------------------------------
# En-tête & Sidebar
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
      <h1>CardioRisk Studio</h1>
      <p>Dashboard SaaS pédagogique d'aide à la décision — prédiction de la maladie cardiaque (Oui/Non)</p>
      <p class="small-note" style="margin-top:.5rem; color:#e2e8f0;">Projet : UCI Heart Disease · Groupe : {TEAM}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Navigation")
st.sidebar.caption("Projet de synthèse — Data & Machine Learning")
page = st.sidebar.radio(
    "Aller à",
    [
        "Contexte & Problématique",
        "Dataset",
        "Notebook (ipynb)",
        "Processus ML & Qualité",
        "Tableau de bord (KPIs)",
        "Données & Glossaire",
        "EDA & Graphiques",
        "Corrélation",
        "Modèles & Métriques",
        "Test (réel vs prédit)",
        "Prédiction (scénario)",
        "Interprétation & Éthique",
    ],
)
st.sidebar.markdown("---")
st.sidebar.info("Aide à la décision pédagogique. Toute prédiction doit être vérifiée par un professionnel de santé.")

# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
try:
    raw = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Impossible de charger '{DATA_PATH}'. Détail: {e}")
    st.stop()

dq = data_quality_report(raw)
data = dq["df_clean"]
data, features_list, results, num_features, cat_features = train_models(raw)

all_metrics = []
for name, res in results.items():
    m = compute_metrics(res["y_test"], res["y_pred"], res["y_proba"])
    m["Modèle"] = name
    all_metrics.append(m)
metrics_df = pd.DataFrame(all_metrics).set_index("Modèle")
ml_metrics_global = metrics_df.drop(index=["Baseline (classe majoritaire)"], errors="ignore")
best_f1_name = ml_metrics_global["F1"].idxmax() if not ml_metrics_global.empty else metrics_df["F1"].idxmax()

# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
if page == "Contexte & Problématique":
    st.header("Contexte, idée et problématique métier")

    # Layout d'ouverture : contenu à gauche, cœur visible immédiatement à droite
    left, right = st.columns([1.15, 0.95], gap="large")
    with right:
        st.markdown("#### Le cœur au centre de la problématique")
        # Afficher directement une image du cœur à l'ouverture
        if not show_image_if_exists("real_heart.jpg", "Cœur humain réel (vue antérieure). Image liée à la problématique cardiovasculaire. Source : Wikimedia Commons."):
            show_image_if_exists("heart_diagram.png", "Schéma anatomique du cœur. Source : Wikimedia Commons.")
        with st.expander("Voir d'autres images liées à la problématique"):
            show_image_if_exists("heart_diagram.png", "Schéma anatomique du cœur (Wikimedia Commons).")
            ecg_svg = ASSETS_DIR / "ecg_wave.svg"
            if ecg_svg.exists():
                try:
                    st.image(str(ecg_svg), caption="Signal ECG (onde électrique du cœur) — examen clé dans l’évaluation cardiaque. Source : Wikimedia Commons.", use_container_width=True)
                except Exception:
                    st.caption("ECG (SVG) disponible dans les assets : `ecg_wave.svg`.")
            show_image_if_exists("risk_factors_illustration.png", "Facteurs cliniques associés au risque (illustration pédagogique liée au projet).")
            show_image_if_exists("prediction_flow.png", "Schéma de principe : de la donnée clinique à l’aide à la décision (Oui/Non).")

    with left:
        st.markdown(CONTEXTE_NOTEBOOK)
        st.divider()
        show_target_explainer()
        st.markdown(
            """
            **Ce que fait le site (SaaS d’aide à la décision) :**
            - explique les données et la qualité (doublons, valeurs manquantes, corrections) ;
            - montre le **processus ML** et les étapes de préparation/modélisation ;
            - présente les analyses, corrélations et graphiques ;
            - compare une baseline et plusieurs modèles avec métriques transparentes ;
            - permet un **scénario patient** → prédiction **Oui/Non**, probabilité, niveau de risque et alerte (validation humaine requise) ;
            - donne accès au **dataset** et au **notebook (ipynb)** du projet directement dans l’application.
            """
        )
        st.caption("Données publiques anonymisées (UCI Heart Disease). Outil pédagogique — ne remplace pas un diagnostic médical.")

elif page == "Dataset":
    st.header("Dataset — jeu de données du projet")
    st.write("Accès au dataset utilisé dans le projet (brut et nettoyé/préparé), avec aperçu, statistiques et téléchargement.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes (brut)", raw.shape[0])
    c2.metric("Colonnes (brut)", raw.shape[1])
    c3.metric("Doublons", dq["duplicates"])
    c4.metric("Valeurs manquantes (brut)", int(raw.isna().sum().sum()))

    tab_raw, tab_clean, tab_exports = st.tabs(["Dataset brut (UCI)", "Dataset nettoyé/préparé", "Exports & téléchargements"])

    with tab_raw:
        st.subheader("Aperçu — `heart_disease_uci.csv`")
        st.dataframe(raw.head(50), use_container_width=True)
        st.markdown("**Statistiques descriptives**")
        st.dataframe(raw.describe(include="all").T, use_container_width=True)
        st.markdown("**Types et valeurs manquantes**")
        info_df = pd.DataFrame({
            "dtype": raw.dtypes.astype(str),
            "non_null": raw.notna().sum(),
            "missing": raw.isna().sum(),
            "missing_%": (raw.isna().mean() * 100).round(2),
        })
        st.dataframe(info_df, use_container_width=True)
        st.download_button(
            "Télécharger le dataset brut (CSV)",
            data=raw.to_csv(index=False).encode("utf-8"),
            file_name="heart_disease_uci.csv",
            mime="text/csv",
        )

    with tab_clean:
        st.subheader("Aperçu — données après nettoyage/préparation de base")
        st.caption("Corrections appliquées : `chol==0` et `trestbps==0` → NaN, création de `target` (Oui si num>0). L'imputation/encodage/scaling pour les modèles se fait dans le pipeline.")
        st.dataframe(data.head(50), use_container_width=True)
        st.dataframe(data.describe(include="all").T, use_container_width=True)
        csv_clean = data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Télécharger le dataset nettoyé/préparé (CSV)",
            data=csv_clean,
            file_name="heart_disease_cleaned.csv",
            mime="text/csv",
        )

    with tab_exports:
        st.subheader("Fichiers/export du projet")
        # raw download again
        st.download_button("Dataset brut — heart_disease_uci.csv", data=raw.to_csv(index=False).encode("utf-8"), file_name="heart_disease_uci.csv", mime="text/csv")
        st.download_button("Dataset nettoyé — heart_disease_cleaned.csv", data=data.to_csv(index=False).encode("utf-8"), file_name="heart_disease_cleaned.csv", mime="text/csv")
        for fname, label in [
            ("resultats_modeles.csv", "Résultats des modèles — resultats_modeles.csv"),
            ("exemples_reel_vs_predit.csv", "Exemples réel vs prédit — exemples_reel_vs_predit.csv"),
        ]:
            p = Path(fname)
            if p.exists():
                st.download_button(label, data=p.read_bytes(), file_name=fname, mime="text/csv")
            else:
                st.caption(f"{fname} non trouvé dans le dossier.")
        st.markdown("---")
        st.markdown("**Aperçu des exports (si disponibles)**")
        for fname in ["resultats_modeles.csv", "exemples_reel_vs_predit.csv"]:
            p = Path(fname)
            if p.exists():
                st.markdown(f"`{fname}`")
                try:
                    st.dataframe(pd.read_csv(p), use_container_width=True)
                except Exception:
                    st.caption("Impossible de lire le fichier.")

elif page == "Notebook (ipynb)":
    st.header("Notebook du projet (ipynb)")
    st.write("Affichage du notebook existant du projet directement dans Streamlit (lecture seule), avec ses analyses, graphiques et résultats.")
    nb_path = Path(NOTEBOOK_PATH)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"**Fichier :** `{NOTEBOOK_PATH}`")
        if nb_path.exists():
            st.download_button(
                "Télécharger le notebook (.ipynb)",
                data=nb_path.read_bytes(),
                file_name=NOTEBOOK_PATH,
                mime="application/x-ipynb+json",
            )
    with c2:
        st.markdown("**Options d’affichage**")
        st.caption("Vous pouvez afficher/masquer le code et les sorties, et choisir le nombre de cellules à rendre.")
    st.divider()
    if nb_path.exists():
        try:
            nb = load_notebook(str(nb_path))
            render_notebook(nb)
        except Exception as e:
            st.error(f"Impossible de charger le notebook : {e}")
    else:
        st.error(f"Notebook introuvable : {NOTEBOOK_PATH}")

elif page == "Processus ML & Qualité":
    st.header("Processus de Machine Learning, qualité des données et préparation")
    process_schema()

    st.subheader("1) Doublons et valeurs manquantes — constat et traitement")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes (brut)", raw.shape[0])
    c2.metric("Colonnes (brut)", raw.shape[1])
    c3.metric("Doublons détectés", dq["duplicates"])
    c4.metric("Total manquants (brut)", int(raw.isna().sum().sum()))

    if dq["duplicates"] == 0:
        st.success("**Doublons :** aucun doublon détecté (aucune suppression nécessaire).")
    else:
        st.warning(f"**Doublons :** {dq['duplicates']} doublon(s) détecté(s) — à traiter (suppression après vérification).")

    st.markdown(
        f"""
        **Valeurs manquantes / invalides (comment modifiées) :**
        - `chol == 0` : **{dq['chol_zero']}** valeurs invalides (non plausibles) → remplacées par `NaN` puis **imputées** (médiane pour numériques) dans le pipeline.
        - `trestbps == 0` : **{dq['trestbps_zero']}** valeur invalide → remplacée par `NaN` puis imputée.
        - Autres manquants (`ca`, `thal`, `slope`, etc.) : conservés puis **imputés** (médiane pour numériques, modalité la plus fréquente pour catégorielles) via `SimpleImputer` dans le **pipeline** (ajusté sur le train uniquement, anti-fuite).
        - Aucune suppression massive de lignes : on préserve l’information du dataset.
        """
    )

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Graphique : valeurs manquantes par variable (avant correction)**")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        miss = raw.isna().sum().sort_values(ascending=True)
        miss[miss > 0].plot(kind="barh", ax=ax, color="#E45756")
        ax.set_title("Valeurs manquantes par variable (données brutes)")
        ax.set_xlabel("Nombre de valeurs manquantes")
        st.pyplot(fig)
        st.caption("Phase nettoyage : identification des variables les plus incomplètes (ex. ca, thal, slope).")
    with colB:
        st.markdown("**Graphique : valeurs manquantes par variable (après correction 0→NaN)**")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        miss_a = data.isna().sum().sort_values(ascending=True)
        miss_a[miss_a > 0].plot(kind="barh", ax=ax, color="#F58518")
        ax.set_title("Valeurs manquantes après correction des 0 invalides")
        ax.set_xlabel("Nombre de valeurs manquantes")
        st.pyplot(fig)
        st.caption("Après correction, chol/trestbps à 0 deviennent manquants (pour être imputés correctement).")

    st.markdown("**Tableau comparatif avant/après (manquants)**")
    st.dataframe(dq["compare"][dq["compare"]["manquants_apres"] > 0], use_container_width=True)

    st.subheader("2) Nouvelle(s) feature(s) / variable(s) créée(s)")
    st.markdown(
        """
        - **`target` (nouvelle variable cible, feature/label dérivée)** : créée à partir de `num`.  
          **Calcul :** `target = 1` si `num > 0` (maladie présente, Oui), sinon `target = 0` (absence, Non).  
          *Objectif :* transformer le diagnostic multi-classe/sévérité en classification binaire adaptée à la question métier.
        - **Variables explicatives (features) utilisées pour le modèle :** `age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal` (13 variables cliniques).
        - **Exclues :** `id` (identifiant), `dataset` (métadonnée), `num` (source directe de la cible → évite la fuite de données).
        - Aucune autre feature calculée manuellement (pas de ratio/agrégat ajouté) ; les transformations (imputation, One-Hot, scaling) sont réalisées dans le pipeline.
        """
    )
    tc = data["target"].value_counts().sort_index()
    st.write(f"Répartition de `target` : Non (0) = **{int(tc.get(0,0))}**, Oui (1) = **{int(tc.get(1,0))}**.")

    st.subheader("3) Répartition par sexe (phase d’exploration/nettoyage)")
    colS1, colS2 = st.columns(2)
    with colS1:
        sex_counts = raw["sex"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(sex_counts.values, labels=sex_counts.index, autopct="%1.1f%%", startangle=90, colors=["#72B7B2", "#E45756"])
        ax.set_title("Répartition des patients par sexe (camembert)")
        st.pyplot(fig)
        st.caption("Visualisation circulaire (pie chart) de la répartition par sexe.")
    with colS2:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        sns.countplot(data=raw, x="sex", hue=(raw["num"] > 0).astype(int), ax=ax, palette=["#4C78A8", "#F58518"])
        ax.set_title("Répartition par sexe selon la maladie (Oui/Non)")
        ax.legend(title="Maladie", labels=["Non (0)", "Oui (1)"])
        st.pyplot(fig)
        st.caption("Le déséquilibre hommes/femmes peut constituer un biais de représentativité.")

elif page == "Tableau de bord (KPIs)":
    st.header("Tableau de bord (vue Power BI-like) — KPIs et interprétation")
    st.caption("Vue synthétique des indicateurs clés (données, qualité, modèle) pour le pilotage et l’aide à la décision.")

    best = metrics_df.loc[best_f1_name]
    test_n = len(results[best_f1_name]["y_test"])
    fn_best = int(((results[best_f1_name]["y_test"].values == 1) & (results[best_f1_name]["y_pred"] == 0)).sum())
    fp_best = int(((results[best_f1_name]["y_test"].values == 0) & (results[best_f1_name]["y_pred"] == 1)).sum())

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown('<div class="card"><div class="kpi-label">Patients (dataset)</div><div class="kpi-value">{}</div><div class="kpi-help">Volume total analysé</div></div>'.format(len(raw)), unsafe_allow_html=True)
    with k2:
        st.markdown('<div class="card"><div class="kpi-label">Taux de maladie (Oui)</div><div class="kpi-value">{:.1%}</div><div class="kpi-help">Part des cas num &gt; 0</div></div>'.format((raw["num"] > 0).mean()), unsafe_allow_html=True)
    with k3:
        st.markdown('<div class="card"><div class="kpi-label">Valeurs manquantes (brut)</div><div class="kpi-value">{}</div><div class="kpi-help">Cellules vides à traiter</div></div>'.format(int(raw.isna().sum().sum())), unsafe_allow_html=True)
    with k4:
        st.markdown('<div class="card"><div class="kpi-label">Doublons</div><div class="kpi-value">{}</div><div class="kpi-help">Intégrité des lignes</div></div>'.format(dq["duplicates"]), unsafe_allow_html=True)
    with k5:
        st.markdown('<div class="card"><div class="kpi-label">Meilleur F1 (test)</div><div class="kpi-value">{:.3f}</div><div class="kpi-help">{}</div></div>'.format(best["F1"], best_f1_name), unsafe_allow_html=True)
    with k6:
        st.markdown('<div class="card"><div class="kpi-label">Rappel (meilleur F1)</div><div class="kpi-value">{:.3f}</div><div class="kpi-help">Détection des cas « Oui »</div></div>'.format(best["Rappel"]), unsafe_allow_html=True)

    st.markdown("#### Explication et interprétation des KPI")
    st.markdown(
        f"""
        - **Patients / Taux de maladie (Oui)** : contexte du jeu de données (volume et équilibre, ≈ {((raw['num']>0).mean()*100):.1f}% de « Oui ») pour interpréter les performances.
        - **Valeurs manquantes (brut) / Doublons** : indicateurs de qualité des données ; ici {int(raw.isna().sum().sum())} cellules manquantes au total (ex. `ca`, `thal`, `slope`) et {dq['duplicates']} doublon, avec correction des 0 invalides avant imputation.
        - **Meilleur F1 / Rappel (modèle {best_f1_name}, test n={test_n})** : indicateurs de performance (compromis précision-rappel). Rappel élevé = moins de **faux négatifs** (FN={fn_best} sur le test) ; faux positifs FP={fp_best}.
        - **Lecture métier** : en dépistage, réduire les FN est prioritaire, avec validation humaine obligatoire. L’ordre des KPI suit une logique **données → qualité → performance**.
        """
    )

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        counts = data["target"].value_counts().sort_index()
        ax.bar(["Non (0)", "Oui (1)"], counts.values, color=["#4C78A8", "#F58518"])
        ax.set_title("Répartition cible (Oui/Non)")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 5, str(int(v)), ha="center")
        st.pyplot(fig)
        st.caption("Interprétation : base relativement équilibrée pour entraîner un classifieur binaire.")
    with c2:
        comp = metrics_df[["Accuracy", "Précision", "Rappel", "F1", "ROC-AUC"]].astype(float)
        fig, ax = plt.subplots(figsize=(6.8, 4))
        comp.plot(kind="bar", ax=ax)
        ax.set_title("Comparaison des modèles (KPIs de performance)")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='x', rotation=20)
        ax.legend(loc="lower right", fontsize=8)
        st.pyplot(fig)
        st.caption("Interprétation : les modèles appris doivent surpasser la baseline ; le choix final dépend du compromis rappel/précision.")

    c3, c4 = st.columns(2)
    with c3:
        miss = raw.isna().sum()
        miss = miss[miss > 0].sort_values()
        fig, ax = plt.subplots(figsize=(6.5, 4))
        miss.plot(kind="barh", ax=ax, color="#E45756")
        ax.set_title("KPI qualité : valeurs manquantes (brut)")
        st.pyplot(fig)
        st.caption("Interprétation : ca/thal/slope concentrent l’essentiel des manquants (limite de fiabilité).")
    with c4:
        resb = results[best_f1_name]
        cm = confusion_matrix(resb["y_test"], resb["y_pred"])
        fig, ax = plt.subplots(figsize=(5.2, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Prédit Non", "Prédit Oui"], yticklabels=["Réel Non", "Réel Oui"])
        ax.set_title(f"Matrice de confusion — {best_f1_name}")
        st.pyplot(fig)
        st.caption("Interprétation : visualise TN/FP/FN/TP ; attention particulière aux FN en contexte santé.")

elif page == "Données & Glossaire":
    st.header("Données, attributs et jargon expliqué")
    st.write("Comprendre les variables (jargon clinique) utilisées pour prédire **Maladie cardiaque : Oui/Non**.")

    st.subheader("Glossaire du jargon (compréhension rapide)")
    st.dataframe(pd.DataFrame(JARGON_GLOSSARY, columns=["Variable / Terme", "Explication simple"]), use_container_width=True, hide_index=True)

    with st.expander("Tableau détaillé des attributs (rôle et description)"):
        desc = pd.DataFrame({
            "Attribut": list(ATTR_DESCRIPTIONS.keys()),
            "Description": list(ATTR_DESCRIPTIONS.values()),
            "Rôle": [
                "exclu", "feature", "feature", "métadonnée", "feature", "feature", "feature",
                "feature", "feature", "feature", "feature", "feature", "feature", "feature",
                "feature", "cible originale",
            ],
        })
        st.dataframe(desc, use_container_width=True, hide_index=True)

    st.subheader("Types, valeurs non-nulles et manquantes (brut)")
    info_df = pd.DataFrame({
        "dtype": raw.dtypes.astype(str),
        "non_null": raw.notna().sum(),
        "missing": raw.isna().sum(),
        "missing_%": (raw.isna().mean() * 100).round(2),
    })
    st.dataframe(info_df, use_container_width=True)

    st.subheader("Statistiques descriptives")
    st.dataframe(raw.describe(include="all").T, use_container_width=True)
    st.info("Des 0 non plausibles existent pour `chol`/`trestbps` (traités comme manquants puis imputés). Voir la page 'Processus ML & Qualité' pour le détail des corrections.")

elif page == "EDA & Graphiques":
    st.header("Analyse exploratoire (EDA) et visualisations")
    st.write("Graphiques utilisés pour comprendre la distribution, la qualité et les relations avec la cible (Oui/Non), en cohérence avec le processus d’analyse.")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        counts = data["target"].value_counts().sort_index()
        ax.bar(["Non (0)", "Oui (1)"], counts.values, color=["#4C78A8", "#F58518"])
        ax.set_title("Répartition de la cible (maladie cardiaque : Oui/Non)")
        ax.set_ylabel("Nombre de patients")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 5, str(int(v)), ha="center")
        st.pyplot(fig)
        st.caption("Interprétation : classes relativement équilibrées (411 Non, 509 Oui).")
    with c2:
        fig, ax = plt.subplots()
        raw["num"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#72B7B2")
        ax.set_title("Diagnostic original (num 0-4, sévérité)")
        ax.set_xlabel("num")
        ax.set_ylabel("Nombre")
        st.pyplot(fig)
        st.caption("Interprétation : sévérités 1-4 déséquilibrées → justification de la cible binaire target.")

    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots()
        sns.histplot(data=data, x="age", hue="target", bins=20, kde=True, ax=ax, palette=["#4C78A8", "#F58518"])
        ax.set_title("Distribution de l'âge selon la cible")
        ax.set_xlabel("Âge (années)")
        st.pyplot(fig)
        st.caption("Interprétation : tendance à plus de cas « Oui » avec l'âge, distributions chevauchées.")
    with c4:
        fig, ax = plt.subplots()
        sns.histplot(raw["age"], bins=20, kde=True, ax=ax, color="#54A24B")
        ax.axvline(raw["age"].mean(), color="red", linestyle="--", label=f"Moyenne = {raw['age'].mean():.1f}")
        ax.axvline(raw["age"].median(), color="blue", linestyle="-", label=f"Médiane = {raw['age'].median():.1f}")
        mode_val = raw["age"].mode().iloc[0]
        ax.axvline(mode_val, color="purple", linestyle=":", label=f"Mode = {mode_val}")
        ax.set_title("Distribution de l'âge (moyenne, médiane, mode)")
        ax.set_xlabel("Âge (années)")
        ax.legend()
        st.pyplot(fig)
        st.caption("Interprétation : indicateurs de tendance centrale pour comprendre la structure d'âge (utile en EDA, comme dans les notebooks de référence).")

    c5, c6 = st.columns(2)
    with c5:
        fig, ax = plt.subplots()
        for sex, grp in raw.groupby("sex"):
            ax.hist(grp["age"], bins=15, alpha=0.5, label=str(sex))
        ax.set_title("Distribution de l'âge par sexe")
        ax.set_xlabel("Âge (années)")
        ax.set_ylabel("Effectif")
        ax.legend()
        st.pyplot(fig)
        st.caption("Interprétation : répartition des âges selon le sexe (démarche EDA des notebooks de référence).")
    with c6:
        fig, ax = plt.subplots()
        sns.countplot(data=raw, x="dataset", hue=(raw["num"] > 0).astype(int), ax=ax)
        ax.set_title("Répartition par centre (dataset) et maladie")
        ax.tick_params(axis='x', rotation=20)
        ax.legend(title="Maladie", labels=["Non (0)", "Oui (1)"])
        st.pyplot(fig)
        st.caption("Interprétation : hétérogénéité multi-centres (volumes et profils différents), à considérer dans les limites.")

    c5b, c6b = st.columns(2)
    with c5b:
        fig, ax = plt.subplots()
        sns.histplot(data=raw, x="age", hue="dataset", bins=20, multiple="stack", ax=ax)
        ax.set_title("Distribution de l'âge par centre (dataset)")
        ax.set_xlabel("Âge (années)")
        st.pyplot(fig)
        st.caption("Interprétation : profils d'âge différents selon les centres (inspiré des notebooks de référence).")
    with c6b:
        fig, ax = plt.subplots()
        sns.countplot(data=raw, x="dataset", hue="sex", ax=ax)
        ax.set_title("Répartition par centre (dataset) et sexe")
        ax.tick_params(axis='x', rotation=20)
        st.pyplot(fig)
        st.caption("Interprétation : composition par sexe variable selon les centres (biais de représentativité possible).")

    c7, c8 = st.columns(2)
    with c7:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.boxplot(data=data, x="target", y="thalch", ax=axes[0], palette=["#4C78A8", "#F58518"])
        axes[0].set_title("thalch vs cible")
        axes[0].set_xticklabels(["Non", "Oui"])
        sns.boxplot(data=data, x="target", y="oldpeak", ax=axes[1], palette=["#4C78A8", "#F58518"])
        axes[1].set_title("oldpeak vs cible")
        axes[1].set_xticklabels(["Non", "Oui"])
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Interprétation : thalch plus faible et oldpeak plus élevé chez les patients « Oui ».")
    with c8:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.boxplot(data=data, x="target", y="trestbps", ax=axes[0], palette=["#4C78A8", "#F58518"])
        axes[0].set_title("trestbps vs cible")
        axes[0].set_xticklabels(["Non", "Oui"])
        sns.boxplot(data=data, x="target", y="chol", ax=axes[1], palette=["#4C78A8", "#F58518"])
        axes[1].set_title("chol vs cible")
        axes[1].set_xticklabels(["Non", "Oui"])
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Interprétation : différences moins marquées individuellement, mais variables utiles en combinaison.")

    c9, c10 = st.columns(2)
    with c9:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(data=raw, x="cp", hue=(raw["num"] > 0).astype(int), ax=ax)
        ax.set_title("Type de douleur thoracique (cp) vs maladie")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title="Maladie", labels=["Non (0)", "Oui (1)"])
        st.pyplot(fig)
        st.caption("Interprétation : 'asymptomatic' très associé aux cas « Oui » dans ce dataset.")
    with c10:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(data=raw, x="cp", hue="sex", ax=ax)
        ax.set_title("Type de douleur thoracique (cp) par sexe")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)
        st.caption("Interprétation : croisement cp/sexe (démarche exploratoire complémentaire).")

    c9b, c10b = st.columns(2)
    with c9b:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(data=raw, x="cp", hue="dataset", ax=ax)
        ax.set_title("Type de douleur thoracique (cp) par centre (dataset)")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)
        st.caption("Interprétation : profils de symptômes variables selon les centres (référence notebooks).")
    with c10b:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(data=raw, x="trestbps", kde=True, hue="sex", ax=ax, element="step")
        ax.set_title("Distribution de trestbps par sexe")
        ax.set_xlabel("Pression artérielle au repos (mm Hg)")
        st.pyplot(fig)
        st.caption("Interprétation : distribution de la pression au repos, comparée entre sexes (référence notebooks).")

    c11, c12 = st.columns(2)
    with c11:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(data=raw, x="age", hue="cp", bins=20, multiple="stack", ax=ax)
        ax.set_title("Distribution de l'âge par type de douleur (cp)")
        st.pyplot(fig)
        st.caption("Interprétation : répartition des profils de douleur selon l'âge.")
    with c12:
        sex_counts = raw["sex"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(sex_counts.values, labels=sex_counts.index, autopct="%1.1f%%", startangle=90, colors=["#72B7B2", "#E45756"])
        ax.set_title("Répartition par sexe (camembert)")
        st.pyplot(fig)
        st.caption("Interprétation : majorité masculine (~79%), biais potentiel de représentativité.")

    st.subheader("Analyse des distributions numériques (détection visuelle d'outliers)")
    numeric_cols_raw = ["age", "trestbps", "chol", "thalch", "oldpeak"]
    fig, axes = plt.subplots(1, len(numeric_cols_raw), figsize=(16, 4))
    for ax, col in zip(axes, numeric_cols_raw):
        sns.boxplot(y=raw[col], ax=ax, color="#B3CDE3")
        ax.set_title(col)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("Interprétation : visualisation des dispersions/valeurs extrêmes (démarche inspirée des notebooks de référence). Les valeurs 0 de chol/trestbps sont traitées comme invalides.")

    c13, c14 = st.columns(2)
    with c13:
        miss = raw.isna().sum().sort_values(ascending=True)
        miss = miss[miss > 0]
        fig, ax = plt.subplots(figsize=(7, 4))
        miss.plot(kind="barh", ax=ax, color="#E45756")
        ax.set_title("Valeurs manquantes par variable (brut)")
        ax.set_xlabel("Nombre")
        st.pyplot(fig)
        st.caption("Interprétation : ca, thal, slope très incomplets.")
    with c14:
        miss_a = data.isna().sum().sort_values(ascending=True)
        miss_a = miss_a[miss_a > 0]
        fig, ax = plt.subplots(figsize=(7, 4))
        miss_a.plot(kind="barh", ax=ax, color="#F58518")
        ax.set_title("Valeurs manquantes après correction 0→NaN")
        ax.set_xlabel("Nombre")
        st.pyplot(fig)
        st.caption("Interprétation : augmentation volontaire sur chol/trestbps car les 0 invalides sont convertis en manquants avant imputation.")

elif page == "Corrélation":
    st.header("Matrice de corrélation et relations avec la cible")
    st.write("Analyse exploratoire des corrélations (valeurs affichées) et associations avec `target` (Oui/Non).")

    corr_df = data.copy()
    for c in corr_df.select_dtypes(exclude=["number"]).columns:
        corr_df[c] = corr_df[c].astype("category").cat.codes.replace(-1, np.nan)
    corr = corr_df.drop(columns=["id", "num", "dataset"], errors="ignore").corr(numeric_only=True)

    st.subheader("Matrice de corrélation (avec valeurs)")
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.3, ax=ax)
    ax.set_title("Matrice de corrélation (Pearson, exploratoire)")
    st.pyplot(fig)
    st.caption("Interprétation : lecture des coefficients (-1 à +1). Encodage numérique des catégories à des fins exploratoires ; corrélation ≠ causalité.")

    st.subheader("Corrélation des variables avec la cible (target)")
    target_corr = corr["target"].drop(labels=["target"], errors="ignore").sort_values(key=lambda s: s.abs(), ascending=False)
    tc_df = target_corr.to_frame("corr_avec_target")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.dataframe(tc_df.style.format("{:.3f}"), use_container_width=True)
    with c2:
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = ["#F58518" if v > 0 else "#4C78A8" for v in target_corr.sort_values().values]
        target_corr.sort_values().plot(kind="barh", ax=ax, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Corrélation des variables avec target (triées)")
        ax.set_xlabel("Coefficient de corrélation")
        st.pyplot(fig)

    st.markdown(
        """
        **Interprétation :** associations les plus fortes (|corr|) typiquement `exang`, `ca`, `thalch`, `oldpeak`, `cp`, `slope`, etc. (cohérentes médicalement en tendance).  
        **Précaution :** variables de natures différentes (numériques/catégorielles) → l'encodage peut influencer les signes (ex. `cp`, `slope`). Méthodes adaptées recommandées : Pearson/Spearman pour numériques, Chi-deux & V de Cramér pour catégorielles. **Corrélation ≠ causalité.**
        """
    )

elif page == "Modèles & Métriques":
    st.header("Modélisation, comparaison et évaluation")
    st.write("Baseline + modèles comparés, entraînés sur train (80%) et évalués sur test (20%), `random_state=42`, pipeline anti-fuite.")

    st.subheader("Tableau comparatif des métriques")
    st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

    best_auc_name = ml_metrics_global["ROC-AUC"].idxmax() if not ml_metrics_global.empty and ml_metrics_global["ROC-AUC"].notna().any() else None
    st.success(
        f"Meilleur F1 (hors baseline) : **{best_f1_name}** ({metrics_df.loc[best_f1_name, 'F1']:.4f}). "
        + (f"Meilleur ROC-AUC : **{best_auc_name}** ({metrics_df.loc[best_auc_name, 'ROC-AUC']:.4f}). " if best_auc_name else "")
        + "Les modèles appris doivent dépasser la baseline."
    )

    st.subheader("Comparaison des modèles (graphique)")
    comp = metrics_df[["Accuracy", "Précision", "Rappel", "F1", "ROC-AUC"]].astype(float)
    fig, ax = plt.subplots(figsize=(11, 5))
    comp.plot(kind="bar", ax=ax)
    ax.set_title("Comparaison des modèles sur le jeu de test")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', rotation=15)
    ax.legend(loc="lower right")
    st.pyplot(fig)
    st.caption("Interprétation : comparaison visuelle des performances (baseline vs modèles). En dépistage, le rappel et le F1 sont particulièrement importants.")

    model_name = st.selectbox("Détail du modèle", list(results.keys()), index=list(results.keys()).index(best_f1_name))
    res = results[model_name]
    yt, yp, yprob = res["y_test"], res["y_pred"], res["y_proba"]

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        cm = confusion_matrix(yt, yp)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Prédit Non (0)", "Prédit Oui (1)"], yticklabels=["Réel Non (0)", "Réel Oui (1)"])
        ax.set_title(f"Matrice de confusion — {model_name}")
        ax.set_ylabel("Vérité terrain")
        ax.set_xlabel("Prédiction")
        st.pyplot(fig)
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}. FN = malades non détectés (risque critique).")
    with col2:
        st.text("Rapport de classification")
        st.code(classification_report(yt, yp, target_names=["Non (0)", "Oui (1)"], zero_division=0))

    st.subheader("Courbes ROC (comparaison)")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, r in results.items():
        try:
            fpr, tpr, _ = roc_curve(r["y_test"], r["y_proba"])
            auc = roc_auc_score(r["y_test"], r["y_proba"])
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], "k--", label="Hasard (AUC=0.5)")
    ax.set_xlabel("Taux de faux positifs (FPR)")
    ax.set_ylabel("Taux de vrais positifs (TPR / Rappel)")
    ax.set_title("Courbes ROC")
    ax.legend(loc="lower right")
    st.pyplot(fig)

    if model_name == "Random Forest":
        st.subheader("Importance des variables (Random Forest)")
        pipe = res["pipeline"]
        pre = pipe.named_steps["preprocess"]
        rf_model = pipe.named_steps["model"]
        ohe = pre.named_transformers_["cat"].named_steps["encoder"]
        cat_names = ohe.get_feature_names_out(cat_features)
        feature_names = np.concatenate([num_features, cat_names])
        importances = rf_model.feature_importances_
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=imp_df, x="importance", y="feature", ax=ax, palette="viridis")
        ax.set_title("Top 15 variables importantes")
        st.pyplot(fig)
        st.caption("Interprétation : variables qui contribuent le plus aux décisions (importance relative, pas causalité).")

    st.markdown("**Métriques affichées :** Accuracy, Précision, Rappel, F1, Matrice de confusion (et ROC-AUC) pour la classification ; MAE, MSE, RMSE, R² en complément.")

elif page == "Test (réel vs prédit)":
    st.header("Cas de test : valeurs réelles vs prédites")
    st.write("Comparaison sur le jeu de test (données non vues) pour analyser les prédictions correctes et les erreurs.")

    model_name = st.selectbox("Modèle", list(results.keys()), index=list(results.keys()).index(best_f1_name))
    res = results[model_name]
    yt = res["y_test"].reset_index(drop=True)
    yp = pd.Series(res["y_pred"]).reset_index(drop=True)
    yprob = pd.Series(res["y_proba"]).reset_index(drop=True)
    X_test_reset = res["X_test"].reset_index(drop=True)

    comp = pd.DataFrame({
        "valeur_reelle": yt.values,
        "valeur_predite": yp.values,
        "proba_maladie_Oui": np.round(yprob.values, 4),
    })
    comp["réel"] = comp["valeur_reelle"].map({0: "Non (0)", 1: "Oui (1)"})
    comp["prédit"] = comp["valeur_predite"].map({0: "Non (0)", 1: "Oui (1)"})
    comp["correct"] = comp["valeur_reelle"] == comp["valeur_predite"]
    comp["analyse"] = np.where(
        comp["correct"], "Correct",
        np.where((comp["valeur_reelle"] == 1) & (comp["valeur_predite"] == 0), "Faux négatif (malade prédit Non)", "Faux positif (sain prédit Oui)")
    )

    detail_cols = ["age", "sex", "cp", "thalch", "oldpeak", "exang"]
    detail = pd.concat([X_test_reset[detail_cols], comp[["réel", "prédit", "proba_maladie_Oui", "analyse"]]], axis=1)
    st.subheader(f"Tableau comparatif (exemples) — {model_name}")
    st.dataframe(detail.head(25), use_container_width=True)

    total = len(comp)
    correct = int(comp["correct"].sum())
    fn = int(((comp["valeur_reelle"] == 1) & (comp["valeur_predite"] == 0)).sum())
    fp = int(((comp["valeur_reelle"] == 0) & (comp["valeur_predite"] == 1)).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients testés", total)
    c2.metric("Prédictions correctes", f"{correct} ({correct/total:.1%})")
    c3.metric("Faux négatifs (FN)", fn)
    c4.metric("Faux positifs (FP)", fp)

    err_counts = comp["analyse"].value_counts().reindex(["Correct", "Faux négatif (malade prédit Non)", "Faux positif (sain prédit Oui)"], fill_value=0)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.barh(err_counts.index, err_counts.values, color=["#54A24B", "#E45756", "#F58518"])
    ax.set_xlabel("Nombre de patients (test)")
    ax.set_title("Bilan des prédictions : corrects vs erreurs")
    st.pyplot(fig)
    st.markdown("**Interprétation :** les **FN** sont critiques en dépistage (maladie manquée) ; les **FP** génèrent des fausses alertes. La probabilité aide à jauger la confiance du modèle.")

elif page == "Prédiction (scénario)":
    st.header("Prédiction pour un scénario patient (Oui/Non)")
    st.write("Saisissez les caractéristiques cliniques : le modèle renvoie la classe prédite, la probabilité, un niveau de risque et une alerte (aide à la décision).")
    show_target_explainer()

    with st.expander("Aide : comprendre le jargon des variables"):
        st.dataframe(pd.DataFrame(JARGON_GLOSSARY, columns=["Variable / Terme", "Explication simple"]), use_container_width=True, hide_index=True)

    ml_only = [m for m in results.keys() if m != "Baseline (classe majoritaire)"]
    default_pred_model = metrics_df.loc[ml_only, "F1"].idxmax() if ml_only else list(results.keys())[0]
    model_name = st.selectbox("Modèle utilisé", list(results.keys()), index=list(results.keys()).index(default_pred_model))
    pipe = results[model_name]["pipeline"]

    with st.form("pred_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Âge (années)", 28, 90, 54)
            sex = st.selectbox("Sexe", ["Male", "Female"])
            cp = st.selectbox("Douleur thoracique (cp)", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
            trestbps = st.number_input("Pression repos (trestbps, mm Hg)", 80, 220, 130)
            chol = st.number_input("Cholestérol (chol, mg/dl)", 100, 600, 240)
        with c2:
            fbs = st.selectbox("Glycémie à jeun > 120 mg/dl (fbs)", [False, True])
            restecg = st.selectbox("ECG repos (restecg)", ["normal", "lv hypertrophy", "st-t abnormality"])
            thalch = st.number_input("Fréquence cardiaque max (thalch)", 60, 220, 150)
            exang = st.selectbox("Angine d'effort (exang)", [False, True])
        with c3:
            oldpeak = st.number_input("Oldpeak (dépression ST)", -2.5, 7.0, 1.0, step=0.1)
            slope = st.selectbox("Pente ST (slope)", ["upsloping", "flat", "downsloping"])
            ca = st.number_input("Nb vaisseaux colorés (ca)", 0.0, 3.0, 0.0, step=1.0)
            thal = st.selectbox("Thalassémie/perfusion (thal)", ["normal", "fixed defect", "reversable defect"])
        submitted = st.form_submit_button("Prédire (Oui/Non)")

    if submitted:
        row = pd.DataFrame([{
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalch": thalch, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
        }])
        pred = pipe.predict(row)[0]
        proba = pipe.predict_proba(row)[0][1] if hasattr(pipe, "predict_proba") else float(pred)

        if proba < 0.33:
            niveau_risque = "Faible"
        elif proba < 0.66:
            niveau_risque = "Modéré"
        else:
            niveau_risque = "Élevé"

        c1, c2, c3 = st.columns(3)
        c1.metric("Classe prédite", "Oui (1)" if pred == 1 else "Non (0)")
        c2.metric("Probabilité de maladie (Oui)", f"{proba:.2%}")
        c3.metric("Niveau de risque estimé", niveau_risque)

        if pred == 1 or niveau_risque == "Élevé":
            st.error(f"Alerte : risque **{niveau_risque.lower()}** — prédiction **{'Oui' if pred==1 else 'Non'}** ({proba:.2%}). Aide à la décision uniquement, **vérification humaine/médicale indispensable**.")
            st.write("**Explication :** le profil se rapproche des cas avec maladie (« Oui ») observés à l'entraînement. Des examens et un avis médical sont nécessaires.")
        elif niveau_risque == "Modéré":
            st.warning(f"Attention : risque **modéré** — prédiction **{'Oui' if pred==1 else 'Non'}** ({proba:.2%}). Aide à la décision ; évaluation clinique recommandée en cas de symptômes.")
            st.write("**Explication :** profil intermédiaire ; le modèle estime une probabilité moyenne, à interpréter avec prudence.")
        else:
            st.success(f"Risque estimé **faible** — prédiction **{'Oui' if pred==1 else 'Non'}** ({proba:.2%}). Aide à la décision, pas un diagnostic.")
            st.write("**Explication :** profil plus proche des cas sans maladie (« Non ») ; en cas de symptômes, consulter reste nécessaire.")

        st.markdown("**Récapitulatif du scénario saisi :**")
        st.dataframe(row, use_container_width=True)
        st.info("Une alerte/prédiction automatique n'est qu'une aide à la décision : **vérification humaine** obligatoire.")

elif page == "Interprétation & Éthique":
    st.header("Interprétation, limites, recommandations et éthique")
    best_row = metrics_df.loc[best_f1_name]
    st.subheader("Synthèse des performances")
    st.write(f"Meilleur modèle (F1, hors baseline) : **{best_f1_name}** — F1={best_row['F1']:.3f}, Accuracy={best_row['Accuracy']:.3f}, Rappel={best_row['Rappel']:.3f}, Précision={best_row['Précision']:.3f}, ROC-AUC={best_row['ROC-AUC']:.3f}.")
    st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

    st.subheader("Comment interpréter")
    st.markdown(
        """
        - **Baseline** : référence minimale. **Rappel** : détection des « Oui » (critique pour limiter les FN). **Précision** : fiabilité des « Oui » prédits. **F1/ROC-AUC** : vision globale.
        - **MAE/MSE/RMSE/R²** : indicateurs complémentaires sur les probabilités (calibration/écart), à lire avec les métriques de classification.
        - **Erreurs** : FN = malade non détecté (risque majeur) ; FP = fausse alerte.
        """
    )

    st.subheader("Corrélations (rappel)")
    st.markdown(
        """
        - Associations notables observées (ex. `exang`, `ca`, `thalch`, `oldpeak`, `cp`) mais **corrélation ≠ causalité**.
        - L'encodage des variables catégorielles peut influencer les signes ; idéalement compléter par Chi-deux/V de Cramér et Spearman selon les types.
        - Ne pas exclure d'office les variables faiblement corrélées individuellement (`chol`, `trestbps`, `restecg`, `fbs`).
        """
    )

    st.subheader("Limites")
    st.markdown("- Données multi-centres, 920 patients, manquants importants (`ca`, `thal`, `slope`), 0 invalides corrigés.  \n- Dataset ancien, biais de représentativité, pas de validation externe ; cible binarisée ; généralisation limitée.")

    st.subheader("Recommandations concrètes")
    st.markdown("1. **Aide au pré-triage uniquement** avec validation humaine systématique, en privilégiant la réduction des faux négatifs.  \n2. **Améliorer la qualité/représentativité des données** (compléter manquants, données récentes multi-populations, suivi des biais).  \n3. *(Perspective)* Validation croisée, optimisation, calibration, analyses d’association adaptées (Chi-deux/V de Cramér), explicabilité (SHAP), validation externe.")

    st.subheader("Éthique et responsabilité")
    st.markdown("- **Conséquences d'une erreur** : FN → retard de prise en charge ; FP → stress/examens inutiles.  \n- **Risques** : biais, usage inapproprié, atteinte possible à l'image/libertés.  \n- **Vérification humaine** : indispensable (aide à la décision, pas décideur autonome).  \n- **Données** : publiques, anonymisées (UCI), usage pédagogique.")

    st.warning("CardioRisk Studio est un outil pédagogique d'aide à l'analyse. Il ne constitue pas un dispositif médical et ne remplace pas un diagnostic professionnel.")
