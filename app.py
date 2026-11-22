import base64
import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.patches import FancyBboxPatch

st.set_page_config(page_title="CardioRisk Studio", page_icon="🫀", layout="wide")
sns.set_theme(style="whitegrid")

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model.joblib"
META_PATH = ROOT / "model_meta.json"
DATA_PATH = ROOT / "heart_disease_uci.csv"
NOTEBOOK_PRIORITY = ROOT / "heart_disease_cardiorisk.ipynb"
GITHUB_URL = "https://github.com/bennoomenfaker/cardiorisk-studio"

TEAM = [
    "Faker BEN NOOMEN",
    "Haythem BEN BOUBAKER",
    "Mariem Dridi",
]

FIELD_INFO = {
    "age": ("Âge (années)", "Âge du patient. Plage plausible 18-100."),
    "trestbps": ("Pression artérielle au repos (mm Hg)", "Tension systolique au repos. Plage plausible 70-250 mm Hg."),
    "chol": ("Cholestérol sérique (mg/dl)", "Cholestérol total en mg/dl. Si résultat en mmol/L, multiplier par 38.7. Plage plausible 80-650."),
    "thalch": ("Fréquence cardiaque maximale atteinte (bpm)", "Fréquence max à l'effort. Plage plausible 50-220 bpm."),
    "oldpeak": ("Dépression ST à l'effort (oldpeak)", "Dépression ST induite par l'effort. Plage plausible -3 à 7."),
    "ca": ("Nombre de vaisseaux majeurs colorés (0-3)", "Examen fluoroscopie, entier 0-3."),
    "sex": ("Sexe", "Variable démographique/catégorielle."),
    "cp": ("Type de douleur thoracique", "Symptôme clé. `asymptomatic` peut être associé à maladie silencieuse."),
    "fbs": ("Glycémie à jeun > 120 mg/dl", "Indicateur glycémique."),
    "restecg": ("ECG au repos", "Résultat électrocardiogramme au repos."),
    "exang": ("Angine induite par l'effort", "Douleur à l'effort."),
    "slope": ("Pente du segment ST à l'effort", "Interprétation ECG effort."),
    "thal": ("Thalassémie/scintigraphie", "Résultat examen perfusion/thalassémie."),
}

VAR_RANGES = {
    "age": {"lo": 18, "hi": 100, "unit": "années"},
    "trestbps": {"lo": 70, "hi": 250, "unit": "mm Hg"},
    "chol": {"lo": 80, "hi": 650, "unit": "mg/dl"},
    "thalch": {"lo": 50, "hi": 220, "unit": "bpm"},
    "oldpeak": {"lo": -3.0, "hi": 7.0, "unit": ""},
    "ca": {"lo": 0, "hi": 3, "unit": "vaisseaux"},
}

def validate_patient_inputs(patient: dict):
    errors, warnings = [], []
    for var, rule in VAR_RANGES.items():
        val = patient.get(var)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        try:
            v = float(val)
        except Exception:
            errors.append(f"{var}: valeur non numérique.")
            continue
        if v < rule["lo"] or v > rule["hi"]:
            label = FIELD_INFO.get(var, (var, ""))[0]
            errors.append(f"{label} = {v:g} hors plage plausible ({rule['lo']:g} à {rule['hi']:g} {rule['unit']}).")
    age = float(patient.get("age", 0) or 0)
    thalch = float(patient.get("thalch", 0) or 0)
    if age > 0 and thalch > (220 - age) + 15:
        warnings.append(f"`thalch` ({thalch:g} bpm) très élevé pour l'âge {age:g} (repère ~{220-age:g} bpm).")
    return errors, warnings


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["target"] = (df["num"] > 0).astype(int)
    return df


def render_notebook(path: Path):
    if not path.exists():
        st.warning(f"Notebook introuvable : {path.name}")
        return
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    st.caption(f"{path.name} — {len(cells)} cellules")
    with open(path, "rb") as f:
        st.download_button("Télécharger le notebook (.ipynb)", f, file_name=path.name, mime="application/x-ipynb+json")
    for i, cell in enumerate(cells):
        ctype = cell.get("cell_type")
        src = "".join(cell.get("source", []))
        if ctype == "markdown":
            st.markdown(src)
        elif ctype == "code":
            st.markdown(f"**Code cellule {i+1}**")
            st.code(src if src.strip() else "# (cellule vide)", language="python")
            for out in cell.get("outputs", []):
                otype = out.get("output_type")
                data = out.get("data", {})
                if otype == "stream":
                    txt = "".join(out.get("text", []))
                    if txt.strip():
                        st.text(txt)
                elif otype in {"execute_result", "display_data"}:
                    if "text/html" in data:
                        st.components.v1.html("".join(data["text/html"]), height=320, scrolling=True)
                    elif "image/png" in data:
                        img = data["image/png"]
                        img = "".join(img) if isinstance(img, list) else img
                        st.image(base64.b64decode(img))
                    elif "text/plain" in data:
                        txt = "".join(data["text/plain"])
                        if txt.strip() and "<" not in txt[:1]:
                            st.text(txt)
                elif otype == "error":
                    st.error("\n".join(out.get("traceback", [])) or out.get("evalue", "Erreur"))
        else:
            st.info(f"Cellule {ctype} non affichée")


@st.cache_data
def get_notebook_process_image():
    """Use same process image as notebook (cell with FancyBboxPatch/Données brutes) if available."""
    if not NOTEBOOK_PRIORITY.exists():
        return None
    try:
        nb = json.loads(NOTEBOOK_PRIORITY.read_text(encoding="utf-8"))
        for cell in nb.get("cells", []):
            src = "".join(cell.get("source", []))
            if cell.get("cell_type") == "code" and "Données brutes" in src and "FancyBboxPatch" in src:
                for out in cell.get("outputs", []):
                    data = out.get("data", {})
                    if "image/png" in data:
                        img = data["image/png"]
                        img = "".join(img) if isinstance(img, list) else img
                        return base64.b64decode(img)
                # generate from notebook code if no output
                local = {"plt": plt, "FancyBboxPatch": FancyBboxPatch}
                exec(compile(src, "<nb_process>", "exec"), local)
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=160, bbox_inches="tight")
                plt.close("all")
                return buf.getvalue()
    except Exception:
        return None
    return None


@st.cache_data
def process_figure():
    img = get_notebook_process_image()
    if img:
        return img
    # fallback
    steps = [
        ("1  Données brutes", "CSV UCI / Kaggle"),
        ("2  Audit & nettoyage", "NaN, zéros invalides,\ncible binaire"),
        ("3  Exploration (EDA)", "distributions,\ncorrélations"),
        ("4  Split train / test", "80 % / 20 %\nstratifié"),
        ("5  Pipeline & modèles", "imputation → encodage\n→ scaling → modèle"),
        ("6  Évaluation", "AUC, rappel,\nmatrice de confusion"),
        ("7  Seuil & déploiement", "seuil métier\n+ app Streamlit"),
        ("8  Export", "model.joblib\nmodel_meta.json"),
    ]
    fig, ax = plt.subplots(figsize=(13, 2.8))
    ax.axis("off")
    for i, (title, sub) in enumerate(steps):
        x = i * 1.45
        rect = FancyBboxPatch((x, 0.35), 1.25, 0.85, boxstyle="round,pad=0.02,rounding_size=0.08",
                              linewidth=1.2, edgecolor="#2c3e50", facecolor="#ecf0f1")
        ax.add_patch(rect)
        ax.text(x + 0.625, 0.95, title, ha="center", va="center", fontsize=8.5, fontweight="bold")
        ax.text(x + 0.625, 0.62, sub, ha="center", va="center", fontsize=7.5)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 1.42, 0.77), xytext=(x + 1.27, 0.77), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.set_xlim(-0.1, 11.7)
    ax.set_ylim(0, 1.4)
    ax.set_title("Processus ML — CardioRisk Studio", fontsize=12, fontweight="bold")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()


@st.cache_data
def correlation_figure(df: pd.DataFrame):
    cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca", "target"]
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Matrice de corrélation (variables numériques + cible)")
    fig.tight_layout()
    return fig, corr


@st.cache_resource
def train_compare_models():
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

    df_all = pd.read_csv(DATA_PATH)
    df_all.loc[df_all["chol"] == 0, "chol"] = np.nan
    df_all.loc[df_all["trestbps"] == 0, "trestbps"] = np.nan
    df_all["target"] = (df_all["num"] > 0).astype(int)
    NUM = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
    CAT = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
    for c in CAT:
        df_all[c] = df_all[c].astype(str).where(df_all[c].notna(), np.nan)
    Xall = df_all[NUM + CAT]
    yall = df_all["target"]
    Xtr, Xte, ytr, yte = train_test_split(Xall, yall, test_size=0.2, stratify=yall, random_state=42)
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT),
    ])
    defs = {
        "Régression logistique": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }
    fitted, rows, cms = {}, [], {}
    for n, m in defs.items():
        pipe = Pipeline([("pre", pre), ("model", m)])
        pipe.fit(Xtr, ytr)
        fitted[n] = pipe
        pred = pipe.predict(Xte)
        proba = pipe.predict_proba(Xte)[:, 1]
        cm = confusion_matrix(yte, pred)
        cms[n] = cm
        rows.append({"Modèle": n, "Accuracy": accuracy_score(yte,pred), "Précision": precision_score(yte,pred,zero_division=0), "Rappel": recall_score(yte,pred,zero_division=0), "F1": f1_score(yte,pred,zero_division=0), "AUC": roc_auc_score(yte,proba), "FN": int(cm[1,0]), "FP": int(cm[0,1])})
    return fitted, pd.DataFrame(rows).set_index("Modèle"), cms


model, meta = load_model()
df = load_data()

ORDER = ["Sain (0)", "Malade (1)"]
SCENARIOS = {
    "Profil faible risque": {"age":42,"trestbps":120,"chol":210,"thalch":170,"oldpeak":0.0,"ca":0.0,"sex":"Female","cp":"non-anginal","fbs":"False","restecg":"normal","exang":"False","slope":"upsloping","thal":"normal"},
    "Profil vigilance (limite)": {"age":55,"trestbps":140,"chol":245,"thalch":145,"oldpeak":1.0,"ca":0.0,"sex":"Male","cp":"atypical angina","fbs":"False","restecg":"normal","exang":"False","slope":"flat","thal":"normal"},
    "Profil risque élevé": {"age":63,"trestbps":150,"chol":280,"thalch":120,"oldpeak":2.3,"ca":2.0,"sex":"Male","cp":"asymptomatic","fbs":"True","restecg":"lv hypertrophy","exang":"True","slope":"flat","thal":"reversable defect"},
}

def predire_scenario(valeurs: dict, seuil_alerte: float | None = None):
    """Prédit la classe, les probabilités et le niveau d'alerte. Aide à la décision, pas décision automatique."""
    seuil_alerte = float(meta["seuil"]) if seuil_alerte is None else float(seuil_alerte)
    Xcols = list(meta["num"].keys()) + list(meta["cat"].keys())
    ligne = pd.DataFrame([valeurs])[Xcols]
    proba = model.predict_proba(ligne)[0]
    classe = ORDER[int(np.argmax(proba))]
    p_grave = float(proba[-1])
    if p_grave >= seuil_alerte:
        niveau = "ALERTE – vérification humaine prioritaire"
    elif p_grave >= seuil_alerte / 2:
        niveau = "VIGILANCE – à surveiller"
    else:
        niveau = "NORMAL"
    return {
        "classe_predite": classe,
        "probabilites": {c: round(float(p), 3) for c, p in zip(ORDER, proba)},
        "P(Malade)": round(p_grave, 3),
        "seuil": round(seuil_alerte, 4),
        "niveau": niveau,
    }

st.title("🫀 CardioRisk Studio")
st.caption("Prototype pédagogique d’aide à la décision — Heart Disease UCI")
st.markdown(f"**GitHub :** [{GITHUB_URL}]({GITHUB_URL})")
st.markdown("**Équipe :** " + " • ".join(TEAM))

page = st.sidebar.radio(
    "Navigation",
    [
        "1. Contexte & Problématique",
        "2. Processus ML (pas à pas)",
        "3. Prédiction / Verdict",
        "4. Notebook (prioritaire)",
    ],
)
st.sidebar.markdown("---")
st.sidebar.info("Outil pédagogique. Ne remplace pas un avis médical.")

# -------------------- PAGE 1 --------------------
if page.startswith("1"):
    st.header("1) Contexte, problématique et variables")

    st.subheader("Contexte général")
    st.markdown("""
Les maladies cardiovasculaires sont un enjeu majeur de santé publique. Les professionnels ont besoin d'un outil d'aide au **dépistage** (repérage précoce/priorisation des patients à risque) pour orienter vers un examen médical.

**Dépistage / pré-dépistage (définitions simples) :**
- **Dépistage** : repérer tôt les personnes qui ont probablement un risque, pour les orienter vers des examens/avis médical.
- **Pré-dépistage / triage** : étape encore plus en amont, rapide, pour prioriser qui doit être contrôlé en priorité.
- **Ce projet** : prototype pédagogique de pré-dépistage/dépistage assisté par données. Il **ne pose pas de diagnostic médical** et ne remplace pas un médecin.

**Domaine :** santé / aide à la décision / classification binaire.
""")

    st.subheader("Problématique")
    st.info("**Peut-on prédire la présence d’une maladie cardiaque à partir de données cliniques et d’examens, de manière fiable et explicable ?**")

    st.subheader("Questions de recherche")
    st.markdown("""
- **Q1.** Quels facteurs cliniques/examens sont associés à la maladie ?
- **Q2.** Quelles variables contribuent le plus à la prédiction ?
- **Q3.** Quelle est la capacité prédictive du modèle (AUC, rappel, métriques VIC) ?
- **Q4.** Quel modèle choisir (baseline, régression logistique, arbre, KNN, Random Forest) ?
- **Q5.** Le modèle est-il fiable pour le pré-dépistage ? Quel seuil de décision choisir pour limiter les malades manqués, et quel compromis accepter ?
""")

    st.subheader("Définitions simples (jargon)")
    st.markdown("""
| Terme | Définition simple |
|---|---|
| **Dépistage** | Repérer tôt les personnes potentiellement à risque pour les orienter vers un médecin/examen. |
| **Pré-dépistage / triage** | Filtre rapide en amont pour prioriser les personnes à contrôler. Ce n'est pas un diagnostic. |
| **Diagnostic** | Décision médicale faite par un professionnel avec examens cliniques. Le modèle ne fait pas ça. |
| **Probabilité** | Score entre 0% et 100% estimé par le modèle : plus c'est élevé, plus le profil ressemble aux cas malades vus à l'entraînement. |
| **Seuil de décision** | Règle de conversion probabilité → classe. Ex: si probabilité ≥ seuil alors "risque élevé". Le seuil est calculé, pas arbitraire. |
| **Faux négatif** | Malade prédit sain — dangereux en dépistage. |
| **Faux positif** | Sain prédit malade — fausse alerte, entraîne un contrôle supplémentaire. |
| **Recall/Rappel** | % de vrais malades détectés. Prioritaire ici. |
""")

    st.subheader("Processus ML (même schéma que le notebook)")
    proc_img = process_figure()
    st.image(proc_img, use_container_width=True)
    st.caption("Même image/schéma que le notebook CardioRisk : de la donnée brute à l’export (model.joblib/model_meta.json) puis l’app Streamlit.")

    st.subheader("Description des variables")
    var_rows = []
    for col, (label, meaning) in FIELD_INFO.items():
        kind = "Numérique" if col in meta["num"] else "Catégorielle"
        var_rows.append({"Variable": col, "Nom lisible": label, "Type": kind, "Description / utilité": meaning})
    var_rows.append({"Variable": "num / target", "Nom lisible": "Diagnostic cible", "Type": "Cible", "Description / utilité": "`num` 0-4 dans le dataset brut. On crée `target=1` si `num>0` (maladie), sinon `0` (sain)."})
    st.dataframe(pd.DataFrame(var_rows), use_container_width=True, hide_index=True)

    st.subheader("Logique : quand dit-on qu'une personne est malade ?")
    st.markdown("""
Dans ce projet, la sortie du modèle est une **probabilité estimée** de maladie (`target=1`).

- **Données passées au modèle** : les variables du formulaire (âge, pression, cholestérol, ECG, douleur thoracique, etc.) après le même prétraitement que l'entraînement.
- **Le modèle calcule** `P(malade | données patient)`.
- **Seuil de décision** : règle pour transformer une probabilité en décision.
  - Exemple par défaut : seuil 0.50 (50%). Mais ce n'est pas obligatoire.
  - Dans notre cas, on choisit un seuil **plus bas que 0.50** (calculé sur validation croisée du train, puis appliqué au test) pour **attraper plus de vrais malades**.
  - Conséquence : moins de malades manqués (**faux négatifs** ↓), mais plus de fausses alertes (**faux positifs** ↑). Ce compromis est acceptable en pré-dépistage, car une alerte peut mener à un contrôle médical.
- **Pourquoi un seuil plus bas ?** En dépistage, manquer un malade (faux négatif) est plus grave qu'une fausse alerte (faux positif). On privilégie donc le **rappel** (sensibilité = part de malades détectés). La valeur exacte du seuil doit être **calculée et justifiée**, pas choisie au hasard.
- **Verdict final** : toujours indicatif. **Validation humaine / avis médical obligatoire**.
""")

# -------------------- PAGE 2 --------------------
elif page.startswith("2"):
    st.header("2) Processus ML — pas à pas (détaillé)")

    st.markdown("""
Cette page suit le **même processus** que le notebook, avec des graphiques explicatifs et une interprétation pour chaque étape.
Pour chaque graphique : **nom / utilité / comment lire / interprétation**.
""")

    # --- Step 1 ---
    st.subheader("Étape 1 — Lecture et compréhension des données")
    st.markdown("""
**Action** : charger `heart_disease_uci.csv`, inspecter `head()`, `shape`, `info()`, `describe()`.
**Utilité** : comprendre le contenu réel avant toute modélisation ; repérer types, valeurs aberrantes, échelles.
**Comment lire** : `shape` = (lignes, colonnes) ; `describe` donne min/25%/50%/75%/max ; `info` donne types et non-nulls.
""")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", df.shape[0])
    c2.metric("Colonnes brutes", 16)
    c3.metric("Sains (0)", int((df["target"] == 0).sum()))
    c4.metric("Malades (1)", int((df["target"] == 1).sum()))
    st.dataframe(df.head(8), use_container_width=True)
    with st.expander("Statistiques descriptives (describe)"):
        st.dataframe(df.describe(include="all").T, use_container_width=True)
        st.caption("Lecture : count faible = valeurs manquantes ; écart mean/50% = asymétrie ; min/max extrêmes = outliers ou erreurs.")

    # --- Step 2 ---
    st.subheader("Étape 2 — Détection qualité : NA, doublons, valeurs invalides")
    st.markdown("""
**Action** : `isna().sum()`, `duplicated().sum()`, vérifier `chol==0`, `trestbps==0`.
**Utilité** : décider stratégie de nettoyage. Sans ça, le modèle apprend sur du bruit ou plante sur NaN.
**Règle** : 0 pour cholestérol/tension au repos est médicalement impossible → traité comme NaN.
""")
    dup = int(df.duplicated().sum())
    chol0 = int((pd.read_csv(DATA_PATH)["chol"] == 0).sum())
    tre0 = int((pd.read_csv(DATA_PATH)["trestbps"] == 0).sum())
    k1,k2,k3 = st.columns(3)
    k1.metric("Doublons", dup)
    k2.metric("chol == 0 (invalide)", chol0)
    k3.metric("trestbps == 0 (invalide)", tre0)

    miss = df.isna().sum().sort_values(ascending=True)
    miss = miss[miss > 0]
    fig_m, ax_m = plt.subplots(figsize=(8, 4))
    miss.plot(kind="barh", ax=ax_m, color="#4C78A8")
    ax_m.set_title("Valeurs manquantes par colonne")
    ax_m.set_xlabel("Nombre de NaN")
    st.pyplot(fig_m); plt.close(fig_m)
    st.caption("**Nom** : barplot horizontal des valeurs manquantes. **Lecture** : longueur = nombre de NaN. **Interprétation** : `ca/thal/slope` très incomplètes → imputation + indicateur de manquant, pas suppression massive.")

    # --- Step 3 ---
    st.subheader("Étape 3 — Préparation / nettoyage / preprocessing")
    st.markdown("""
**Utilité** : rendre les données utilisables par les modèles et éviter la fuite de données.

**SimpleImputer — c'est quoi ?**  
`SimpleImputer` remplace les valeurs manquantes (`NaN`) par une valeur simple :
- `strategy="median"` (numérique) : prend la **médiane**. Plus robuste que la moyenne quand il y a des valeurs extrêmes/outliers.
- `strategy="most_frequent"` (catégoriel) : prend le **mode** (valeur la plus fréquente). On ne peut pas faire une moyenne sur du texte.
- `add_indicator=True` : ajoute une colonne 0/1 "était manquant ?" pour garder l'information de manque (utile pour `ca`).

**ColumnTransformer — c'est quoi ?**  
Il applique des traitements différents aux colonnes numériques et catégorielles, puis recolle le résultat.

**Version claire (pipeline scikit-learn, utilisée dans le projet)** :
""")
    st.code("""NUM = ["age","trestbps","chol","thalch","oldpeak","ca"]
CAT = ["sex","cp","fbs","restecg","exang","slope","thal"]

prep = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median", add_indicator=True)),  # NaN -> médiane + indicateur
        ("sc", StandardScaler())                                       # même échelle
    ]), NUM),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),              # NaN -> mode
        ("oh", OneHotEncoder(handle_unknown="ignore"))                 # texte -> colonnes 0/1
    ]), CAT),
])""", language="python")
    st.markdown("""
**Version plus simple / plus lisible (même idée, sans ColumnTransformer)** :
""")
    st.code("""# 1) Copier les données
X = df[NUM + CAT].copy()

# 2) Numériques : remplacer NaN par médiane
for col in NUM:
    X[col] = X[col].fillna(X[col].median())

# 3) Catégorielles : remplacer NaN par mode (valeur la plus fréquente)
for col in CAT:
    X[col] = X[col].fillna(X[col].mode()[0])

# 4) Encoder catégories en 0/1
X = pd.get_dummies(X, columns=CAT, drop_first=False)

# 5) Mettre numériques à la même échelle
X[NUM] = StandardScaler().fit_transform(X[NUM])""", language="python")
    st.warning("La version simple est bonne pour comprendre. Dans le projet, on préfère la version **Pipeline/ColumnTransformer** car elle évite la fuite de données: le `fit` (médiane/mode/scale/encoder) se fait **sur le train uniquement**, puis on applique au test/app.")

    st.info("""**Export fin de processus** : `model.joblib` (pipeline entraîné) + `model_meta.json` (seuil calculé, bornes, catégories, citation). 
L'app Streamlit charge `joblib` directement avec `joblib.load(...)` — pas besoin de réentraîner à chaque prédiction.""")

    # --- Step 4 EDA ---
    st.subheader("Étape 4 — Exploration (EDA) : graphiques explicatifs")
    st.markdown("**Utilité** : comprendre les relations variable-cible avant modélisation. Chaque graphique répond à une question.")

    # target distribution
    fig_t, ax_t = plt.subplots(figsize=(5,3.5))
    counts = df["target"].value_counts().sort_index()
    ax_t.bar(["Sain (0)", "Malade (1)"], [counts.get(0,0), counts.get(1,0)], color=["#4C78A8","#E45756"])
    ax_t.set_title("Distribution de la cible")
    ax_t.set_ylabel("Nombre de patients")
    for i,v in enumerate([counts.get(0,0), counts.get(1,0)]): ax_t.text(i,v+5,str(int(v)),ha="center")
    st.pyplot(fig_t); plt.close(fig_t)
    st.caption("**Nom** : countplot/barplot cible. **Utilité** : vérifier équilibre des classes. **Interprétation** : classes assez équilibrées (~45/55) ; accuracy utilisable mais on surveille aussi Recall/F1.")

    # age by target
    fig_age, axes_age = plt.subplots(1,2,figsize=(11,4))
    for t,label,color in [(0,'Sain (0)','#4C78A8'),(1,'Malade (1)','#E45756')]:
        axes_age[0].hist(df.loc[df['target']==t,'age'].dropna(), bins=15, alpha=0.5, label=label, color=color)
    axes_age[0].set_title('Âge par cible (histogramme)'); axes_age[0].set_xlabel('Âge'); axes_age[0].set_ylabel('Nombre'); axes_age[0].legend()
    sns.boxplot(data=df, x='target', y='age', ax=axes_age[1], palette=['#4C78A8','#E45756'])
    axes_age[1].set_title('Âge par cible (boxplot)'); axes_age[1].set_xticks([0,1]); axes_age[1].set_xticklabels(['Sain (0)','Malade (1)'])
    plt.tight_layout(); st.pyplot(fig_age); plt.close(fig_age)
    st.caption("**Nom** : histogramme superposé + boxplot âge/target. **Utilité** : voir si l'âge sépare les classes. **Lecture** : histogramme = distribution ; boxplot = médiane/quartiles/outliers. **Interprétation** : malades légèrement plus âgés, fort chevauchement => âge informatif mais pas suffisant seul.")

    # numeric boxplots
    NUM = ["age","trestbps","chol","thalch","oldpeak","ca"]
    fig_b, axes_b = plt.subplots(2,3,figsize=(13,6))
    for ax,col in zip(axes_b.ravel(), NUM):
        sns.boxplot(data=df, x='target', y=col, ax=ax, palette=['#4C78A8','#E45756'])
        ax.set_title(col); ax.set_xticks([0,1]); ax.set_xticklabels(['Sain','Malade'])
    plt.tight_layout(); st.pyplot(fig_b); plt.close(fig_b)
    st.caption("**Nom** : boxplots numériques par cible. **Utilité** : comparer médiane, dispersion, outliers entre sains/malades. **Lecture** : boîte=Q1-Q3, trait=médiane, points=outliers. **Interprétation** : `thalch` plus bas chez malades, `oldpeak/ca` plus élevés ; `chol/trestbps` se chevauchent beaucoup.")

    # categorical rates
    CAT = ["sex","cp","fbs","restecg","exang","slope","thal"]
    fig_c, axes_c = plt.subplots(2,4,figsize=(15,6))
    for ax,col in zip(axes_c.ravel(), CAT):
        tmp = df.copy(); tmp[col]=tmp[col].astype(str).where(tmp[col].notna(), np.nan)
        sns.barplot(data=tmp, x=col, y='target', ax=ax, errorbar=None, color='#72B7B2')
        ax.set_title(col); ax.set_ylabel('Taux maladie'); ax.tick_params(axis='x', rotation=25)
    axes_c.ravel()[-1].axis('off')
    plt.tight_layout(); st.pyplot(fig_c); plt.close(fig_c)
    st.caption("**Nom** : barplots taux de maladie par catégorie. **Utilité** : identifier modalités à risque. **Lecture** : hauteur = proportion de malades dans la modalité. **Interprétation** : `cp asymptomatic`, `exang True`, `thal reversable/fixed defect`, `slope flat/downsloping` associés à taux de maladie plus élevé.")

    # --- Step 5 correlation ---
    st.subheader("Étape 5 — Étude de corrélation")
    st.markdown("""
**Utilité** : mesurer lien linéaire entre variables numériques et avec `target` ; détecter redondance.
**Comment lire** : valeur -1 à +1 ; proche +1 même sens, -1 sens inverse, 0 pas de lien linéaire ; diagonale = 1.
**Limite** : corrélation ≠ causalité, linéaire seulement.
""")
    fig_corr, corr = correlation_figure(df)
    st.pyplot(fig_corr); plt.close(fig_corr)
    tc = corr["target"].drop("target").sort_values(key=lambda s: s.abs(), ascending=False)
    st.dataframe(tc.rename("corr_avec_target").to_frame(), use_container_width=True)
    st.caption("**Interprétation** : `ca` (+0.46), `oldpeak` (+0.39), `thalch` (-0.39) sont les plus liés à target ; `chol/trestbps` faibles. `ca` a beaucoup de NA, corrélation à interpréter avec prudence.")

    # --- Step 6 split/models ---
    st.subheader("Étape 6 — Split, pipeline, modèles")
    st.markdown("""
- **Split** : 80% train / 20% test, `stratify=y`, `random_state=42`. **Utilité** : tester généralisation sur données jamais vues.
- **Pipeline anti-fuite** : imputation + encoding + scaling dans `ColumnTransformer/Pipeline`, fit sur train.
- **Modèles** : Baseline (référence), Régression logistique (simple/interprétable), Arbre (règles), KNN (voisins), Random Forest (ensemble d'arbres).
- **Pourquoi Random Forest ?** Robuste, capture non-linéarités/interactions, moins sensible au bruit ; `n_estimators=300`, `min_samples_leaf=3`, `random_state=42`.
""")

    # --- Step 7 evaluation ---
    st.subheader("Étape 7 — Évaluation (VIC + classification)")
    st.markdown("""
| Métrique | Définition simple | Comment lire | Dans notre cas |
|---|---|---|---|
| **MAE** | Erreur absolue moyenne | plus bas = mieux | exigence VIC, rapportée sur probabilité/classe |
| **MSE** | Erreur quadratique moyenne | pénalise grosses erreurs | exigence VIC |
| **R²** | Part de variance expliquée | 1 bon, 0 naïf, <0 mauvais | exigence VIC ; prudence en classification |
| **Recall** | Vrais malades détectés / vrais malades | proche 1 = peu de malades manqués | **prioritaire en pré-dépistage** |
| **Precision** | Vrais malades parmi prédits malades | proche 1 = peu de fausses alertes | compromis avec recall |
| **AUC** | Séparation sain/malade tous seuils | 0.5 hasard, 1 parfait | mesure globale |

**Pourquoi recall prioritaire ?** Faux négatif = malade non détecté, risque grave. On accepte plus de fausses alertes en pré-dépistage.
""")

    st.subheader("Étape 8 — Seuil, export, déploiement, mise à jour")
    st.markdown("""
- **Seuil** : transforme probabilité en décision. Calculé sur validation croisée train (objectif rappel ≥ 90%), puis appliqué au test. Pas un chiffre fixe arbitraire.
- **Export** : `model.joblib` + `model_meta.json` sauvegardent pipeline et métadonnées.
- **App Streamlit** : charge le `.joblib` avec `joblib.load()` et prédit sans réentraîner.
- **Retrain / update** : quand nouvelles données validées arrivent, réentraîner, réévaluer, puis réexporter joblib/json.
""")
    st.code("""# Export (notebook / entraînement)
joblib.dump(pipeline, "model.joblib")
json.dump(meta, open("model_meta.json", "w"))

# Consommation (Streamlit app)
model = joblib.load("model.joblib")
meta = json.load(open("model_meta.json"))
proba = model.predict_proba(X_new)[:, 1]""", language="python")

    st.subheader("Étape 9 — Main steps of ML (vue globale)")
    ml_steps = [
        ("1. Collect data", "Collecter données", "CSV/DB/API, définir source et cible"),
        ("2. Clean & prepare", "Nettoyer/préparer", "NA, doublons, types, encodage, scaling"),
        ("3. Choose model", "Choisir modèle", "Baseline, LogReg, KNN, RF selon problème"),
        ("4. Train model", "Entraîner", "fit sur train uniquement"),
        ("5. Evaluate model", "Évaluer", "MAE/MSE/R² + Recall/F1/AUC, matrice confusion"),
        ("6. Make prediction", "Prédire", "nouveau patient/formulaire/scénario"),
        ("7. Deploy model", "Déployer", "export joblib/json + app Streamlit"),
        ("8. Retrain & update", "Réentraîner/MAJ", "nouvelles données, monitoring, nouvelle version"),
    ]
    st.table(pd.DataFrame(ml_steps, columns=["Step (EN)", "Étape (FR)", "Détail"]))
    st.info("**Joblib vs JSON dans notre cas** : `joblib` est meilleur pour sauvegarder le pipeline/modèle sklearn entraîné (objet Python complexe). `JSON` est mieux pour métadonnées lisibles: seuil, colonnes, min/max/médiane, catégories, citation. Donc on utilise **les deux**: `model.joblib` + `model_meta.json`.")

# -------------------- PAGE 3 --------------------
elif page.startswith("3"):
    st.header("3) Prédiction / Verdict final (formulaire)")
    st.markdown("Saisir les données patient. Les 3 modèles comparent la même entrée. **Aide à la décision, pas décision automatique.**")

    with st.expander("Voir la fonction simple predire_scenario (style demandé, adapté CardioRisk binaire)"):
        st.code("""ORDER = ["Sain (0)", "Malade (1)"]

def predire_scenario(valeurs: dict, seuil_alerte: float = 0.38):
    \"\"\"Prédit la classe, les probabilités et le niveau d'alerte. Aide à la décision, pas décision automatique.\"\"\"
    ligne = pd.DataFrame([valeurs])[X.columns]
    proba = model.predict_proba(ligne)[0]                 # [P(sain), P(malade)]
    classe = ORDER[int(np.argmax(proba))]
    p_grave = float(proba[-1])                            # probabilité classe grave = Malade
    if p_grave >= seuil_alerte:        niveau = "ALERTE – vérification humaine prioritaire"
    elif p_grave >= seuil_alerte / 2:  niveau = "VIGILANCE – à surveiller"
    else:                              niveau = "NORMAL"
    return {
        "classe_predite": classe,
        "probabilites": {c: round(float(p), 3) for c, p in zip(ORDER, proba)},
        "P(Malade)": round(p_grave, 3),
        "seuil": seuil_alerte,
        "niveau": niveau
    }

for nom, vals in SCENARIOS.items():
    print(f\"--- Scénario : {nom}\")
    print(predire_scenario(vals))
    print()""", language="python")

    patient = {}
    with st.form("formulaire_prediction"):
        colA, colB = st.columns(2)
        num_items = list(meta["num"].items())
        half = (len(num_items) + 1) // 2
        with colA:
            st.markdown("#### Mesures numériques")
            for col, s in num_items[:half]:
                label, desc = FIELD_INFO.get(col, (col, ""))
                patient[col] = st.number_input(label, min_value=float(s["min"]), max_value=float(s["max"]), value=float(s["median"]), key=f"n_{col}", help=desc)
        with colB:
            st.markdown("#### Mesures numériques (suite)")
            for col, s in num_items[half:]:
                label, desc = FIELD_INFO.get(col, (col, ""))
                patient[col] = st.number_input(label, min_value=float(s["min"]), max_value=float(s["max"]), value=float(s["median"]), key=f"n_{col}", help=desc)
        st.markdown("#### Variables catégorielles")
        cat_cols = st.columns(2)
        for i, (col, options) in enumerate(meta["cat"].items()):
            opts = [o for o in options if o is not None]
            label, desc = FIELD_INFO.get(col, (col, ""))
            with cat_cols[i % 2]:
                patient[col] = st.selectbox(label, opts, index=0, key=f"c_{col}", help=desc)
        envoyer = st.form_submit_button("Prédire / Comparer les 3 modèles", use_container_width=True)

    if envoyer:
        X = pd.DataFrame([patient])
        seuil = float(meta["seuil"])
        errors, warns = validate_patient_inputs(patient)
        st.markdown("### Données saisies")
        st.dataframe(X, use_container_width=True)
        for w in warns: st.warning(w)
        if errors:
            for e in errors: st.error(e)
            st.stop()

        models3, cmp_metrics, cms = train_compare_models()
        rows = []
        for name, m in models3.items():
            prob = float(m.predict_proba(X)[0, 1])
            pred_i = int(prob >= seuil)
            rows.append({"Modèle": name, "Probabilité malade %": round(prob*100,2), "Seuil %": round(seuil*100,2), "Prédiction": "Malade (1)" if pred_i else "Sain (0)", "Verdict": "⚠️ Risque élevé" if pred_i else "✅ Risque faible"})
        st.markdown("### Comparaison des 3 modèles (mêmes données formulaire)")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("### Matrice de confusion + métriques par modèle (même jeu de test)")
        st.dataframe(cmp_metrics.style.format({"Accuracy":"{:.2%}","Précision":"{:.2%}","Rappel":"{:.2%}","F1":"{:.4f}","AUC":"{:.2%}","FN":"{:.0f}","FP":"{:.0f}"}), use_container_width=True)
        best_global = cmp_metrics.sort_values(["Accuracy","Précision","F1"], ascending=False).index[0]
        best_recall = cmp_metrics["Rappel"].idxmax()
        st.success(f"**Meilleur performance globale (Accuracy/Précision/F1)** : **{best_global}** — Accuracy={cmp_metrics.loc[best_global,'Accuracy']:.2%}, Précision={cmp_metrics.loc[best_global,'Précision']:.2%}, F1={cmp_metrics.loc[best_global,'F1']:.4f}, FP={int(cmp_metrics.loc[best_global,'FP'])}. Moins de fausses alertes/erreurs globales, mais pas forcément le plus de malades détectés.")
        st.info(f"**Meilleur en pré-dépistage (priorité Rappel/FN)** : **{best_recall}** — Rappel={cmp_metrics.loc[best_recall,'Rappel']:.2%}, FN={int(cmp_metrics.loc[best_recall,'FN'])}, FP={int(cmp_metrics.loc[best_recall,'FP'])}. Manquer un malade (FN) est critique.")
        with st.expander("Quel est le meilleur modèle ? (interprétation vérifiée)"):
            st.markdown("""
**KNN est le meilleur modèle en termes de performance globale** sur Accuracy, Précision et F1 (résultats vérifiés sur le même jeu de test).

**Pourquoi ?**
- **Accuracy = 84,78 %** → il réalise le plus grand nombre de prédictions correctes.
- **Précision = 84,91 %** → parmi les personnes prédites malades, environ 85 % sont réellement malades.
- **F1 = 0,8654** → il offre un bon équilibre entre précision et rappel.
- **16 FP** → il génère moins de fausses alertes que la régression logistique et Random Forest.

**Cependant, KNN n'est pas celui qui détecte le plus de malades.**

### Interprétation des métriques
- **Accuracy** = (TP+TN)/(TP+TN+FP+FN)
- **Précision** = TP/(TP+FP)
- **Rappel** = TP/(TP+FN)
- **F1** = 2*(Précision*Rappel)/(Précision+Rappel)
- **AUC** : capacité à distinguer malades/non-malades, indépendamment d'un seuil fixé.

### Contexte médical (pré-dépistage)
| Modèle | FN (malades manqués) | Lecture |
|---|---:|---|
| Régression logistique | 11 | Détecte le plus de malades parmi les modèles comparés. |
| KNN | 12 | Un malade manqué de plus que la régression logistique. |
| Random Forest | 12 | Un malade manqué de plus que la régression logistique. |

- **FN (False Negative)** : malade prédit non malade → risque critique.
- **FP (False Positive)** : non malade prédit malade → fausse alerte.

> **Conclusion** : KNN = meilleur équilibre global. Régression logistique = meilleur pour limiter les FN en pré-dépistage. Le choix final dépend du compromis métier validé par un humain.
""")
        cm_cols = st.columns(3)
        for i,(name,cm) in enumerate(cms.items()):
            tn,fp,fn,tp = cm.ravel()
            fig_cm,ax_cm = plt.subplots(figsize=(4.2,3.4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm, xticklabels=['Prédit Sain','Prédit Malade'], yticklabels=['Réel Sain','Réel Malade'])
            ax_cm.set_title(name, fontsize=9)
            with cm_cols[i%3]:
                st.pyplot(fig_cm); plt.close(fig_cm)
                st.caption(f"TN={tn} FP={fp} FN={fn} TP={tp}")

        with st.expander("Interprétation : réel vs prédit (4 zones)"):
            st.markdown("""
**Ligne rouge (seuil)** : au-dessus du seuil → prédit *Malade (1)* ; en dessous → prédit *Sain (0)*.

| Zone | Position | Signification | Lecture |
|---|---|---|---|
| **Haut gauche** | Réel Sain (0), au-dessus ligne rouge | **Faux positif (FP)**, orange | Sain prédit malade = fausse alerte. |
| **Bas gauche** | Réel Sain (0), sous ligne rouge | **Vrai négatif (TN)**, vert | Sain bien prédit sain. |
| **Haut droite** | Réel Malade (1), au-dessus ligne rouge | **Vrai positif (TP)**, vert | Malade bien détecté. |
| **Bas droite** | Réel Malade (1), sous ligne rouge | **Faux négatif (FN)**, rouge | Malade prédit sain = malade manqué (critique). |

**Couleurs** : vert = correct, orange = FP, rouge = FN. Les points rouges en bas à droite sont les plus critiques en pré-dépistage.
""")

        with st.expander("Comparaison 3 stratégies valeurs manquantes (ca, thal) — même RandomForest final"):
            @st.cache_data
            def compare_missing_strategies():
                from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
                from sklearn.compose import ColumnTransformer
                from sklearn.pipeline import Pipeline
                from sklearn.impute import SimpleImputer
                from sklearn.preprocessing import OneHotEncoder, StandardScaler
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                RS=42
                dfm=pd.read_csv(DATA_PATH).copy()
                dfm.loc[dfm['chol']==0,'chol']=np.nan; dfm.loc[dfm['trestbps']==0,'trestbps']=np.nan
                dfm['target']=(dfm['num']>0).astype(int)
                NUM_ALL=['age','trestbps','chol','thalch','oldpeak','ca']; CAT_ALL=['sex','cp','fbs','restecg','exang','slope','thal']
                for c in CAT_ALL: dfm[c]=dfm[c].astype(str).where(dfm[c].notna(), np.nan)
                X=dfm[NUM_ALL+CAT_ALL].copy(); y=dfm['target']
                Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,stratify=y,random_state=RS)
                cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=RS)
                def preproc(num_cols,cat_cols):
                    return ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),num_cols),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cat_cols)])
                def evalp(pipe,Xtr,ytr,Xte,yte):
                    pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); proba=pipe.predict_proba(Xte)[:,1]
                    out={'Accuracy':accuracy_score(yte,pred),'Precision':precision_score(yte,pred,zero_division=0),'Recall':recall_score(yte,pred,zero_division=0),'F1-Score':f1_score(yte,pred,zero_division=0),'ROC-AUC':roc_auc_score(yte,proba)}
                    sc=cross_validate(Pipeline(pipe.steps), pd.concat([Xtr,Xte]), pd.concat([ytr,yte]), cv=cv, scoring={'ROC-AUC':'roc_auc'}, n_jobs=-1)
                    out['CV ROC-AUC']=float(sc['test_ROC-AUC'].mean()); return out
                # Exp1
                Xtr1,Xte1=Xtr.drop(columns=['ca','thal']),Xte.drop(columns=['ca','thal'])
                m1=evalp(Pipeline([('pre',preproc([c for c in NUM_ALL if c!='ca'],[c for c in CAT_ALL if c!='thal'])),('rf',RandomForestClassifier(n_estimators=300,random_state=RS,n_jobs=-1))]),Xtr1,ytr,Xte1,yte)
                # Exp2
                m2=evalp(Pipeline([('pre',preproc(NUM_ALL,CAT_ALL)),('rf',RandomForestClassifier(n_estimators=300,random_state=RS,n_jobs=-1))]),Xtr.copy(),ytr,Xte.copy(),yte)
                # Exp3
                Xtr3,Xte3=Xtr.copy(),Xte.copy()
                for d in [Xtr3,Xte3]: d['ca_missing']=d['ca'].isna().astype(int); d['thal_missing']=d['thal'].isna().astype(int)
                imp_num=['age','trestbps','chol','thalch','oldpeak']; imp_cat=['sex','cp','fbs','restecg','exang','slope']
                imp_pre=ColumnTransformer([('num',SimpleImputer(strategy='median'),imp_num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),imp_cat)])
                imp_pre.fit(Xtr3[imp_num+imp_cat])
                Xtr_imp=imp_pre.transform(Xtr3[imp_num+imp_cat]); Xte_imp=imp_pre.transform(Xte3[imp_num+imp_cat])
                ca_known=Xtr3['ca'].notna(); rf_ca=RandomForestClassifier(n_estimators=300,random_state=RS,n_jobs=-1)
                rf_ca.fit(Xtr_imp[ca_known.values], Xtr3.loc[ca_known,'ca'].astype(float).astype(int).astype(str))
                for d,Xi in [(Xtr3,Xtr_imp),(Xte3,Xte_imp)]:
                    mask=d['ca'].isna();
                    if mask.any(): d.loc[mask,'ca']=rf_ca.predict(Xi[mask.values]).astype(float)
                th_known=Xtr3['thal'].notna(); rf_th=RandomForestClassifier(n_estimators=300,random_state=RS,n_jobs=-1)
                rf_th.fit(Xtr_imp[th_known.values], Xtr3.loc[th_known,'thal'].astype(str))
                for d,Xi in [(Xtr3,Xtr_imp),(Xte3,Xte_imp)]:
                    mask=d['thal'].isna();
                    if mask.any(): d.loc[mask,'thal']=rf_th.predict(Xi[mask.values])
                m3=evalp(Pipeline([('pre',preproc(NUM_ALL+['ca_missing','thal_missing'],CAT_ALL)),('rf',RandomForestClassifier(n_estimators=300,random_state=RS,n_jobs=-1))]),Xtr3,ytr,Xte3,yte)
                return pd.DataFrame({'Exp 1 - Suppression ca,thal':m1,'Exp 2 - Médiane/mode (baseline)':m2,'Exp 3 - RF Imputation + indicateurs':m3}).T.round(4)
            try:
                cmp_na = compare_missing_strategies()
                st.dataframe(cmp_na, use_container_width=True)
                best_na = cmp_na['ROC-AUC'].astype(float).idxmax()
                st.success(f"**Meilleure stratégie selon ROC-AUC test** : {best_na}. (Même split, même RandomForest final, random_state=42, CV 5-fold ROC-AUC affichée)")
            except Exception as e:
                st.caption(f"Comparaison indisponible: {e}")
            # scatter réel vs prédit for best_recall model
            try:
                from sklearn.model_selection import train_test_split
                df_all = pd.read_csv(DATA_PATH)
                df_all.loc[df_all["chol"]==0,"chol"]=np.nan
                df_all.loc[df_all["trestbps"]==0,"trestbps"]=np.nan
                df_all["target"]=(df_all["num"]>0).astype(int)
                NUM=["age","trestbps","chol","thalch","oldpeak","ca"]; CAT=["sex","cp","fbs","restecg","exang","slope","thal"]
                for c in CAT: df_all[c]=df_all[c].astype(str).where(df_all[c].notna(), np.nan)
                Xall=df_all[NUM+CAT]; yall=df_all["target"]
                Xtr,Xte,ytr,yte= train_test_split(Xall,yall,test_size=0.2,stratify=yall,random_state=42)
                m=models3[best_recall]
                proba_test=m.predict_proba(Xte)[:,1]
                pred_test=(proba_test>=seuil).astype(int)
                yte=yte.reset_index(drop=True)
                jitter=np.random.normal(0,0.04,len(yte))
                colors=np.where((yte.values==1)&(pred_test==0),'#d62728',np.where((yte.values==0)&(pred_test==1),'#ff7f0e','#2ca02c'))
                fig_s,ax_s=plt.subplots(figsize=(7,4.5))
                ax_s.scatter(yte.values+jitter, proba_test, c=colors, alpha=0.65, edgecolor='white', linewidth=0.3)
                ax_s.axhline(seuil, color='red', linestyle='--', label=f'Seuil = {seuil:.2%}')
                ax_s.set_xticks([0,1]); ax_s.set_xticklabels(['Réel: Sain (0)','Réel: Malade (1)'])
                ax_s.set_ylabel('Probabilité prédite de maladie'); ax_s.set_ylim(-0.02,1.02)
                ax_s.set_title(f'Réel vs prédit (test) — {best_recall}')
                ax_s.legend(); st.pyplot(fig_s); plt.close(fig_s)
            except Exception as e:
                st.caption(f"Graphique réel vs prédit indisponible: {e}")

        main_p = float(models3[best_recall].predict_proba(X)[0,1])
        verdict = "MALADE / RISQUE ÉLEVÉ" if main_p >= seuil else "SAIN / RISQUE FAIBLE"
        st.markdown("### Verdict final")
        if main_p >= seuil:
            st.error(f"**Verdict ({best_recall}, priorité rappel) : {verdict}** — Probabilité {main_p:.1%} ≥ seuil {seuil:.1%}. Avis médical recommandé.")
        else:
            st.success(f"**Verdict ({best_recall}, priorité rappel) : {verdict}** — Probabilité {main_p:.1%} < seuil {seuil:.1%}. Pas d'alerte au seuil choisi, rester vigilant si symptômes.")

        st.markdown("### Pourquoi ce verdict ? (logique)")
        reasons = []
        if patient.get("cp") == "asymptomatic":
            reasons.append("`cp=asymptomatic` : dans le dataset, associé à un taux élevé de maladie.")
        if str(patient.get("exang")) in ["True", "1", "True.0"]:
            reasons.append("`exang=True` : angine à l'effort, signal fort.")
        if float(patient.get("oldpeak", 0)) >= 1.0:
            reasons.append("`oldpeak` élevé : dépression ST à l'effort, associée au risque.")
        if float(patient.get("thalch", 200)) < 140:
            reasons.append("`thalch` faible : fréquence cardiaque max réduite, souvent observée chez malades.")
        if float(patient.get("ca", 0)) >= 1:
            reasons.append("`ca >= 1` : vaisseaux majeurs colorés, signal important.")
        if patient.get("thal") in ["reversable defect", "fixed defect"]:
            reasons.append("`thal` anormal : associé à plus de maladie que `normal`.")
        if not reasons:
            reasons.append("Aucun signal fort évident isolé ; le modèle combine toutes les variables (effets non linéaires possibles).")
        st.write("- " + "\n- ".join(reasons))
        st.warning("Verdict pédagogique issu d'un modèle statistique. Il ne remplace pas un diagnostic médical.")

# -------------------- PAGE 4 --------------------
else:
    st.header("4) Notebook prioritaire")
    st.success("Le notebook CardioRisk contient le détail complet : code, sorties, graphiques, interprétations figure par figure.")
    render_notebook(NOTEBOOK_PRIORITY)

st.sidebar.markdown("---")
st.sidebar.caption("CardioRisk Studio — ISCAE")
st.sidebar.caption("Équipe: " + ", ".join(TEAM))
st.sidebar.markdown(f"[GitHub]({GITHUB_URL})")
