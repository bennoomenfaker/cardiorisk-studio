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
    """Données brutes (phases 1-2) : aucune correction."""
    df = pd.read_csv(DATA_PATH)
    df["target"] = (df["num"] > 0).astype(int)
    return df


@st.cache_data
def clean_data():
    """Phase 3 : nettoyage de base, AVANT toute analyse visuelle ou corrélation.

    - `chol = 0` et `trestbps = 0` sont impossibles -> valeurs manquantes
    - `id` (numéro) et `dataset` (hôpital d'origine) ne sont pas des prédicteurs
    """
    d = load_data()
    for col in ("chol", "trestbps"):
        d.loc[d[col] == 0, col] = np.nan
    return d.drop(columns=[c for c in ("id", "dataset") if c in d.columns])


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


def _cramers_v(x, y):
    """V de Cramér (0 à 1) calculé sur la table de contingence, sans coder les modalités en nombres."""
    tab = pd.crosstab(x, y)
    if tab.empty or min(tab.shape) < 2:
        return np.nan
    obs = tab.to_numpy(); n = obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / n
    phi2 = ((obs - exp) ** 2 / exp).sum() / n
    r, k = tab.shape
    phi2_c = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_c = r - ((r - 1) ** 2) / (n - 1)
    k_c = k - ((k - 1) ** 2) / (n - 1)
    denom = min(k_c - 1, r_c - 1)
    return float(np.sqrt(phi2_c / denom)) if denom > 0 else np.nan


def _eta(cat, num):
    """Rapport de corrélation eta (0 à 1) : part de variance du numérique portée par les groupes de la catégorie."""
    d = pd.DataFrame({"groupe": cat, "valeur": num}).dropna()
    if d["groupe"].nunique() < 2 or d["valeur"].nunique() < 2:
        return np.nan
    g = d.groupby("groupe")["valeur"]
    ss_inter = (g.count() * (g.mean() - d["valeur"].mean()) ** 2).sum()
    ss_total = ((d["valeur"] - d["valeur"].mean()) ** 2).sum()
    return float(np.sqrt(ss_inter / ss_total)) if ss_total > 0 else np.nan


@st.cache_data
def corr_cat_figure(df: pd.DataFrame):
    cat = [c for c in meta["cat"] if c in df.columns]
    mat = pd.DataFrame([[1.0 if a == b else _cramers_v(df[a], df[b]) for b in cat] for a in cat], index=cat, columns=cat)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax)
    ax.set_title("V de Cramér — catégorielle × catégorielle")
    fig.tight_layout()
    return fig, mat


@st.cache_data
def assoc_mixte_figure(df: pd.DataFrame):
    num = [c for c in meta["num"] if c in df.columns]
    cat = [c for c in meta["cat"] if c in df.columns]
    mat = pd.DataFrame([[_eta(df[c], df[n]) for c in cat] for n in num], index=num, columns=cat)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax)
    ax.set_title("eta (η) — numérique × catégorielle")
    fig.tight_layout()
    return fig, mat


@st.cache_data
def lien_cible_table(df: pd.DataFrame):
    num = [c for c in meta["num"] if c in df.columns]
    cat = [c for c in meta["cat"] if c in df.columns]
    rows = []
    for c in num + cat:
        a = abs(df[[c, "target"]].corr().iloc[0, 1]) if c in num else _cramers_v(df[c], df["target"])
        rows.append({"Variable": c, "Mesure": "|Pearson|" if c in num else "V de Cramér",
                     "Association": round(float(a), 3), "Patients utilisés": int(df[c].notna().sum())})
    return pd.DataFrame(rows).sort_values("Association", ascending=False).reset_index(drop=True)


@st.cache_data
def taux_modalites_table(df: pd.DataFrame):
    rows = []
    for c in [c for c in meta["cat"] if c in df.columns]:
        t = df.groupby(c)["target"].agg(["mean", "size"])
        for m, l in t.iterrows():
            rows.append({"Variable": c, "Modalité": str(m), "Patients": int(l["size"]),
                         "Taux de malades %": round(100 * l["mean"], 1)})
    return pd.DataFrame(rows).sort_values(["Variable", "Taux de malades %"], ascending=[True, False]).reset_index(drop=True)


def show_fig(fig, caption: str | None = None):
    """Affiche une figure matplotlib puis la libère (évite `st.pyplot(fig); plt.close(fig)` partout)."""
    st.pyplot(fig)
    plt.close(fig)
    if caption:
        st.caption(caption)


def show_table(rows, columns):
    """Affiche une liste de tuples sous forme de tableau sans index."""
    st.dataframe(pd.DataFrame(rows, columns=columns), use_container_width=True, hide_index=True)


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

    df_all = clean_data()          # même nettoyage de base que la page 2 (Phase 3)
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
        "2. Processus ML (18 phases)",
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
    st.header("2) Processus ML — les 18 phases (même découpage que le notebook)")
    st.markdown("""
Cette page suit **les 18 phases du notebook `heart_disease_cardiorisk.ipynb`**, dans le **même ordre**.

> **Règle d'ordre appliquée :** nettoyage **avant** l'analyse visuelle et la corrélation ; encodage / imputation / mise à l'échelle **après** le `train_test_split`, appris sur `X_train` seulement.

Pour chaque graphique : **nom / utilité / comment lire / interprétation**.
""")

    PHASES = [
        ("1", "Chargement des données", "lire le CSV brut dans `df`", "920 lignes brutes × 16"),
        ("2", "Compréhension structurelle", "`info`, `describe`, `value_counts` — sans rien modifier", "types, stats, modalités"),
        ("3", "Nettoyage de base **avant** visuel / corrélation", "NaN diagnostiqués, catégories harmonisées (3.1), zéros invalides (3.2), colonnes inutiles + doublons (3.3)", "918 × 14"),
        ("4", "Cible et familles de variables", "`num` (0–4) → `target` binaire ; listes `NUM` / `CAT`", "target 0/1"),
        ("5", "Valeurs aberrantes (IQR)", "détecter, interpréter, décider — aucune suppression à l'aveugle (nettoyage)", "tableau de décision"),
        ("6", "Analyse visuelle", "distributions, boxplots par classe, taux de maladie par modalité", "figures + lectures"),
        ("7", "Associations en 3 blocs", "7.1 Pearson (num×num) · 7.2 V de Cramér (cat×cat) · 7.3 eta (num×cat) · 7.4 lien avec target", "3 matrices + taux"),
        ("8", "Séparation Train / Test", "80/20 stratifié, `random_state=42`", "734 / 184"),
        ("9", "Préparation (Pipeline)", "médiane, mode, `StandardScaler`, `OneHotEncoder` — `fit` sur le train", "30 colonnes"),
        ("10", "3 stratégies de valeurs manquantes", "suppression / médiane-mode / patient similaire (KNN) sur `ca` et `thal`", "comparatif CV"),
        ("11", "Comparaison des modèles", "baseline + 5 modèles, puis vérification sur 8 modèles", "AUC + rappel"),
        ("12", "Évaluation sur le test", "matrice de confusion, ROC, R²/MAE/MSE/RMSE", "scores test"),
        ("13", "Seuil de décision", "seuil calculé pour un rappel ≥ 90 % (pas 0,50 arbitraire)", "≈ 0,41"),
        ("14", "Réel vs prédit", "contrôle visuel des probabilités autour du seuil", "figure test"),
        ("15", "Importance des variables", "permutation importance", "classement"),
        ("16", "Nouvelles données", "formulaire / scénarios, comparés sur 3 modèles", "prédictions"),
        ("17", "Export & déploiement", "`model.joblib` + `model_meta.json` + application Streamlit", "fichiers"),
        ("18", "Conclusion", "réponses Q1–Q5, limites, perspectives", "synthèse"),
    ]
    show_table(PHASES, ["Phase", "Titre", "Ce qu'on y fait", "Sortie / décision"])
    st.caption("Chaque phase est détaillée ci-dessous, dans l'ordre, puis reprise dans le notebook avec les sorties exécutées.")

    st.markdown("""
**Pourquoi cet ordre ? (la question qui revient le plus souvent)**

| Question | Réponse |
|---|---|
| Pourquoi le nettoyage **avant** la corrélation ? | Un `chol = 0` impossible ou une catégorie mal écrite fausse à la fois les graphiques **et** les coefficients de corrélation. On corrige d'abord, on analyse ensuite. |
| Alors pourquoi les **valeurs aberrantes** (Phase 7) viennent-elles après la corrélation ? | Parce que ce n'est **pas du nettoyage** : aucune ligne n'est supprimée. C'est un **diagnostic** : on ne juge pas un extrême (cholestérol à 603) sans avoir vu sa distribution. |
| Pourquoi l'**imputation** (Phases 9–10) après le split ? | Imputer avant reviendrait à calculer la médiane/le mode sur **tout** le jeu, test inclus → **fuite de données** et scores trop optimistes. Le `Pipeline` apprend ces valeurs sur `X_train` uniquement. |
| Et la corrélation, elle utilise quelles données ? | Les données **nettoyées** (Phase 3). Pour les catégories, une copie d'analyse les remplit par le mode : c'est autorisé pour un graphique, jamais pour entraîner un modèle. |
""")

    # Données brutes pour les phases 1-2, données nettoyées pour les phases 3+ :
    # le code suit ainsi le même ordre que le texte (nettoyage AVANT visuel/corrélation).
    raw = df
    d = clean_data()
    NUM_APP = list(meta["num"].keys())
    CAT_APP = list(meta["cat"].keys())

    # --- Phase 1 ---
    st.subheader("Phase 1 — Chargement des données")
    st.markdown("""
**Action** : lire `heart_disease_uci.csv` et inspecter les premières lignes.
**Utilité** : point de départ — vérifier que le fichier est bien lu (colonnes, formats, valeurs manquantes visibles en `NaN`).
**Comment lire** : chaque ligne = un patient ; mélange de nombres (`age`, `chol`) et de texte (`Male`, `typical angina`).
""")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations (patients)", raw.shape[0])
    c2.metric("Colonnes brutes", raw.shape[1] - 1)  # hors cible ajoutée
    c3.metric("Sains (0)", int((raw["target"] == 0).sum()))
    c4.metric("Malades (1)", int((raw["target"] == 1).sum()))
    st.dataframe(raw.head(8), use_container_width=True)

    # --- Phase 2 ---
    st.subheader("Phase 2 — Compréhension structurelle (sans modification)")
    st.markdown("""
**Action** : `info()`, `describe()`, `value_counts()`.
**Utilité** : connaître les types et le nombre de valeurs non vides avant de décider quoi corriger.
**Comment lire** : `count` faible = valeurs manquantes ; écart entre `mean` et `50%` = asymétrie ; `min`/`max` extrêmes = outliers ou erreurs.
""")
    with st.expander("Statistiques descriptives (describe) — données brutes"):
        desc = raw.describe(include="all").T
        desc["top"] = desc["top"].map(lambda v: "" if pd.isna(v) else str(v))  # bool + texte -> texte
        st.dataframe(desc, use_container_width=True)
        st.caption("`chol` : minimum 0 → impossible. Ce sont des manquants déguisés, corrigés en Phase 3.2.")
    with st.expander("Types, valeurs non nulles et modalités (info / value_counts)"):
        types_df = pd.DataFrame({
            "type": raw.dtypes.astype(str),
            "non nulles": raw.notna().sum(),
            "manquantes": raw.isna().sum(),
            "modalités": raw.nunique(),
        })
        st.dataframe(types_df, use_container_width=True)
        st.caption("Colonnes complètes : `id`, `age`, `sex`, `dataset`, `cp`, `num`. Les plus incomplètes : `ca` (66 %), `thal` (53 %), `slope` (34 %).")

    # --- Phase 3 ---
    st.subheader("Phase 3 — Nettoyage de base (AVANT l'analyse visuelle et la corrélation)")
    st.markdown("""
| Étape | Action | Pourquoi |
|---|---|---|
| **3.0** | Diagnostiquer les valeurs manquantes (nombre + %) | savoir combien manque et où avant de choisir une méthode |
| **3.1** | **Harmoniser les catégories** : espaces parasites, doublons de casse, modalités rares | deux écritures d'une même catégorie fausseraient graphiques, moyennes et encodage |
| **3.2** | Remplacer les valeurs **impossibles** (`chol = 0`, `trestbps = 0`) par `NaN` | ce sont des manquants codés « 0 » |
| **3.3** | Supprimer `id` et `dataset`, vérifier les doublons | pas de prédicteur sans information médicale ni de fuite de contexte |

> L'**imputation** et l'**encodage** ne se font pas ici : ils sont appris sur `X_train` seulement (Phase 9).
""")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Doublons", int(d.drop(columns="target").duplicated().sum()))
    k2.metric("chol == 0 corrigés", int((raw["chol"] == 0).sum()))
    k3.metric("trestbps == 0 corrigés", int((raw["trestbps"] == 0).sum()))
    k4.metric("Colonnes conservées", d.shape[1] - 1)  # hors cible

    miss = d.isna().sum().sort_values()
    miss = miss[miss > 0]
    fig_m, ax_m = plt.subplots(figsize=(8, 4))
    miss.plot(kind="barh", ax=ax_m, color="#4C78A8")
    ax_m.set_title("Valeurs manquantes par colonne (après nettoyage de base)")
    ax_m.set_xlabel("Nombre de NaN")
    show_fig(fig_m, "**Nom** : barplot horizontal des valeurs manquantes. **Lecture** : longueur = nombre de NaN, **après** correction des zéros impossibles. **Interprétation** : `ca`/`thal`/`slope` très incomplètes → imputation + indicateur de manquant, jamais de suppression massive de lignes.")

    # --- Phase 4 ---
    st.subheader("Phase 4 — Variable cible et familles de variables")
    st.markdown("""
**Action** : `target = 1` si `num > 0`, sinon `0` ; puis classer les variables en `NUM` (6) et `CAT` (7).
**Utilité** : `num` va de 0 à 4 (stades) ; le pré-dépistage est une question **binaire**. Les deux familles seront traitées différemment dans le pipeline.
""")
    counts = d["target"].value_counts().sort_index()
    fig_t, ax_t = plt.subplots(figsize=(5, 3.5))
    ax_t.bar(["Sain (0)", "Malade (1)"], [counts.get(0, 0), counts.get(1, 0)], color=["#4C78A8", "#E45756"])
    ax_t.set_title("Distribution de la cible")
    ax_t.set_ylabel("Nombre de patients")
    for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        ax_t.text(i, v + 5, str(int(v)), ha="center")
    show_fig(fig_t, "**Nom** : barplot de la cible. **Utilité** : vérifier l'équilibre des classes. **Interprétation** : ~45 % sains / 55 % malades → classes équilibrées, accuracy utilisable, mais on surveille aussi rappel/F1.")
    st.code(f"""NUM = {NUM_APP}
CAT = {CAT_APP}
df["target"] = (df["num"] > 0).astype(int)   # 0 = sain, 1 = maladie""", language="python")

    # --- Phase 5 ---
    st.subheader("Phase 5 — Valeurs aberrantes (règle IQR) : détecter, puis décider")
    st.markdown("""
**Action** : signaler tout patient hors de `[Q1 − 1,5 × IQR ; Q3 + 1,5 × IQR]` pour chaque variable numérique.
**Utilité** : repérer les valeurs extrêmes, sans les supprimer automatiquement : un cholestérol à 603 mg/dL ou une tension à 200 mmHg sont extrêmes mais **cliniquement possibles**.
**C'est du nettoyage, et il est placé AVANT l'analyse visuelle et la corrélation** : on décide du sort des valeurs extrêmes, puis on interprète les graphiques. Aucune ligne n'est finalement retirée.
""")
    iqr_rows = []
    for col in NUM_APP:
        q1, q3 = d[col].quantile([0.25, 0.75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        iqr_rows.append({
            "Variable": col,
            "Borne basse": round(lo, 1),
            "Borne haute": round(hi, 1),
            "Hors bornes": int(((d[col] < lo) | (d[col] > hi)).sum()),
            "Décision": "règle non applicable (discrète)" if col == "ca" else "conserver (plausible)",
        })
    show_table(iqr_rows, ["Variable", "Borne basse", "Borne haute", "Hors bornes", "Décision"])
    st.caption("**Interprétation** : aucune ligne n'est supprimée. Les valeurs extrêmes sont porteuses d'information (patients les plus malades) ; le `StandardScaler` du pipeline limite leur influence et les arbres y sont insensibles.")

    # --- Phase 6 ---
    st.subheader("Phase 6 — Analyse visuelle (après nettoyage, avant corrélation)")
    st.markdown("**Utilité** : comprendre les relations variable-cible avant modélisation. Chaque graphique répond à une question.")

    fig_age, axes_age = plt.subplots(1, 2, figsize=(11, 4))
    for t, label, color in [(0, "Sain (0)", "#4C78A8"), (1, "Malade (1)", "#E45756")]:
        axes_age[0].hist(d.loc[d["target"] == t, "age"].dropna(), bins=15, alpha=0.5, label=label, color=color)
    axes_age[0].set_title("Âge par cible (histogramme)")
    axes_age[0].set_xlabel("Âge"); axes_age[0].set_ylabel("Nombre"); axes_age[0].legend()
    sns.boxplot(data=d, x="target", y="age", ax=axes_age[1], palette=["#4C78A8", "#E45756"])
    axes_age[1].set_title("Âge par cible (boxplot)")
    axes_age[1].set_xticks([0, 1]); axes_age[1].set_xticklabels(["Sain (0)", "Malade (1)"])
    plt.tight_layout()
    show_fig(fig_age, "**Nom** : histogramme superposé + boxplot âge/target. **Utilité** : voir si l'âge sépare les classes. **Lecture** : histogramme = distribution ; boxplot = médiane/quartiles/outliers. **Interprétation** : malades légèrement plus âgés, fort chevauchement → âge informatif mais pas suffisant seul.")

    fig_b, axes_b = plt.subplots(2, 3, figsize=(13, 6))
    for ax, col in zip(axes_b.ravel(), NUM_APP):
        sns.boxplot(data=d, x="target", y=col, ax=ax, palette=["#4C78A8", "#E45756"])
        ax.set_title(col); ax.set_xticks([0, 1]); ax.set_xticklabels(["Sain", "Malade"])
    plt.tight_layout()
    show_fig(fig_b, "**Nom** : boxplots numériques par cible. **Utilité** : comparer médiane, dispersion, outliers entre sains/malades. **Lecture** : boîte = Q1–Q3, trait = médiane, points = outliers. **Interprétation** : `thalch` plus bas chez les malades, `oldpeak`/`ca` plus élevés ; `chol`/`trestbps` se chevauchent beaucoup.")

    cat_plot = d.copy()
    for col in CAT_APP:
        cat_plot[col] = cat_plot[col].astype(str).where(d[col].notna(), np.nan)
    fig_c, axes_c = plt.subplots(2, 4, figsize=(15, 6))
    for ax, col in zip(axes_c.ravel(), CAT_APP):
        sns.barplot(data=cat_plot, x=col, y="target", ax=ax, errorbar=None, color="#72B7B2")
        ax.set_title(col); ax.set_ylabel("Taux maladie"); ax.tick_params(axis="x", rotation=25)
    axes_c.ravel()[-1].axis("off")
    plt.tight_layout()
    show_fig(fig_c, "**Nom** : barplots du taux de maladie par catégorie. **Utilité** : identifier les modalités à risque. **Lecture** : hauteur = proportion de malades dans la modalité. **Interprétation** : `cp = asymptomatic`, `exang = True`, `thal` anormal, `slope` flat/downsloping sont associés à un taux de maladie plus élevé.")

    # --- Phase 7 ---
    st.subheader("Phase 7 — Associations en 3 blocs séparés (pas de Pearson sur des catégories)")
    st.markdown("""
**Règle** : une seule mesure ne peut pas couvrir les trois situations. Un coefficient de Pearson sur `cp` codée 0-1-2-3 ne mesure que l'ordre de numérotation (démonstration faite dans le notebook). On sépare donc :

| Bloc | Variables | Mesure | Encodage ? |
|---|---|---|---|
| **7.1** | 6 numériques × 6 numériques | **r de Pearson** (−1 à +1) | non, ce sont des quantités |
| **7.2** | 7 catégorielles × 7 catégorielles | **V de Cramér** (0 à 1) | non, modalités brutes |
| **7.3** | 6 numériques × 7 catégorielles | **eta η** (0 à 1) | non, comparaison de moyennes |
| **7.4** | les 13 variables × `target` | mesure adaptée au type + taux de malades | non |
""")
    with st.expander("Ce que je fais, pourquoi, et pourquoi pas du One-Hot avant le split"):
        st.markdown("""
**Ce que je fais** : j'observe les variables **brutes**, sans les modifier, avec une mesure adaptée à chaque type de paire — Pearson (num × num), V de Cramér (cat × cat), eta (num × cat) — puis je classe les 13 variables par association avec `target` et je lis le sens dans les taux de malades par modalité.

**Pourquoi ce découpage** : une seule mesure ne peut pas couvrir les trois situations. Pearson suppose deux quantités avec une distance qui a un sens ; `cp` codée 0-1-2-3 n'en a pas (la démonstration du notebook donne −0,39, +0,47 ou −0,54 pour la même variable selon la numérotation). Le V de Cramér et eta travaillent sur les modalités brutes, sans leur inventer un ordre.

**Pourquoi pas de `OneHotEncoder` avant le split** :

| Raison | Explication |
|---|---|
| **Fuite de données** | l'encodeur apprend la liste des modalités sur les données qu'on lui donne ; appliqué avant `train_test_split`, il apprend des modalités présentes seulement dans le test. Il doit rester dans le `Pipeline` (Phase 9). |
| **Statistiques faussées** | Pearson sur des colonnes 0/1 écrase les coefficients (rarement > 0,40 même pour un lien fort) et fabrique de **fausses corrélations négatives** entre modalités d'une même variable (`cp_asymptomatic` est anti-corrélé à `cp_non-anginal`, leur somme vaut 1). |
| **Matrice illisible** | 13 variables → ~26 colonnes 0/1 : une matrice 26 × 26 inexploitable en soutenance. |

**La phrase à retenir** : *avant le split on observe les variables brutes avec l'outil adapté à chaque type ; après le split on transforme les catégorielles en One-Hot, appris sur `X_train` seulement.*
""")

    st.markdown("#### 7.1 — Numérique × numérique : Pearson")
    fig_corr, corr = correlation_figure(d)
    show_fig(fig_corr)
    corr_num = corr.drop(index="target", errors="ignore").drop(columns="target", errors="ignore")
    tc = corr["target"].drop("target").sort_values(key=lambda s: s.abs(), ascending=False)
    c71, c72 = st.columns(2)
    with c71:
        st.dataframe(tc.rename("r avec target").to_frame(), use_container_width=True)
    with c72:
        pairs = corr_num.where(np.triu(np.ones(corr_num.shape, dtype=bool), k=1)).stack()
        pairs = pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(6)
        st.dataframe(pairs.rename("r (entre features)").to_frame(), use_container_width=True)
    st.caption("**Interprétation 7.1** : paires les plus fortes `age`–`ca` 0,370, `age`–`thalch` −0,365, `oldpeak`–`ca` 0,282. **Aucune ne dépasse |r| = 0,40** : pas de redondance, on garde les 6 numériques. `age` est au centre de 4 paires : facteur de confusion classique. Pearson ne voit que le linéaire et ne doit jamais être calculé sur une catégorielle.")

    st.markdown("#### 7.2 — Catégorielle × catégorielle : V de Cramér")
    fig_cat, mat_cat = corr_cat_figure(d)
    show_fig(fig_cat)
    paires_cat = mat_cat.where(np.triu(np.ones(mat_cat.shape, dtype=bool), k=1)).stack().sort_values(ascending=False).head(6)
    st.dataframe(paires_cat.rename("V de Cramér").to_frame(), use_container_width=True)
    st.caption("**Interprétation 7.2** : `cp`–`exang` **0,447**, `sex`–`thal` 0,375, `exang`–`slope` 0,343, `exang`–`thal` 0,337. Toutes **sous 0,50** : associations modérées au plus, aucune redondance à supprimer. Ces quatre variables décrivent l'ischémie d'effort. Le V de Cramér dit « lié », pas « dans quel sens » : le sens se lit dans les taux par modalité (7.4).")

    st.markdown("#### 7.3 — Numérique × catégorielle : eta (η)")
    fig_eta, mat_eta = assoc_mixte_figure(d)
    show_fig(fig_eta)
    paires_eta = mat_eta.stack().sort_values(ascending=False).head(8)
    st.dataframe(paires_eta.rename("eta").to_frame(), use_container_width=True)
    st.caption("**Interprétation 7.3** : `oldpeak`–`slope` **0,421**, `thalch`–`slope` 0,410, `oldpeak`–`exang` 0,394, `thalch`–`cp` 0,372. η² ≤ 0,18 : les groupes expliquent au plus 18 % de la variance du numérique, donc signaux **modérés** et partiellement redondants (mesures de l'effort). Avec `chol` et `trestbps`, η ≤ 0,15.")

    st.markdown("#### 7.4 — Lien de chaque variable avec `target`")
    st.dataframe(lien_cible_table(d), use_container_width=True, hide_index=True)
    st.dataframe(taux_modalites_table(d), use_container_width=True, hide_index=True)
    st.caption("**Interprétation 7.4** : le podium est **catégoriel** — `cp` 0,538, `thal` 0,497, `exang` 0,461 — avant `ca` 0,456, `thalch` 0,393, `oldpeak` 0,386. Sens médical : `cp = asymptomatic` **79 %** de malades (contre 13,9 % en atypical angina), `exang = True` 83,6 %, `thal` anormal ~76–80 % (contre 29,6 % si normal), `slope` flat/downsloping ~77 % (contre 38,4 % upsloping), homme 63,2 % contre 25,9 %. ⚠️ `ca` n'est calculé que sur **309 patients** et `thal` sur 434 : associations fragiles. `chol` (0,119) reste 2ᵉ en importance dans le modèle (Phase 15) : association faible ≠ variable inutile.")

    # --- Phase 8 ---
    st.subheader("Phase 8 — Séparation Train / Test (avant tout apprentissage)")
    n_test = int(round(d.shape[0] * 0.2))
    s1, s2, s3 = st.columns(3)
    s1.metric("Train (80 %)", d.shape[0] - n_test)
    s2.metric("Test (20 %)", n_test)
    s3.metric("random_state", 42)
    st.markdown("""
- **Split** : 80 % train / 20 % test, `stratify=y` (même proportion de malades des deux côtés), `random_state=42`.
- **Utilité** : évaluer sur des patients **jamais vus** et éviter le sur-apprentissage.
- **Pourquoi ici** : tout ce qui *apprend* des données (médiane, mode, moyenne/écart-type, catégories) doit être ajusté sur `X_train` seulement — sinon l'information du test fuit dans l'entraînement.
""")

    # --- Phase 9 ---
    st.subheader("Phase 9 — Préparation des variables (apprise sur le train uniquement)")
    st.markdown("""
**Utilité** : rendre les données utilisables par les modèles et éviter la fuite de données.

**SimpleImputer — c'est quoi ?**  
`SimpleImputer` remplace les valeurs manquantes (`NaN`) par une valeur simple :
- `strategy="median"` (numérique) : prend la **médiane**. Plus robuste que la moyenne quand il y a des valeurs extrêmes/outliers.
- `strategy="most_frequent"` (catégoriel) : prend le **mode** (valeur la plus fréquente). On ne peut pas faire une moyenne sur du texte.
- `add_indicator=True` : ajoute une colonne 0/1 « était manquant ? » pour garder l'information de manque (utile pour `ca`).

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
    st.warning("La version simple est bonne pour comprendre. Dans le projet, on préfère la version **Pipeline/ColumnTransformer** car elle évite la fuite de données : le `fit` (médiane/mode/scale/encoder) se fait **sur le train uniquement**, puis on applique au test/app.")

    # --- Phase 10 ---
    st.subheader("Phase 10 — Valeurs manquantes : les 3 stratégies comparées")
    st.markdown("""
`ca` (66 % manquants) et `thal` (53 %) sont les deux colonnes critiques. Le notebook compare **3 solutions** sur le même split, avec le même `RandomForestClassifier` et une validation croisée 5-fold :

| Stratégie | Idée | Traitement de `ca` / `thal` |
|---|---|---|
| **1. Suppression** | retirer les colonnes trop incomplètes | colonnes supprimées |
| **2. Médiane / mode** (retenue) | remplacer par une valeur typique | `ca` = médiane, `thal` = mode |
| **3. Patient similaire (KNN)** | copier la valeur des patients les plus proches | `KNNImputer` sur les autres variables |
""")
    show_table([
        ("Exp 1 — Supprimer ca/thal", 0.8043, 0.8302, 0.8987, 0.8626),
        ("Exp 2 — Médiane / mode (retenue)", 0.8207, 0.8421, 0.9133, 0.8756),
        ("Exp 3 — Patient similaire (KNN)", 0.8478, 0.8667, 0.9116, 0.8747),
    ], ["Stratégie", "Accuracy test", "F1 test", "ROC-AUC test", "CV ROC-AUC"])
    st.caption("**Interprétation** : la **médiane/mode** a le meilleur ROC-AUC test (0,913) **et** le meilleur CV ROC-AUC (0,876). Le KNN garde le meilleur rappel au seuil 0,5 mais passe derrière sur la discrimination une fois les distances standardisées. Les IC se recoupent : classement indicatif. On garde **médiane/mode** — meilleure, plus simple à expliquer et c'est elle qui est **exportée** dans l'app.")
    st.info("Résultats mesurés dans le notebook, **Phase 10** (même split, `random_state=42`).")

    # --- Phase 11 ---
    st.subheader("Phase 11 — Comparaison des modèles (validation croisée sur le train)")
    st.markdown("""
- **Préparation identique** pour tous (même `Pipeline`), **même validation croisée 5-fold**.
- **Baseline** (`DummyClassifier`) : prédit toujours la classe majoritaire (55 %) — tout modèle doit faire nettement mieux.
- **Critère de choix** : l'**AUC** (classement indépendant du seuil) complétée par le **rappel** (part des malades détectés), prioritaire en dépistage.
- **Phase 11bis** : la même comparaison est rejouée avec 8 familles de modèles (SVM, Naive Bayes, AdaBoost, Gradient Boosting…) pour vérifier qu'aucune n'est meilleure.
""")
    show_table([
        ("Régression logistique (retenue)", 0.883, 0.852),
        ("SVM (noyau RBF)", 0.879, 0.869),
        ("Random Forest", 0.876, 0.857),
        ("AdaBoost", 0.874, 0.827),
        ("KNN", 0.864, 0.847),
        ("Naive Bayes", 0.857, 0.813),
        ("Gradient Boosting", 0.862, 0.825),
        ("Arbre de décision", 0.804, 0.786),
    ], ["Modèle", "AUC (CV 5-fold)", "Rappel (CV 5-fold)"])
    st.caption("**Interprétation** : aucune famille ajoutée ne dépasse la régression logistique. Le SVM a une AUC quasi identique (0,879) mais reste moins interprétable. L'arbre seul est le plus faible (0,804) : un arbre unique est instable, les ensembles le corrigent.")

    # --- Phase 12 ---
    st.subheader("Phase 12 — Évaluation sur le jeu test (une seule fois)")
    st.markdown("""
| Métrique | Définition simple | Comment lire | Dans notre cas |
|---|---|---|---|
| **MAE** | Erreur absolue moyenne | plus bas = mieux | **0,2317** sur la probabilité (0,1522 sur la classe) |
| **MSE** | Erreur quadratique moyenne | pénalise les grosses erreurs | **0,1067** (0,1522 sur la classe) |
| **RMSE** | Racine du MSE, même unité que la cible | plus bas = mieux | **0,3267** (0,3901 sur la classe) |
| **R²** | Part de variance expliquée | 1 bon, 0 naïf, <0 mauvais | **0,5680** sur la probabilité (0,3840 sur la classe) |
| **Recall** | Vrais malades détectés / vrais malades | proche 1 = peu de malades manqués | **prioritaire en pré-dépistage** |
| **Precision** | Vrais malades parmi les prédits malades | proche 1 = peu de fausses alertes | compromis avec le rappel |
| **AUC** | Séparation sain/malade tous seuils | 0.5 hasard, 1 parfait | mesure globale |

**Pourquoi le rappel est prioritaire ?** Un faux négatif = malade non détecté, risque grave. On accepte plus de fausses alertes en pré-dépistage.

**Rappel ou R² ?** C'est le **rappel** qui pilote la décision : un faux négatif coûte un malade non dépisté, alors que le R² traite symétriquement un malade manqué et une fausse alerte, et reste borné en classification binaire (0,568 ici). Le R², le MAE, le MSE et le RMSE servent de **contrôle méthodologique** (exigence VIC, comparaison des modèles, lecture de l'erreur), pas de critère de choix du seuil. Ordre retenu : **rappel → AUC → précision/F1 → R²/MAE/MSE/RMSE**.
""")
    with st.spinner("Recalcul des métriques sur le jeu test…"):
        _fitted, cmp_metrics, _cms = train_compare_models()
    st.dataframe(cmp_metrics.round(3), use_container_width=True)
    st.caption("Recalculé en direct par l'application, avec le même split et la même graine que le notebook (80/20 stratifié, `random_state=42`, seuil par défaut 0,50).")

    st.markdown("**Matrice de confusion des 3 modèles comparés (seuil 0,50)**")
    cm_rows = []
    for name, cm in _cms.items():
        tn, fp, fn, tp = cm.ravel()
        cm_rows.append({"Modèle": name, "TN (sain→sain)": int(tn), "FP (sain→malade)": int(fp),
                        "FN (malade→sain)": int(fn), "TP (malade→malade)": int(tp)})
    st.dataframe(pd.DataFrame(cm_rows), use_container_width=True, hide_index=True)
    st.caption("**Interprétation** : sur 184 patients (82 sains, 102 malades), la régression logistique détecte 92 malades sur 102 (**rappel 0,902**) en alertant à tort 18 sains (**précision 0,836**) ; Random Forest et KNN en détectent 89 (0,873) avec respectivement 18 et 14 fausses alertes. Les **FN** sont l'erreur critique : ce sont eux que le seuil abaissé à 0,41 fait passer de 10 à 8. Le R² (0,568 sur la probabilité) résume la qualité d'ensemble mais ne dit pas combien de malades ont été manqués : c'est pourquoi la décision se lit avec la matrice + le rappel.")

    # --- Phase 13 ---
    st.subheader("Phase 13 — Choix du seuil de décision")
    q1, q2 = st.columns(2)
    q1.metric("Seuil par défaut", "0,50")
    q2.metric("Seuil retenu (model_meta.json)", f"{float(meta['seuil']):.3f}".replace(".", ","))
    st.markdown("""
- Le seuil transforme une **probabilité** en **décision**. 0,50 n'est qu'une convention.
- Ici, le seuil est **calculé sur des probabilités out-of-fold du train** pour garantir un **rappel ≥ 90 %**, puis appliqué une seule fois au test.
- Conséquence : moins de malades manqués (**faux négatifs** ↓), plus de fausses alertes (**faux positifs** ↑) — compromis acceptable en pré-dépistage, puisqu'une alerte mène à un contrôle médical.
""")

    # --- Phase 14 ---
    st.subheader("Phase 14 — Contrôle visuel : réel vs prédit (test)")
    st.markdown("""
**Ce que montre la figure du notebook** : chaque point = un patient du test ; axe X = classe réelle (0/1, légèrement étalée pour éviter les superpositions), axe Y = probabilité prédite ; la ligne rouge est le seuil.
**Comment lire** : au-dessus du seuil = prédit « malade ». Les points rouges en bas à droite sont des **faux négatifs** (malades classés sains) ; les points orange en haut à gauche des **faux positifs**.
**Interprétation** : la majorité des malades est au-dessus du seuil → le rappel est élevé ; les erreurs restantes sont peu nombreuses et visibles.
""")

    # --- Phase 15 ---
    st.subheader("Phase 15 — Importance des variables (permutation)")
    st.markdown("""
**Principe** : on mélange au hasard une variable dans le test et on mesure la baisse d'AUC. Plus la baisse est forte, plus la variable est importante.
**Pourquoi cette méthode** : elle fonctionne pour tous les modèles, contrairement aux coefficients (modèles linéaires seulement) et aux importances Gini (biaisées vers les variables à nombreuses modalités).

**Classement obtenu (notebook, Phase 15)** :
1. `cp` (type de douleur thoracique) — de loin la plus contributive
2. `oldpeak` (dépression du segment ST à l'effort)
3. `thalch` (fréquence cardiaque maximale)
4. `exang` (angine induite par l'effort)
5. `ca` (nombre de vaisseaux colorés)
6. `age`

`chol` apparaît artificiellement haut car l'indicateur de manquant (`add_indicator`) est corrélé à l'hôpital d'origine : à interpréter avec prudence.
""")

    # --- Phase 16 ---
    st.subheader("Phase 16 — Test sur de nouvelles données (formulaire)")
    st.markdown("""
- **Ce qu'on teste** : des patients fictifs saisis comme dans un formulaire, **hors** train et test, pour vérifier le comportement opérationnel.
- **3 modèles** (régression logistique, Random Forest, KNN) reçoivent **exactement les mêmes données** : on compare probabilité, classe et décision par seuil.
- **Où ?** → page **« 3. Prédiction / Verdict »**. L'« attendu » y est une attente métier/démonstration, pas une vérité clinique.
""")

    # --- Phase 17 ---
    st.subheader("Phase 17 — Export du pipeline et déploiement")
    st.markdown("""
- **Règle** : on exporte le **pipeline complet** (préparation + modèle), pas seulement le modèle — sinon il faudrait refaire l'imputation et l'encodage à la main dans l'app, avec un risque d'écart.
- **`model.joblib`** : pipeline entraîné sur **toutes** les données (après évaluation).
- **`model_meta.json`** : seuil calculé, bornes min/médiane/max, listes de catégories, citation de la source.
""")
    st.code("""# Export (notebook / entraînement)
joblib.dump(pipeline, "model.joblib")
json.dump(meta, open("model_meta.json", "w"))

# Consommation (Streamlit app)
model = joblib.load("model.joblib")
meta = json.load(open("model_meta.json"))
proba = model.predict_proba(X_new)[:, 1]""", language="python")
    st.info("**Joblib vs JSON dans notre cas** : `joblib` est meilleur pour sauvegarder le pipeline/modèle sklearn entraîné (objet Python complexe). `JSON` est mieux pour des métadonnées lisibles : seuil, colonnes, min/max/médiane, catégories, citation. On utilise donc **les deux**.")

    # --- Phase 18 ---
    st.subheader("Phase 18 — Conclusion")
    st.markdown("""
### Les deux réponses clés

**1) Meilleur modèle : la régression logistique.**

| Modèle | AUC CV | AUC test | Rappel (0,50) | FN |
|---|---:|---:|---:|---:|
| **Régression logistique (retenue)** | **0,883** | **0,927** | **0,90** | **10 (8 au seuil 0,41)** |
| Random Forest | 0,876 | 0,917 | 0,87 | 13 |
| KNN | 0,864 | 0,898 | 0,87 | 13 |
| Arbre de décision | 0,804 | — | — | — |

**2) Meilleure stratégie de valeurs manquantes : médiane / mode** (à égalité de modèle).

| Stratégie | ROC-AUC test | CV ROC-AUC | Complexité |
|---|---:|---:|---|
| **Médiane / mode (retenue)** | **0,9133** | **0,8756** | une valeur apprise par variable |
| Patient similaire (KNN) | 0,9116 | 0,8747 | stocke tous les patients, distances à calculer |
| Suppression `ca` / `thal` | 0,8987 | 0,8626 | jette deux examens |

Le KNN garde un meilleur rappel au seuil 0,50 mais passe derrière sur la discrimination, et `ca` manquant à 66 % rend ses voisins peu informatifs. Les IC bootstrap se recoupent : on retient donc la médiane/mode pour sa **simplicité** et parce que c'est **elle qui est exportée** ici.
""")
    st.markdown("""
- **Q1 — Facteurs associés** (Phase 7, mesure adaptée au type) : côté **catégories** `cp` 0,538 · `thal` 0,497 · `exang` 0,461 · `slope` 0,377 · `sex` 0,304 ; côté **numériques** `ca` 0,456 · `thalch` 0,393 · `oldpeak` 0,386 · `age` 0,282. Sens : `cp = asymptomatic` 79 % de malades, `exang = True` 83,6 %, `thal` anormal ~76–80 %, `slope` flat/downsloping ~77 %.
- **Q2 — Variables les plus contributives** (permutation) : `cp` très loin devant, puis `chol`, `oldpeak`, `ca`, `exang`. `age` et `thalch` ressortent à ≈ 0 (information déjà portée par les autres variables).
- **Q3 — Capacité prédictive** : AUC test **0,927** (CV 0,883), accuracy **0,848**, rappel 0,902 (baseline 0,55).
- **Q4 — Modèle retenu** : régression logistique (AUC CV 0,883), devant Random Forest (0,876), KNN (0,864) et arbre seul (0,804) ; aucune des 8 familles testées ne fait mieux.
- **Q4bis — Valeurs manquantes** : médiane/mode en tête (ROC-AUC test 0,913 / CV 0,876), devant KNN (0,912 / 0,875) et suppression (0,899 / 0,863), avec des IC qui se recoupent : c'est aussi la stratégie exportée dans l'app.
- **Q5 — Fiabilité** : rappel 0,90 au seuil 0,50 (10 faux négatifs / 102 malades, 18 faux positifs / 82 sains) et **0,92 au seuil 0,41** (8 faux négatifs, 23 faux positifs) ; **aide à la décision, jamais un diagnostic**.
- **Limites** : 918 patients (920 lignes brutes − 2 doublons exacts), 4 centres, `ca`/`thal` très incomplètes, données rétrospectives. **Perspectives** : validation externe, calibration des probabilités, suivi de dérive du modèle.
""")
    st.caption("Le détail complet (sorties, figures, interprétations) se trouve dans le notebook, page « 4. Notebook (prioritaire) ».")

    with st.expander("Rappel — les 8 étapes génériques de tout projet ML"):
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
        st.caption("Ces 8 étapes génériques se déclinent en 18 phases détaillées dans ce projet.")


# -------------------- PAGE 3 --------------------
elif page.startswith("3"):
    st.header("3) Prédiction / Verdict final (formulaire)")
    st.markdown("Saisir les données patient. Les 3 modèles comparent la même entrée. **Aide à la décision, pas décision automatique.**")

    with st.expander("Voir la fonction simple predire_scenario (style demandé, adapté CardioRisk binaire)"):
        st.code("""ORDER = ["Sain (0)", "Malade (1)"]

def predire_scenario(valeurs: dict, seuil_alerte: float = 0.41):
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
**La régression logistique est le meilleur modèle du projet** : meilleure AUC en validation croisée (0,883) **et** sur le test (0,927), et elle est la plus explicable (un coefficient par variable).

### Ce que dit le test (184 patients jamais vus)
| Modèle | AUC | Rappel (0,50) | FN | FP |
|---|---:|---:|---:|---:|
| **Régression logistique** | **0,927** | **0,902** | **10** | 18 |
| Random Forest | 0,917 | 0,873 | 13 | 18 |
| KNN | 0,898 | 0,873 | 13 | 14 |

- **Rappel** = TP/(TP+FN) : part des malades réellement détectés.
- **FN (faux négatif)** : malade prédit sain → risque critique, c'est ce qu'on minimise.
- **FP (faux positif)** : sain prédit malade → fausse alerte, coût de temps seulement.
- Au **seuil retenu (0,41)**, la régression logistique ne manque que **8 malades sur 102** (contre 10 au seuil 0,50).

**KNN gagne sur l'accuracy et la précision** (0,853 / 0,864 contre 0,848 / 0,836) avec moins de fausses alertes, **mais il détecte moins de malades et classe moins bien les probabilités (AUC 0,898)**. En pré-dépistage, ce compromis n'est pas acceptable.

> **Conclusion** : régression logistique = **le meilleur modèle** (AUC test, rappel, explicabilité). Un modèle qui rate moins de malades et se laisse auditer cliniquement prime sur quelques points d'accuracy.
""")
        cm_cols = st.columns(3)
        for i,(name,cm) in enumerate(cms.items()):
            tn,fp,fn,tp = cm.ravel()
            fig_cm,ax_cm = plt.subplots(figsize=(4.2,3.4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm, xticklabels=['Prédit Sain','Prédit Malade'], yticklabels=['Réel Sain','Réel Malade'])
            ax_cm.set_title(name, fontsize=9)
            with cm_cols[i%3]:
                show_fig(fig_cm, f"TN={tn} FP={fp} FN={fn} TP={tp}")

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
                from sklearn.impute import SimpleImputer, KNNImputer
                from sklearn.preprocessing import OneHotEncoder, StandardScaler
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                RS = 42
                dfm = clean_data().copy()                 # même nettoyage de base que la page 2
                NUM_ALL = list(meta["num"]); CAT_ALL = list(meta["cat"])
                for c in CAT_ALL:
                    dfm[c] = dfm[c].astype(str).where(dfm[c].notna(), np.nan)
                X = dfm[NUM_ALL + CAT_ALL].copy()
                y = dfm['target']
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)

                def preproc(num_cols, cat_cols):
                    return ColumnTransformer([
                        ('num', Pipeline([('imp', SimpleImputer(strategy='median', add_indicator=True)), ('sc', StandardScaler())]), num_cols),
                        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),
                    ])

                def evalp(pipe, Xtr, ytr, Xte, yte):
                    pipe.fit(Xtr, ytr)
                    pred = pipe.predict(Xte)
                    proba = pipe.predict_proba(Xte)[:, 1]
                    return {
                        'Accuracy': accuracy_score(yte, pred),
                        'Precision': precision_score(yte, pred, zero_division=0),
                        'Recall': recall_score(yte, pred, zero_division=0),
                        'F1-Score': f1_score(yte, pred, zero_division=0),
                        'ROC-AUC': roc_auc_score(yte, proba),
                    }

                def cv_pipe(pipe, Xcv, ycv):
                    sc = cross_validate(pipe, Xcv, ycv, cv=cv,
                                        scoring={'Accuracy': 'accuracy', 'Precision': 'precision', 'Recall': 'recall', 'F1': 'f1', 'ROC-AUC': 'roc_auc'},
                                        n_jobs=-1)
                    return {f'CV {k}': float(sc[f'test_{k}'].mean()) for k in ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']}

                # Exp 1 — supprimer ca et thal
                Xtr1, Xte1 = Xtr.drop(columns=['ca', 'thal']), Xte.drop(columns=['ca', 'thal'])
                num1 = [c for c in NUM_ALL if c != 'ca']
                cat1 = [c for c in CAT_ALL if c != 'thal']
                m1 = evalp(Pipeline([('pre', preproc(num1, cat1)), ('rf', RandomForestClassifier(n_estimators=300, random_state=RS, n_jobs=-1))]), Xtr1, ytr, Xte1, yte)
                m1.update(cv_pipe(Pipeline([('pre', preproc(num1, cat1)), ('rf', RandomForestClassifier(n_estimators=300, random_state=RS, n_jobs=-1))]), Xtr1, ytr))

                # Exp 2 — médiane / mode (baseline)
                m2 = evalp(Pipeline([('pre', preproc(NUM_ALL, CAT_ALL)), ('rf', RandomForestClassifier(n_estimators=300, random_state=RS, n_jobs=-1))]), Xtr.copy(), ytr, Xte.copy(), yte)
                m2.update(cv_pipe(Pipeline([('pre', preproc(NUM_ALL, CAT_ALL)), ('rf', RandomForestClassifier(n_estimators=300, random_state=RS, n_jobs=-1))]), Xtr, ytr))

                # Exp 3 — patient similaire (KNN)
                THAL_MAP = {'normal': 0, 'fixed defect': 1, 'reversable defect': 2}
                THAL_INV = {v: k for k, v in THAL_MAP.items()}
                SIM_NUM = [c for c in NUM_ALL if c != 'ca']
                SIM_CAT = [c for c in CAT_ALL if c != 'thal']

                def impute_similarity(X_fit, X_apply, n_neighbors=5):
                    num_imp = SimpleImputer(strategy='median').fit(X_fit[SIM_NUM])
                    cat_fill = X_fit[SIM_CAT].mode(dropna=True).iloc[0]
                    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(X_fit[SIM_CAT].fillna(cat_fill))

                    def build(df):
                        num = pd.DataFrame(num_imp.transform(df[SIM_NUM]), columns=SIM_NUM, index=df.index)
                        cat = pd.DataFrame(ohe.transform(df[SIM_CAT].fillna(cat_fill)), columns=ohe.get_feature_names_out(SIM_CAT), index=df.index)
                        ca = df[['ca']].astype(float)
                        thal = df['thal'].map(THAL_MAP).rename('thal_code').to_frame()
                        return pd.concat([num, cat, ca, thal], axis=1)

                    A_fit = build(X_fit)
                    A_apply = build(X_apply)
                    knn = KNNImputer(n_neighbors=n_neighbors, weights='uniform').fit(A_fit)
                    A_fit_imp = pd.DataFrame(knn.transform(A_fit), columns=A_fit.columns, index=A_fit.index)
                    A_apply_imp = pd.DataFrame(knn.transform(A_apply), columns=A_apply.columns, index=A_apply.index)

                    def restore(df_orig, A_imp):
                        out = df_orig.copy()
                        out['ca'] = A_imp['ca'].round().clip(0, 3).astype(float)
                        codes = A_imp['thal_code'].round().clip(0, 2).astype(int)
                        out['thal'] = codes.map(THAL_INV)
                        out['ca_missing'] = df_orig['ca'].isna().astype(int)
                        out['thal_missing'] = df_orig['thal'].isna().astype(int)
                        return out

                    return restore(X_fit, A_fit_imp), restore(X_apply, A_apply_imp)

                def cv_similarity(Xcv, ycv):
                    rows = []
                    for tr_idx, va_idx in cv.split(Xcv, ycv):
                        Xtr_imp, Xva_imp = impute_similarity(Xcv.iloc[tr_idx], Xcv.iloc[va_idx])
                        pipe = Pipeline([('pre', preproc(NUM_ALL + ['ca_missing', 'thal_missing'], CAT_ALL)), ('rf', RandomForestClassifier(n_estimators=300, random_state=RS, n_jobs=-1))])
                        pipe.fit(Xtr_imp, ycv.iloc[tr_idx])
                        pred = pipe.predict(Xva_imp)
                        proba = pipe.predict_proba(Xva_imp)[:, 1]
                        rows.append({
                            'Accuracy': accuracy_score(ycv.iloc[va_idx], pred),
                            'Precision': precision_score(ycv.iloc[va_idx], pred, zero_division=0),
                            'Recall': recall_score(ycv.iloc[va_idx], pred, zero_division=0),
                            'F1-Score': f1_score(ycv.iloc[va_idx], pred, zero_division=0),
                            'ROC-AUC': roc_auc_score(ycv.iloc[va_idx], proba),
                        })
                    cv_table = pd.DataFrame(rows).rename(columns={'F1-Score': 'F1'})
                    return {f'CV {k}': float(cv_table[k].mean()) for k in ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']}

                Xtr3, Xte3 = impute_similarity(Xtr, Xte)
                m3 = evalp(Pipeline([('pre', preproc(NUM_ALL + ['ca_missing', 'thal_missing'], CAT_ALL)), ('rf', RandomForestClassifier(n_estimators=300, random_state=RS, n_jobs=-1))]), Xtr3, ytr, Xte3, yte)
                m3.update(cv_similarity(Xtr, ytr))

                return pd.DataFrame({
                    'Exp 1 - Suppression ca,thal': m1,
                    'Exp 2 - Médiane/mode (baseline)': m2,
                    'Exp 3 - Patient similaire (KNN)': m3,
                }).T.round(4)
            try:
                cmp_na = compare_missing_strategies()
                st.dataframe(cmp_na, use_container_width=True)
                best_na = cmp_na['ROC-AUC'].astype(float).idxmax()
                st.success(f"**Meilleure stratégie selon ROC-AUC test** : {best_na}. (Même split, même RandomForest, CV 5-fold ; imputation KNN refaite dans chaque pli)")
            except Exception as e:
                st.caption(f"Comparaison indisponible: {e}")
            # scatter réel vs prédit for best_recall model
            try:
                from sklearn.model_selection import train_test_split
                df_all = clean_data()                     # même nettoyage que les pages 2 et 3
                NUM=list(meta["num"]); CAT=list(meta["cat"])
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
                ax_s.legend(); show_fig(fig_s)
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
