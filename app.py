import base64
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
    "age": ("Âge (années)", "Âge du patient. Le risque cardiovasculaire augmente souvent avec l'âge, mais l'âge seul ne suffit pas."),
    "trestbps": ("Pression artérielle au repos (mm Hg)", "Tension au repos. Valeurs très élevées augmentent le risque ; dans ce dataset, pouvoir discriminant modéré."),
    "chol": ("Cholestérol sérique (mg/dl)", "Cholestérol total. Utile mais peut être bruité/manquant ; `chol==0` traité comme valeur invalide/manquante."),
    "thalch": ("Fréquence cardiaque maximale atteinte", "Max heart rate à l'effort. Souvent plus bas chez les patients malades (moindre tolérance à l'effort)."),
    "oldpeak": ("Dépression ST à l'effort (oldpeak)", "Mesure d'anomalie ECG à l'effort. Plus élevé = signal plus associé à la maladie."),
    "ca": ("Nombre de vaisseaux majeurs colorés (0-3)", "Examen fluoroscopie. Variable forte mais très manquante ; on l'impute avec indicateur de manquant."),
    "sex": ("Sexe", "Variable démographique/catégorielle. Dans ce dataset, les hommes ont un taux de maladie plus élevé."),
    "cp": ("Type de douleur thoracique", "Symptôme clé. `asymptomatic` a un taux de maladie élevé dans ce dataset (maladie silencieuse possible)."),
    "fbs": ("Glycémie à jeun > 120 mg/dl", "Indicateur glycémique. Signal faible seul, mais peut contribuer avec les autres."),
    "restecg": ("ECG au repos", "Résultat électrocardiogramme au repos. Certaines anomalies augmentent le risque."),
    "exang": ("Angine induite par l'effort", "Douleur à l'effort. Très informative : oui -> taux de maladie plus élevé."),
    "slope": ("Pente du segment ST à l'effort", "Interprétation ECG effort. `flat`/`downsloping` plus associés à la maladie que `upsloping`."),
    "thal": ("Thalassémie/scintigraphie", "Résultat examen. `reversable/fixed defect` associés à plus de maladie que `normal`, mais beaucoup de NA."),
}


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
def process_figure():
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
    return fig


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
    Xtr, _, ytr, _ = train_test_split(Xall, yall, test_size=0.2, stratify=yall, random_state=42)
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT),
    ])
    defs = {
        "Régression logistique": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }
    fitted = {}
    for n, m in defs.items():
        pipe = Pipeline([("pre", pre), ("model", m)])
        pipe.fit(Xtr, ytr)
        fitted[n] = pipe
    return fitted


model, meta = load_model()
df = load_data()

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

    st.subheader("Processus ML (image)")
    fig_proc = process_figure()
    st.pyplot(fig_proc)
    plt.close(fig_proc)
    st.caption("Schéma du processus : données brutes → audit/nettoyage → EDA → split → pipeline/modèles → évaluation → seuil → export (joblib/json) → app Streamlit.")

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
    st.header("2) Processus ML — pas à pas (explications)")

    st.subheader("Étape 1 — Lecture des fichiers / compréhension")
    st.markdown("""
- **Action** : charger `heart_disease_uci.csv`, afficher `head`, `shape`, `info`, `describe`.
- **Utilité** : comprendre la structure, les types, les plages de valeurs, et détecter anomalies.
- **Observation dataset** : 920 observations, 16 colonnes brutes, 5 centres.
""")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", df.shape[0])
    c2.metric("Colonnes brutes", 16)
    c3.metric("Sains (0)", int((df["target"] == 0).sum()))
    c4.metric("Malades (1)", int((df["target"] == 1).sum()))
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Étape 2 — Détection qualité : NA, doublons, types")
    st.markdown("""
- **Valeurs manquantes** : `ca`, `thal`, `slope` très incomplètes ; `chol`, `fbs`, `oldpeak`, `trestbps`, `thalch`, `exang` partiellement.
- **Doublons** : à vérifier avec `df.duplicated().sum()`.
- **Zéros invalides** : `chol==0` et `trestbps==0` sont médicalement impossibles → convertis en NaN.
- **Variables catégorielles** : `sex, cp, fbs, restecg, exang, slope, thal` (texte/catégories).
- **Variables numériques** : `age, trestbps, chol, thalch, oldpeak, ca`.
""")
    miss = df.isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0]
    st.bar_chart(miss)
    st.caption("Barres = nombre de valeurs manquantes par colonne. Plus la barre est longue, plus il faut une stratégie d'imputation prudente.")

    st.subheader("Étape 3 — Data preparation / cleaning / preprocessing")
    st.markdown("""
**Pourquoi nettoyer/préparer ?** Les modèles ne gèrent pas bien les NaN, les échelles différentes, ni le texte brut.

| Problème | Traitement | Pourquoi |
|---|---|---|
| `chol==0`, `trestbps==0` | Remplacer par NaN | Valeur impossible, ce n'est pas une vraie mesure |
| NA numériques | **Médiane** (`SimpleImputer(strategy='median')`) | Robuste aux valeurs extrêmes/outliers ; la moyenne serait tirée par les valeurs extrêmes |
| NA catégorielles | **Mode** (`most_frequent`) | Pour une catégorie, la valeur la plus fréquente est le remplacement simple ; on ne peut pas faire de moyenne sur du texte |
| Catégorielles | **OneHotEncoder** | Transformer texte en colonnes 0/1 sans ordre artificiel |
| Numériques | **StandardScaler** | Mettre à la même échelle (moyenne 0, écart-type 1), important pour KNN/régression logistique |
| NA important (`ca`) | `add_indicator=True` | Ajoute une colonne “valeur manquante” pour que le modèle distingue un vrai 0 d'un 0 imputé |

**Anti-fuite** : tout `fit` (médiane, mode, scaler, encoder) est fait **sur le train uniquement** dans un `Pipeline`.
""")
    st.info("""
**Export (fin du processus, pour l'app)**  
Après entraînement/évaluation, on sauvegarde : `model.joblib` (pipeline complet) et `model_meta.json` (seuil calculé, bornes, catégories, citation).  
Ainsi l'application Streamlit charge le modèle sauvegardé pour prédire, sans réentraîner.
""")

    st.subheader("Étape 4 — Cible (target)")
    st.markdown("""
- `num` = 0..4 (gravité). Pour classification binaire : **`target = 1` si `num > 0` sinon `0`**.
- **Pourquoi ?** Simplifie en dépistage : malade / non malade. Les classes sont assez équilibrées (~55% / 45%).
""")

    st.subheader("Étape 5 — Étude de corrélation")
    st.markdown("""
**Pourquoi ?** Mesurer le lien linéaire entre variables numériques et cible, détecter redondance, orienter l'interprétation.

**Comment lire la heatmap :**
- Valeur entre **-1 et +1**.
- **+1** : les deux variables augmentent ensemble.
- **-1** : quand l'une augmente, l'autre baisse.
- **0** : pas de relation linéaire claire.
- **Diagonale = 1** : corrélation d'une variable avec elle-même.
- Règle pratique : |r| < 0.3 faible, 0.3-0.7 modérée, >0.7 forte.

**Attention :** corrélation ≠ causalité ; ne capte que le linéaire.
""")
    fig_corr, corr = correlation_figure(df)
    st.pyplot(fig_corr)
    plt.close(fig_corr)
    tc = corr["target"].drop("target").sort_values(key=lambda s: s.abs(), ascending=False)
    st.write("Corrélation avec la cible (triée par importance absolue):")
    st.dataframe(tc.rename("corr_avec_target").to_frame(), use_container_width=True)
    st.info("Interprétation : `ca`, `oldpeak` positifs ; `thalch` négatif ; `chol`/`trestbps` faibles. `ca` est calculé sur peu de valeurs (beaucoup de NA).")

    st.subheader("Étape 6 — Features, split, modèles")
    st.markdown("""
- **Features** : variables numériques + catégorielles (sauf `id`, `dataset`, `num`, `target`).
- **Split** : 80% train / 20% test, `stratify=y`, `random_state=42`.
- **Pourquoi stratify ?** garder la même proportion malades/sains.
- **Modèles comparés** : Baseline, Régression logistique, Arbre, KNN, Random Forest.
- **Random Forest (pourquoi/arguments)** : ensemble d'arbres qui votent, capture non-linéarités/interactions, robuste. Ex. `n_estimators=300` (300 arbres), `min_samples_leaf=3` (évite feuilles trop petites/sur-apprentissage), `random_state=42` (reproductible).
""")

    st.subheader("Étape 7 — Évaluation : R², MAE, MSE, Recall etc.")
    st.markdown("""
**Métriques demandées VIC :**
- **R² (coefficient de détermination)** : 1 - SS_res/SS_tot. 1 parfait, 0 = modèle naïf (moyenne), <0 pire. En classification, on le rapporte sur probabilité et on l'explique avec prudence.
- **MAE (Mean Absolute Error)** : moyenne des |erreur|. Plus bas = mieux. Facile à lire : erreur moyenne.
- **MSE (Mean Squared Error)** : moyenne des erreurs². Pénalise fortement les grosses erreurs.

**Métriques classification (importantes ici) :**
- **Accuracy** : % correct global. Peut tromper si déséquilibre.
- **Precision** : parmi les prédits malades, combien sont vraiment malades.
- **Recall (Rappel / Sensibilité)** : parmi les vrais malades, combien sont détectés. **Le plus important dans notre cas de pré-dépistage** car manquer un malade (faux négatif) est grave.
- **F1** : compromis précision/rappel.
- **AUC** : capacité à séparer sain/malade sur tous les seuils (0.5 hasard, 1 parfait).

**Pourquoi Recall prioritaire ?** En pré-dépistage, un faux négatif = malade non détecté. On accepte plus de fausses alertes pour réduire les malades manqués (ajustement du seuil).

**Étape 8 — Export pour l'application (joblib/json)**
- **Pourquoi exporter ?** Pour que l'app Streamlit utilise le modèle entraîné sans réentraîner à chaque clic.
- **`model.joblib`** : contient le pipeline entraîné (prétraitement + modèle).
- **`model_meta.json`** : contient les métadonnées nécessaires au formulaire et à la décision (seuil calculé, bornes min/max/médiane, catégories, citation source).
- **Important** : le seuil sauvegardé doit provenir de l'évaluation (validation croisée/train), pas être écrit à la main.
""")

# -------------------- PAGE 3 --------------------
elif page.startswith("3"):
    st.header("3) Prédiction / Verdict final (formulaire)")
    st.markdown("Saisir les données patient. Les 3 modèles comparent la même entrée.")

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
        st.markdown("### Données saisies")
        st.dataframe(X, use_container_width=True)

        models3 = train_compare_models()
        rows = []
        for name, m in models3.items():
            prob = float(m.predict_proba(X)[0, 1])
            pred_i = int(prob >= seuil)
            rows.append({
                "Modèle": name,
                "Probabilité malade %": round(prob * 100, 2),
                "Seuil décision %": round(seuil * 100, 2),
                "Prédiction": "Malade (1)" if pred_i else "Sain (0)",
                "Verdict": "⚠️ Risque élevé" if pred_i else "✅ Risque faible",
            })
        st.markdown("### Comparaison des 3 modèles (mêmes données)")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        main_p = float(models3["Régression logistique"].predict_proba(X)[0, 1])
        verdict = "MALADE / RISQUE ÉLEVÉ" if main_p >= seuil else "SAIN / RISQUE FAIBLE"
        st.markdown("### Verdict final")
        if main_p >= seuil:
            st.error(f"**Verdict : {verdict}** — Probabilité {main_p:.1%} ≥ seuil {seuil:.1%} (seuil calculé pour favoriser le rappel/pré-dépistage). Recommandation : avis médical / examens complémentaires.")
        else:
            st.success(f"**Verdict : {verdict}** — Probabilité {main_p:.1%} < seuil {seuil:.1%}. Pas d'alerte modèle au seuil choisi, mais rester vigilant si symptômes.")

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
