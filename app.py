import base64
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CardioRisk Studio", page_icon="🫀", layout="wide")
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model.joblib"
META_PATH = ROOT / "model_meta.json"
DATA_PATH = ROOT / "heart_disease_uci.csv"
NOTEBOOK_PRIORITY = ROOT / "heart_disease_cardiorisk.ipynb"

FIELD_INFO = {
    "age": "Âge (années)",
    "trestbps": "Pression artérielle au repos (mm Hg)",
    "chol": "Cholestérol sérique (mg/dl)",
    "thalch": "Fréquence cardiaque maximale atteinte",
    "oldpeak": "Dépression ST à l'effort (oldpeak)",
    "ca": "Nombre de vaisseaux majeurs colorés (0-3)",
    "sex": "Sexe",
    "cp": "Type de douleur thoracique",
    "fbs": "Glycémie à jeun > 120 mg/dl",
    "restecg": "ECG au repos",
    "exang": "Angine induite par l'effort",
    "slope": "Pente du segment ST à l'effort",
    "thal": "Résultat thalassémie/scintigraphie",
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


def render_notebook(path: Path):
    if not path.exists():
        st.warning(f"Notebook introuvable : {path.name}")
        return
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    st.caption(f"{path.name} — {len(cells)} cellules")
    with open(path, "rb") as f:
        st.download_button("Télécharger le notebook (.ipynb)", f, file_name=path.name, mime="application/x-ipynb+json")
    for cell in cells:
        ctype = cell.get("cell_type")
        src = "".join(cell.get("source", []))
        if ctype == "markdown":
            st.markdown(src)
        elif ctype == "code":
            if src.strip():
                st.code(src, language="python")
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


model, meta = load_model()
df = load_data()

st.title("🫀 CardioRisk Studio")
st.caption("Prototype pédagogique d’aide à la décision — Heart Disease UCI")

page = st.sidebar.radio("Navigation", ["Contexte & EDA", "Prédiction", "Notebook (prioritaire)"], index=0)
st.sidebar.markdown("---")
st.sidebar.info("Outil pédagogique. Ne remplace pas un avis médical.")

if page == "Contexte & EDA":
    st.subheader("Contexte du projet")
    st.markdown("""
**Problématique** : Peut-on prédire la présence d’une maladie cardiaque à partir de données cliniques et d’examens ?

**Cible** : `target = 1` si `num > 0` (sténose > 50%), sinon `0`.

**Pipeline** : audit/nettoyage → EDA/corrélation → split stratifié 80/20 → pipeline anti-fuite (`SimpleImputer` + `add_indicator` + `StandardScaler` + `OneHotEncoder`) → comparaison modèles → évaluation + seuil de dépistage.

**Limites** : valeurs manquantes importantes (`ca`, `thal`, `slope`), données multi-centres historiques, outil pédagogique, validation humaine obligatoire.
""")
    if df is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Observations", df.shape[0])
        c2.metric("Colonnes", df.shape[1])
        target = (df["num"] > 0).astype(int)
        c3.metric("% malades", f"{target.mean():.1%}")
        c4.metric("Seuil dépistage", f"{float(meta['seuil']):.0%}")

        st.markdown("### Aperçu des données")
        st.dataframe(df.head(), use_container_width=True)

        st.markdown("### Qualité / valeurs manquantes")
        miss = df.isna().sum().sort_values(ascending=False)
        miss = miss[miss > 0]
        if not miss.empty:
            st.bar_chart(miss)
        st.caption("Les zéros invalides `chol==0` et `trestbps==0` sont traités comme manquants dans le pipeline du notebook.")

        st.markdown("### Distribution cible")
        dist = pd.DataFrame({"classe": ["Sain (0)", "Malade (1)"], "count": [(target==0).sum(), (target==1).sum()]})
        st.bar_chart(dist.set_index("classe"))

        st.markdown("### Corrélations numériques (exploratoire)")
        num_cols = [c for c in df.select_dtypes(include="number").columns if c not in ["id"]]
        corr = df[num_cols].corr(numeric_only=True)
        st.dataframe(corr.style.background_gradient(cmap="RdBu_r", axis=None), use_container_width=True)
        st.caption("Corrélation = association linéaire, pas causalité. L’interprétation détaillée est dans le notebook prioritaire.")
    st.info("Pour les explications figure par figure, ouvrir l’onglet **Notebook (prioritaire)**.")

elif page == "Prédiction":
    st.subheader("Estimation du risque")
    st.write("Renseigner les informations patient. Les champs utilisent les plages/valeurs du modèle entraîné.")
    patient = {}
    with st.form("formulaire"):
        colA, colB = st.columns(2)
        num_items = list(meta["num"].items())
        half = (len(num_items) + 1) // 2
        with colA:
            st.markdown("#### Mesures numériques")
            for col, s in num_items[:half]:
                label = FIELD_INFO.get(col, col)
                patient[col] = st.number_input(label, min_value=float(s["min"]), max_value=float(s["max"]), value=float(s["median"]), key=f"n_{col}")
        with colB:
            st.markdown("#### Mesures numériques (suite)")
            for col, s in num_items[half:]:
                label = FIELD_INFO.get(col, col)
                patient[col] = st.number_input(label, min_value=float(s["min"]), max_value=float(s["max"]), value=float(s["median"]), key=f"n_{col}")
        st.markdown("#### Variables catégorielles")
        cat_cols = st.columns(2)
        for i, (col, options) in enumerate(meta["cat"].items()):
            opts = [o for o in options if o is not None]
            label = FIELD_INFO.get(col, col)
            with cat_cols[i % 2]:
                patient[col] = st.selectbox(label, opts, index=0, key=f"c_{col}")
        envoyer = st.form_submit_button("Estimer le risque", use_container_width=True)

    if envoyer:
        X = pd.DataFrame([patient])
        seuil = float(meta["seuil"])
        st.markdown("### Résultat — comparaison 3 modèles (mêmes données formulaire)")
        # Train 3 models on demand for comparison using same data schema as notebook
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
            df_all.loc[df_all["chol"]==0,"chol"] = pd.NA
            df_all.loc[df_all["trestbps"]==0,"trestbps"] = pd.NA
            df_all["target"] = (df_all["num"]>0).astype(int)
            NUM = ["age","trestbps","chol","thalch","oldpeak","ca"]
            CAT = ["sex","cp","fbs","restecg","exang","slope","thal"]
            for c in CAT: df_all[c]=df_all[c].astype(str).where(df_all[c].notna(), pd.NA)
            Xall=df_all[NUM+CAT]; yall=df_all["target"]
            Xtr,Xte,ytr,yte=train_test_split(Xall,yall,test_size=0.2,stratify=yall,random_state=42)
            pre = ColumnTransformer([
                ("num", Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True)),("sc",StandardScaler())]), NUM),
                ("cat", Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]), CAT),
            ], remainder="drop")
            defs = {
                "Régression logistique": LogisticRegression(max_iter=2000),
                "Random Forest": RandomForestClassifier(n_estimators=300,min_samples_leaf=3,random_state=42),
                "KNN": KNeighborsClassifier(n_neighbors=7),
            }
            fitted={}
            for n,m in defs.items():
                pipe=Pipeline([("pre",pre),("model",m)])
                pipe.fit(Xtr,ytr)
                fitted[n]=pipe
            return fitted
        try:
            models3 = train_compare_models()
            rows=[]
            for name,m in models3.items():
                prob=float(m.predict_proba(X)[0,1])
                pred_i=int(prob>=seuil)
                rows.append({"Modèle":name,"Probabilité_%":round(prob*100,2),"Seuil_%":round(seuil*100,2),"Prédiction":"Malade (1)" if pred_i else "Sain (0)","Décision":"Au-dessus seuil" if pred_i else "Sous seuil"})
            res=pd.DataFrame(rows)
            st.dataframe(res, use_container_width=True)
            st.caption("Les 3 modèles reçoivent exactement les mêmes données saisies dans le formulaire. Comparer les probabilités/décisions aide à voir la robustesse.")
            main_p = float(models3["Régression logistique"].predict_proba(X)[0,1])
        except Exception as e:
            st.warning(f"Comparaison 3 modèles indisponible: {e}")
            main_p = float(model.predict_proba(X)[0,1])
        c1,c2,c3 = st.columns(3)
        c1.metric("Probabilité (modèle retenu)", f"{main_p:.1%}")
        c2.metric("Seuil de dépistage", f"{seuil:.1%}")
        c3.metric("Décision", "Risque élevé" if main_p>=seuil else "Risque faible")
        if main_p >= seuil:
            st.error("Au-dessus du seuil de dépistage : recommander un avis médical / examens complémentaires.")
        else:
            st.success("En dessous du seuil : pas d’alerte modèle, mais rester vigilant si symptômes/facteurs de risque.")
        st.progress(min(max(main_p, 0.0), 1.0))
        st.caption("Résultat indicatif issu d’un modèle pédagogique. Validation humaine obligatoire.")

    st.markdown("### Interprétation rapide")
    st.markdown("""
- **Probabilité** : estimation du modèle, pas un diagnostic.
- **Seuil** : choisi pour privilégier le dépistage (rappel) plutôt que manquer des malades.
- **Explications détaillées** : voir le notebook prioritaire (EDA, corrélation, matrice de confusion, permutation importance).
""")
    st.caption(meta.get("citation", ""))

else:
    st.subheader("Notebook prioritaire — CardioRisk")
    st.success("Ce notebook est la référence prioritaire : contexte, interprétations figure par figure, corrélation, pipeline anti-fuite, comparaison modèles, seuil, limites.")
    render_notebook(NOTEBOOK_PRIORITY)
