import json, joblib, pandas as pd, streamlit as st

st.set_page_config(page_title="CardioRisk Studio", page_icon="🫀")
model = joblib.load("model.joblib")
meta = json.load(open("model_meta.json"))

st.title("🫀 CardioRisk Studio")
st.warning("Outil pédagogique basé sur le jeu UCI Heart Disease. Ne remplace pas un avis médical.")

patient = {}
with st.form("formulaire"):
    for col, s in meta["num"].items():
        patient[col] = st.number_input(col, s["min"], s["max"], s["median"])
    for col, options in meta["cat"].items():
        patient[col] = st.selectbox(col, options)
    envoyer = st.form_submit_button("Estimer")

if envoyer:
    p = float(model.predict_proba(pd.DataFrame([patient]))[0, 1])
    st.metric("Probabilité estimée de maladie coronarienne", f"{p:.0%}")
    if p >= meta["seuil"]:
        st.error("Au-dessus du seuil de dépistage")
    else:
        st.success("En dessous du seuil de dépistage")
st.caption(meta["citation"])
