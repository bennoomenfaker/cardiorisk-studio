# CardioRisk Studio — Prédiction de la maladie cardiaque (UCI)

Application Streamlit pédagogique d'aide à la décision pour estimer le risque de présence d'une maladie cardiaque (Oui/Non) à partir de données cliniques (UCI Heart Disease).

## Contenu
- `app.py` : dashboard Streamlit (contexte, dataset, notebook, EDA, corrélations, modèles/métriques, test réel vs prédit, prédiction/scénario)
- `heart_disease_uci.csv` : jeu de données brut
- `heart_disease_project_organized.ipynb` : notebook pédagogique organisé du projet
- `assets/` : images et visuels
- Livrables : rapport, fiches, exports CSV

## Installation locale
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

## Lancer l'application
```bash
streamlit run app.py
```

## Déploiement Streamlit Community Cloud
1. Pousser ce dépôt sur GitHub.
2. Sur [share.streamlit.io](https://share.streamlit.io), créer une nouvelle app.
3. Sélectionner le dépôt, la branche `main` et le fichier principal `app.py`.
4. Déployer.

## Avertissement
Outil pédagogique d'aide à l'analyse. Il ne constitue pas un dispositif médical et ne remplace pas un diagnostic professionnel (validation humaine indispensable).
