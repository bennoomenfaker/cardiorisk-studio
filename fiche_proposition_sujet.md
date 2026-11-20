# Fiche de proposition du sujet (à faire valider)

| Rubrique | À remplir |
|---|---|
| **Nom et prénom des étudiants** | Faker BEN NOOMEN, Haythem BEN BOUBAKER, Mariem DRIDI |
| **Secteur choisi** | Santé (cardiologie) |
| **Titre provisoire du projet** | Prédiction de la présence d’une maladie cardiaque à partir de données cliniques |
| **Problématique ou question métier** | Peut-on prédire la présence d’une maladie cardiaque à partir de variables cliniques de routine, afin d’aider au dépistage et à la priorisation des examens ? |
| **Source et lien du jeu de données** | UCI Heart Disease (Cleveland, Hungary, Switzerland, VA Long Beach) — CSV combiné : `heart_disease_uci.csv` ; UCI : https://archive.ics.uci.edu/dataset/45/heart+disease ; Kaggle : https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data |
| **Tâche prévue** | **Classification** binaire (cible `target` = 1 si `num > 0`, sinon 0) — comparaison d’une baseline et de plusieurs modèles (Régression Logistique, Random Forest, KNN) |

## Justification rapide
- **Données utilisables et décrites :** 920 observations, 16 variables cliniques, source publique reconnue (UCI).
- **Question métier claire :** aide au dépistage/pré-triage en santé.
- **Prédiction réalisable :** variable cible identifiable (`num` → `target` binaire).
- **Faisable avec le module :** pandas, matplotlib/seaborn, scikit-learn (prétraitement, split train/test, classification, métriques, interprétation).
