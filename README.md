#  Datathon BCG – Groupe Bison Fûté

## Objectif du projet

L’objectif de ce projet est de **prédire le débit horaire de circulation et le taux d'occupation** sur trois axes parisiens :  
- **Champs-Élysées**  
- **Convention**  
- **Saint-Pères**  

Les prédictions s’appuient sur des données **temporelles**, **calendaires** et **météorologiques**.

 Le livrable final est un fichier **`output_bison_futé.csv`**, contenant les **prédictions du 9 au 11 novembre 2025** pour chaque axe.

---

##  Organisation du dépôt


- datathon-BCG/
  - data/
    - traffic/        : Données de trafic brutes par axe
    - meteo/          : Données météorologiques (Open-Meteo)
    - vacances/       : Calendrier scolaire et jours fériés
  - notebooks/
    - eda.ipynb                       : Analyse exploratoire (EDA)
    - forecasting_champs.ipynb        : Modélisation LSTM – Champs-Élysées
    - forecasting_convention.ipynb    : Modélisation XGBoost – Convention
    - forecasting_peres.ipynb         : Modélisation XGBoost – Saint-Pères
    - final_training_prediction.ipynb : Prédictions finales et génération du CSV
  - src/
    - preprocess.py       : Fonctions de préparation et enrichissement des données
    - xgb_forecasting.py  : Entraînement et prédiction via XGBoost
    - lstm_forecasting.py : Entraînement et prédiction via LSTM
---

##  Pipeline de Préparation des Données

La fonction centrale **`pipeline()`** orchestre la préparation complète des données :

### Étapes principales :
1. **Ordonnancement temporel**  
   Tri du DataFrame par date et heure pour garantir la cohérence chronologique.

2. **Création des variables temporelles**  
   Extraction : mois, jour, heure, jour de la semaine, indicateur week-end.

3. **Identification des vacances scolaires**  
   Ajout d’une variable binaire indiquant les vacances de la **zone C (Paris)**.

4. **Ajout des jours fériés et types de jours**  
   Intégration d’un indicateur `day_type` : ouvré, week-end, férié, etc.

5. **Encodage cyclique**  
   Transformation sinusoïdale des variables temporelles (heures, jours, mois) pour capturer la cyclicité.

6. **Gestion des valeurs manquantes**  
   Interpolation temporelle des variables cibles afin d’assurer la continuité des séries.

7. **Fusion avec les données météorologiques**  
   Variables intégrées : température, vent, précipitations, couverture nuageuse.

8. **Détection d’événements particuliers**  
   Marquage d’événements influençant le trafic (courses, cérémonies, 14 juillet, Nouvel An, etc.).

9. **Lissage optionnel**  
   Application d’une **moyenne glissante** si le paramètre `window > 0` est défini.

---

##  Modélisation

### Choix des modèles

| Axe | Modèle | Justification |
|-----|---------|----------------|
| **Champs-Élysées** | LSTM | Historique complet et régulier → idéal pour capturer les dépendances temporelles |
| **Convention** | XGBoost | Données plus bruitées et discontinues → modèle robuste aux valeurs manquantes |
| **Saint-Pères** | XGBoost | Idem Convention |

---

### Entraînement et validation

- Entraînement sur les données jusqu’au **7 novembre 2025** inclus
- Validation via **cross-validation temporelle**  
- Prédiction des **72 heures suivantes (9–11 novembre 2025)**  

#### Détails techniques :
- Les modèles **LSTM** utilisent des fenêtres glissantes de **168 heures (7 jours)** pour prédire les **72 heures suivantes**.  
- Les modèles **XGBoost** exploitent un jeu de features enrichi incluant les variables temporelles et météorologiques.

---

##  Génération du fichier final

Les prédictions issues des trois modèles sont **concaténées** pour produire le fichier :

**`output_bison_futé.csv`**

### Format de sortie

| Colonne | Description |
|----------|--------------|
| `arc` | Nom de l’axe routier (Champs-Élysées, Convention, Saint-Pères) |
| `datetime` | Horodatage de la prédiction (`%Y-%m-%d %H:%M`) |
| `debit_horaire` | Prédiction du débit horaire (véhicules/heure) |
| `taux_occupation` | Prédiction du taux d’occupation (%) |

---

##  Synthèse

Ce projet combine :
-  **Un pipeline de préparation robuste** intégrant météo et calendrier  
-  **Deux approches de modélisation complémentaires (LSTM & XGBoost)**  
-  **Un processus automatisé** de prédiction multi-axes  
-  **Un rendu final homogène et exploitable** pour l’analyse du trafic parisien

---


**Groupe Bison Fûté – Datathon BCG 2025**  
Projet académique réalisé dans le cadre du datathon BCG.  

