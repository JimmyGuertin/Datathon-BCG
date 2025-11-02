# 🚗 Datathon-BCG – Prédiction du Débit Horaire

## 🧠 Objectif
Ce projet vise à **prédire le débit horaire de circulation** sur un axe routier parisien à partir de **données temporelles, calendaires et météorologiques**.  
Le modèle principal est un **réseau de neurones LSTM**, entraîné sur des **séquences temporelles glissantes**.

---

## ⚙️ Pipeline de Préparation des Données

### 🗓️ 1. Ordonnancement temporel
Les données sont triées par date et heure afin d’assurer la cohérence chronologique et d’éviter le *data leakage*.

---

### 🕒 2. Extraction des composantes temporelles
À partir de la colonne `date`, on extrait plusieurs variables utiles :  
- Mois (`month`)  
- Jour (`day`)  
- Heure (`hour`)  
- Jour de la semaine (`weekday`)  
- Indicateur week-end (`is_weekend`)

---

### 🎒 3. Vacances scolaires
On identifie si une date correspond à une période de **vacances scolaires parisiennes** (zone C).  
Périodes principales : Toussaint, Noël, Hiver, Printemps, Été 2024-2025.  

---

### 🎉 4. Jours fériés
Création d’une variable binaire pour signaler les **jours fériés français** (1er janvier, 8 mai, 14 juillet, 25 décembre, etc.).

---

### 🔄 5. Encodage cyclique des variables temporelles
Pour capturer la **périodicité naturelle du temps** (heures, jours, mois), on encode les variables temporelles de manière cyclique.

---

### 🧩 6. Gestion des valeurs manquantes
Les valeurs manquantes de la variable cible (`Débit horaire`) sont **interpolées temporellement** afin de garantir la continuité du signal.

---

### 🌦️ 7. Fusion avec les données météorologiques
On fusionne les données trafic avec les données météorologiques issues de l’API **Open-Meteo**.  
Variables intégrées : température, vent, précipitations, couverture nuageuse.

---

## 🤖 Modélisation LSTM

### 🧱 1. Sélection des features
Les variables explicatives incluent les composantes temporelles, les indicateurs calendaires et les données météorologiques.

---

### ⚙️ 2. Normalisation
Toutes les features et la variable cible sont normalisées pour l’apprentissage du modèle.

---

### 🧮 3. Création des séquences temporelles
Des **séquences glissantes** de longueur 24h ou 168h sont créées pour alimenter le LSTM.

---

### 🔀 4. Split temporel train/test
Le découpage du jeu de données respecte la **chronologie** : pas de mélange aléatoire.

---

### 🧠 5. Modèle LSTM
Le modèle est un **réseau LSTM séquentiel** avec régularisation par Dropout et une couche dense pour la régression.

---

### 🏋️‍♂️ 6. Entraînement
Le modèle est entraîné sur le jeu d’entraînement et validé sur le jeu de test.

---

### 📊 7. Évaluation
L’évaluation se fait avec la **Root Mean Squared Error (RMSE)** et l’**erreur relative** sur le jeu de test.  
On peut ainsi mesurer la performance et la précision de la prédiction du débit horaire.
