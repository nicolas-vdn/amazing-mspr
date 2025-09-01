# Data Processing Pipeline avec Snowflake & GitHub Actions

## Description
Ce projet met en place une pipeline de traitement de données automatisée avec deux notebooks principaux.  
Chaque notebook est transformé en script Python et exécuté dans un conteneur Docker via des **GitHub Actions** planifiées.

---

## Architecture du Projet

### 1. **Notebook 1 – Ingestion & Transformation**
- Récupère les données **RAW_DATA** depuis **Snowflake**  
- Effectue les transformations nécessaires  
- Stocke les données transformées dans la table **TRANSFORMED_DATA** sur **Snowflake**

### 2. **Notebook 2 – Clustering & Traitement**
- Récupère les données depuis la table **TRANSFORMED_DATA**  
- Applique le modèle de clustering (IA/ML adapté)  
- Stocke les résultats traités dans la table **PROCESSED_DATA** sur **Snowflake**

---

## Exécution Automatisée

Chaque notebook dispose d’un **workflow GitHub Actions** associé :

- **Déclencheurs** :  
  - Tous les jours à **07h00 (UTC)** 
  - **Exécution manuelle** possible via l’interface GitHub Actions  

- **Étapes automatisées** :  
  - **Build** d’une image **Docker** spécifique au notebook  
  - Création d’un **container**  
  - Conversion du notebook en **script Python exécutable**  
  - Lancement du script dans le container  

---

## Stack Technique

- **Snowflake** : stockage et traitement de données  
- **Jupyter / Python** : langage principal pour les notebooks et scripts  
- **Docker** : conteneurisation des notebooks  
- **GitHub Actions** : orchestration CI/CD et scheduling
- **Matplotlib / Seaborn / Plotly** : analyse de données
- **Pandas / Scikit-learn / Kneed / Numpy / Matplotlib / Joblib** : transformations & clustering  

---

## Organisation des Données

- **RAW_DATA** : données sources non transformées  
- **TRANSFORMED_DATA** : données nettoyées et préparées  
- **PROCESSED_DATA** : données enrichies par le clustering et prêtes à l’usage  

---

## Utilisation Manuelle

1. Aller dans **GitHub Actions**  
2. Sélectionner le workflow du **Notebook 1** (Transformation) ou **Notebook 2** (Clustering)  
3. Cliquer sur **Run workflow**  

---

## Sécurité & Secrets

Les credentials Snowflake et autres variables sensibles sont stockés en tant que **GitHub Secrets**, injectés dans l’environnement au runtime.  
