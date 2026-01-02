# Système de Perception de Trafic Urbain
### Phase 1 : Pipeline d'Entraînement & Data Intelligence

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch)
![YOLOv11](https://img.shields.io/badge/Model-YOLOv11-00FFFF?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Phase_1_Complete-success?style=for-the-badge)

---

## 📋 Vue d'ensemble
Ce module constitue le **moteur cognitif** du projet. Il a pour but d'entraîner une intelligence artificielle capable de détecter et classifier les flux urbains (véhicules, piétons, cyclistes) à partir de vues aériennes complexes.

L'objectif de cette **Phase 1** est de produire un modèle robuste ("poids") capable de gérer les défis spécifiques à l'imagerie drone : **haute densité**, **objets minuscules** et **déséquilibre de classes**.

---

## 📊 Data Intelligence (Analyse Exploratoire)
Avant tout entraînement, une analyse approfondie du dataset VisDrone a été réalisée pour calibrer l'architecture du réseau neuronal. Voici les insights clés traduits pour une compréhension métier.

### 1. Distribution des Classes
![Class Distribution](runs/analysis/class_distribution.png)
> **Le constat :** Le dataset est massivement dominé par les voitures ("car") et les piétons ("pedestrian"). Les deux-roues (vélos, tricycles) sont beaucoup plus rares.
>
> **Impact Métier :** Le modèle sera naturellement excellent pour compter le trafic routier lourd. Pour la "mobilité douce" (vélos), des stratégies de pondération spécifiques ont été appliquées pour éviter qu'ils ne soient ignorés.

### 2. Le Défi des "Objets Microscopiques"
| Répartition des Tailles | Heatmap de Position |
| :---: | :---: |
| ![Size Dist](runs/analysis/object_sizes_dist.png) | ![Heatmap](runs/analysis/object_heatmap.png) |

> **Le constat (Gauche) :** **85.3%** des objets font moins de 32 pixels de large (la ligne rouge). C'est extrêmement petit, souvent invisible pour une caméra de surveillance classique.
>
> **Le constat (Droite) :** L'action se concentre au centre de l'image (zone jaune), avec peu d'activité sur les bords extrêmes.
>
> **Impact Métier :** L'architecture a été configurée pour travailler en **haute résolution (1024px+)**. Utiliser une résolution standard aurait rendu 85% du trafic invisible au système.

### 3. Géométrie des Objets
![Box Sizes](runs/analysis/box_sizes.png)
> **Le constat :** Le nuage de points rouge montre la forme des objets. On voit une forte concentration en bas à gauche.
>
> **Impact Métier :** Cela confirme la nécessité d'algorithmes spécialisés pour les objets denses et non carrés.

---

## 🏗️ Architecture du Code
La structure suit les principes du *Clean Code* et de la séparation des responsabilités pour garantir la reproductibilité industrielle.

```text
.
├── data/                 # Données brutes et converties
├── runs/                 # Artefacts (Logs, Poids, Graphiques d'analyse)
├── src_training/         # Code source du moteur d'entraînement
│   ├── config/           # Hyperparamètres et validation stricte
│   ├── data/             # Processeurs et validateurs de données
│   ├── engine/           # Moteur d'entraînement et Callbacks
│   └── utils/            # Logging et reproductibilité (Seeding)
├── tools/                # Scripts de préparation "Offline" (ETL)
└── main_training.py      # Point d'entrée unique
```

---

## 🚀 Démarrage Rapide

### 1. Installation
Environnement Python 3.11+ recommandé. Installer les dépendances :

```bash
pip install -r requirements.txt
```

### 2. Préparation des données
Conversion du format VisDrone brut vers le standard YOLO et génération des rapports d'analyse (images ci-dessus).

```bash
# Conversion & Audit
python tools/convert_visdrone_to_yolo.py --data_root ./data/VisDrone --output_dir ./data/VisDrone_YOLO

# Analyse Exploratoire (Génère les graphs)
python tools/analyze_data.py --data_root ./data/VisDrone_YOLO
```

### 3. Entraînement du modèle
Exécution du pipeline sur GPU (détection automatique).

```bash
python main_training.py --data_root ./data/VisDrone_YOLO --epochs 100 --imgsz 1024
```

### 4. Résultats de performance 
Résumé des métriques clés obtenues lors de la validation finale :

| Métrique              | Score | Interprétation                                                                 |
|-----------------------|-------|--------------------------------------------------------------------------------|
| mAP@50 (Global)       | ~0.52 | Précision globale correcte, bonne détection de présence.                        |
| mAP (Voitures)        | 0.87  | 🟢 Excellente fiabilité pour le trafic automobile.                              |
| mAP (Piétons)         | 0.59  | 🟡 Performance moyenne, nécessite une haute résolution.                         |
| mAP (Vélos)           | 0.30  | 🔴 Point d'attention : confusion fréquente avec les motos.                     |

# Phase 2 : Pipeline de Production & Tracking

## 📋 Vue d'ensemble

C'est le **cœur applicatif** du projet.  
Alors que la Phase 1 se concentrait sur l'apprentissage (le *cerveau*), cette Phase 2 déploie l'intelligence sur le terrain pour résoudre le problème concret :

> **Compter et classifier les véhicules sur un flux vidéo CCTV réel**

L'architecture repose sur une chaîne de traitement séquentielle optimisée pour :
- éviter les **doubles comptages**
- garantir la **persistance des identifiants (ID)**, même lors d’occlusions temporaires

---

## 🎬 Démonstration des Résultats

Le système est capable de :
- suivre plusieurs objets simultanément
- maintenir leur identité (ID unique)
- détecter le franchissement d'une **ligne virtuelle bidirectionnelle**

<div align="center">
  <video src="visuel/results_comptage_montage.mp4"
         width="100%"
         controls
         autoplay
         loop
         muted>
  </video>
  <p><em>
    Sortie du pipeline : Visualisation des Bounding Boxes, des IDs uniques
    et du tableau de bord statistique en temps réel.
  </em></p>
</div>

---

## ⚙️ Mécanique du Pipeline (Under the Hood)

Le script `main_comptage.py` orchestre **quatre modules distincts** pour transformer des pixels en données statistiques exploitables.

---

### 1. Détection — YOLOv11

- Scanne chaque frame pour localiser les objets
- **Optimisation :** regroupement sémantique des classes  
  *(ex: `car`, `truck`, `bus` → `vehicle`)*  
  afin de simplifier le reporting

---

### 2. Tracking — BoT-SORT

- Assigne un **ID unique** à chaque objet
- Prédit la position future via un **Filtre de Kalman**

**Rôle clé :**  
Empêcher qu'une voiture soit comptée 50 fois simplement parce qu’elle apparaît sur 50 frames consécutives.

---

### 3. Géométrie — Détection de Franchissement

- Analyse vectorielle du mouvement
- Comparaison de la position `t-1` et `t0` par rapport à la ligne virtuelle
- Utilisation du **produit vectoriel** pour déterminer le sens de passage

---

### 4. Visualisation

- Rendu graphique **découplé du calcul**
- Architecture prête pour une exécution **headless** (serveur / edge)

---

## 🏗️ Architecture du Code

L’organisation privilégie la **modularité** et la **maintenabilité**.  
Changer de moteur de détection (ex: TensorRT) ou d’algorithme de tracking n’impacte qu’un seul module dans `core/`.

```text
src_comptage/
├── config/           # Configuration centralisée (Ligne, Seuils, Chemins)
├── core/             # Logique Métier Pure
│   ├── detector.py   # Wrapper d'inférence (abstraction du modèle)
│   ├── tracker.py    # Gestion de l'association temporelle (IDs)
│   └── counter.py    # Machine à états (logique d'entrée/sortie)
├── utils/            # Outils Techniques
│   ├── geometry.py   # Mathématiques vectorielles (intersections, directions)
│   ├── video_io.py   # Lecture vidéo robuste & threadée
│   └── visualizer.py # Moteur de rendu graphique (overlays)
└── main_comptage.py  # Orchestrateur
```

## 🚀 Utilisation

### Lancement standard

Utilise la configuration par défaut définie dans `settings.py`.

```bash
python main_comptage.py
```

### Surcharge des paramètres
Spécifier une source vidéo ou un modèle différent :

```bash
python main_comptage.py \
  --source ./data/video_test/Road_traffic_cctv.mp4 \
  --model ./weights/best_yolo11s_71.pt
```

## 💡 Analyse Critique & Améliorations Futures

Analyse des limites actuelles et **roadmap technique** pour un déploiement industriel.

---

### 1. Le défi du *Domain Shift* (Drone vs CCTV)

**Constat :**
- Entraînement sur **VisDrone** (vue aérienne verticale)
- Déploiement sur **CCTV** (vue angulaire rasante)

**Impact :**
- Généralisation correcte
- Baisse de précision pour :
  - véhicules vus de face
  - objets très éloignés (effets de perspective)

**Solution idéale :**
- Fine-tuning ou ré-entraînement sur **UA-DETRAC**
- Dataset spécifiquement conçu pour la surveillance routière CCTV

> *Note : VisDrone a été conservé ici pour des raisons de rapidité d’itération et de disponibilité des poids.*

---

### 2. Optimisation de l’Inférence (Edge Computing)

**Actuel :**
- Inférence PyTorch standard
- ~30–60 FPS sur GPU serveur

**Futur :**
- Export **TensorRT** (NVIDIA) ou **ONNX Runtime**

**Bénéfices :**
- FPS doublés
- Déploiement sur matériel embarqué léger  
  *(NVIDIA Jetson Orin, edge roadside)*

---

### 3. Estimation de Vitesse

**Extension prévue :**
- Calibration de caméra via **homographie**
- Projection des pixels 2D vers un plan réel 3D

**Résultat :**
- Calcul de la vitesse des véhicules en **km/h**

**Métrique clé pour :**
- gestion du trafic
- détection d’anomalies
- applications smart-city

---

## ✅ Conclusion

Cette Phase 2 transforme un modèle de vision par ordinateur en un **système opérationnel de comptage intelligent**, prêt à évoluer vers une solution industrielle **scalable**, **performante** et **déployable en edge computing**.

<div align="center">
    <sub>Nathan Edery</sub>
</div>
