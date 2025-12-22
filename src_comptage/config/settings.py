"""
Configuration centrale du pipeline d'inférence (Détection + Tracking + Comptage).
Définit les chemins, les hyperparamètres du modèle et les règles métier (groupement de classes).
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import torch


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent

@dataclass(frozen=True)
class InferenceConfig:
    # --- Chemins ---
    PROJECT_ROOT: Path = field(default_factory=get_project_root)
    
    # Poids du modèle fine-tuné
    MODEL_PATH: Path = field(default_factory=lambda: get_project_root() / "weights" / "best_yolo11s_71.pt")
    
    # Vidéo d'entrée (fichier local ou URL/Index stream)
    VIDEO_SOURCE: str | int = field(default_factory=lambda: str(get_project_root() / "data" / "video_test" / "Road_traffic_cctv.mp4"))
    
    # Sortie des artefacts (vidéos, CSVs)
    OUTPUT_DIR: Path = field(default_factory=lambda: get_project_root() / "runs" / "detect" / "count_inference")

    # --- Modèle & Hardware ---
    # Taille d'inférence (multiple de 32)
    IMG_SIZE: int = 1024
    
    CONF_THRESHOLD: float = 0.5
    IOU_THRESHOLD: float = 0.45
    
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Tracking ---
    # Algorithme de tracking (BoT-SORT recommandé pour les mouvements caméra, ByteTrack pour fixe)
    TRACKER_TYPE: str = "botsort.yaml"
    
    # Mémoire du tracker (frames) pour compenser les occlusions momentanées
    TRACK_BUFFER: int = 30

    # --- Logique Métier (Classes) ---
    # Mapping ID YOLO -> Label business regroupé
    # Ignorés : 0 (pedestrian), 1 (people)
    # Groupes : 
    #   - vehicle : car(3), van(4), truck(5), bus(8)
    #   - 2-wheels : bicycle(2), tricycle(6), awning-tricycle(7), motor(9)
    TARGET_CLASSES: Dict[int, str] = field(default_factory=lambda: {
        2: "2-wheels",
        3: "vehicle",
        4: "vehicle",
        5: "vehicle",
        6: "2-wheels",
        7: "2-wheels",
        8: "vehicle",
        9: "2-wheels"
    })
    
    # Liste explicite des IDs à inférer (filtre les piétons dès le début)
    CLASSES_TO_COUNT: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7, 8, 9])

    # --- Géométrie (Comptage) ---
    # Segment de ligne virtuelle [Point A, Point B] en pixels
    # On met la ligne au milieu (y=180) et sur toute la largeur (0 à 640)
    LINE_COORDINATES: Tuple[Tuple[int, int], Tuple[int, int]] = (
        (0, 180),    # Point A : Tout à gauche, milieu hauteur
        (640, 180)   # Point B : Tout à droite, milieu hauteur
    )

    # --- Rendu ---
    LINE_THICKNESS: int = 2
    TEXT_SCALE: float = 1.0
    LINE_COLOR: Tuple[int, int, int] = (0, 255, 0)
    TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)

    def __post_init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        if not self.MODEL_PATH.exists():
            logging.warning(f"⚠️ Modèle introuvable : {self.MODEL_PATH}")

SETTINGS = InferenceConfig()

if __name__ == "__main__":
    print(f"✅ Config chargée. Mapping : {SETTINGS.TARGET_CLASSES}")
    print(f"🎯 IDs filtrés : {SETTINGS.CLASSES_TO_COUNT}")