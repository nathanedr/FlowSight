"""
Module de gestion du Tracking Multi-Objets (MOT).
Encapsule la logique de tracking d'Ultralytics (BoT-SORT/ByteTrack) pour garantir la persistance des IDs.
Essentiel pour le comptage temporel (évite de compter plusieurs fois le même véhicule).
"""

import sys
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import torch

# Gestion des imports pour exécution standalone ou module
try:
    from src_comptage.config.settings import SETTINGS
except ImportError:
    # Fallback pour test local
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src_comptage.config.settings import SETTINGS

from ultralytics import YOLO

logger = logging.getLogger(__name__)

class ObjectTracker:
    """
    Wrapper pour le tracking d'objets.
    Utilise 'model.track()' d'Ultralytics qui combine détection et association (kalman filter + hungarian algo).
    """

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialise le tracker et le modèle sous-jacent.
        """
        self.path = model_path or SETTINGS.MODEL_PATH
        self.device = SETTINGS.DEVICE
        self.conf_thres = SETTINGS.CONF_THRESHOLD
        self.iou_thres = SETTINGS.IOU_THRESHOLD
        self.classes = SETTINGS.CLASSES_TO_COUNT
        self.img_size = SETTINGS.IMG_SIZE
        
        # Paramètres spécifiques au tracker
        self.tracker_config = SETTINGS.TRACKER_TYPE
        self.persist = True # Indispensable pour maintenir les IDs entre les frames
        
        self.model = self._load_model()

    def _load_model(self) -> YOLO:
        """Charge le modèle YOLO."""
        if not self.path.exists():
            msg = f"❌ Modèle introuvable : {self.path}"
            logger.critical(msg)
            raise FileNotFoundError(msg)

        try:
            logger.info(f"⚖️ Chargement du modèle pour tracking : {self.path} ({self.device})")
            return YOLO(str(self.path), task="detect")
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle : {e}")
            raise e

    def update(self, frame: np.ndarray) -> np.ndarray:
        """
        Met à jour le tracker avec une nouvelle frame.
        
        Args:
            frame (np.ndarray): Image actuelle (BGR).

        Returns:
            np.ndarray: Tableau des tracks actifs de forme (N, 7).
                        Format: [x1, y1, x2, y2, track_id, conf, class_id]
                        Retourne un tableau vide si aucun track.
        """
        # Appel à ultralytics track()
        # persist=True est critique pour garder l'historique
        results = self.model.track(
            source=frame,
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=self.img_size,
            classes=self.classes,
            device=self.device,
            tracker=self.tracker_config,
            persist=self.persist,
            verbose=False
        )

        if not results:
            return np.empty((0, 7))

        result = results[0]
        
        # Si aucun objet détecté ou si le tracker n'a pas encore assigné d'ID
        if result.boxes is None or result.boxes.id is None:
            return np.empty((0, 7))

        # Récupération des données: .boxes.data contient souvent [x1, y1, x2, y2, id, conf, cls] pour track()
        # Note: Ultralytics retourne parfois xyxy, id, conf, cls séparés. On sécurise via .boxes.data
        # Le format standard de result.boxes.data pour track est (N, 7) : x1, y1, x2, y2, id, conf, cls
        tracks = result.boxes.data.cpu().numpy()
        
        return tracks

    @property
    def names(self) -> Dict[int, str]:
        """Mapping des noms de classes."""
        return self.model.names


def main():
    """Validation unitaire du Tracker."""
    logging.basicConfig(level=logging.INFO)
    import cv2
    
    video_source = SETTINGS.VIDEO_SOURCE
    
    if isinstance(video_source, str) and not video_source.isdigit() and not Path(video_source).exists():
        logger.warning(f"⚠️ Vidéo source non trouvée ({video_source}). Test sur image noire uniquement.")
        cap = None
    else:
        logger.info(f"🎞 Test sur flux vidéo : {video_source}")
        cap = cv2.VideoCapture(video_source)

    tracker = ObjectTracker()

    if cap and cap.isOpened():
        # Test sur 30 frames pour voir si les IDs restent stables
        logger.info("🏃 Lancement de la boucle de test (30 frames)...")
        for i in range(30):
            ret, frame = cap.read()
            if not ret:
                break
            
            tracks = tracker.update(frame)
            
            # Log simple des IDs vus
            if len(tracks) > 0:
                unique_ids = np.unique(tracks[:, 4]).astype(int).tolist()
                logger.info(f"Frame {i}: {len(tracks)} véhicules tracked. IDs: {unique_ids}")
            else:
                logger.info(f"Frame {i}: Aucun track.")
        
        cap.release()
    else:
        # Test synthétique
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # On dessine un carré qui bouge
        cv2.rectangle(dummy_frame, (100, 100), (200, 200), (255, 255, 255), -1)
        
        logger.info("📸 Test synthétique (Frame 1)...")
        tracks = tracker.update(dummy_frame)
        logger.info(f"Tracks trouvés : {len(tracks)}")
    
    logger.info("✅ Test Tracker terminé.")

if __name__ == "__main__":
    main()