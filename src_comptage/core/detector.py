"""
Module d'abstraction pour la détection d'objets.
Isole la logique spécifique à la librairie (Ultralytics YOLO) du reste de l'application.
Permet de changer de backend (TensorRT, ONNX) sans impacter le tracking/comptage.
"""

import sys
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
import torch

# Gestion des imports pour exécution standalone ou module
try:
    from src_comptage.config.settings import SETTINGS
except ImportError:
    # Fallback pour test local si lancé directement depuis core/
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src_comptage.config.settings import SETTINGS

from ultralytics import YOLO

logger = logging.getLogger(__name__)

class ObjectDetector:
    """
    Wrapper autour du modèle de détection (YOLOv11).
    Responsabilité : Charger le modèle et transformer une image en liste de Bounding Boxes.
    """

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialise le détecteur.
        
        Args:
            model_path: Chemin vers le poids .pt (utilise SETTINGS par défaut).
        """
        self.path = model_path or SETTINGS.MODEL_PATH
        self.device = SETTINGS.DEVICE
        self.conf_thres = SETTINGS.CONF_THRESHOLD
        self.iou_thres = SETTINGS.IOU_THRESHOLD
        self.classes = SETTINGS.CLASSES_TO_COUNT
        self.img_size = SETTINGS.IMG_SIZE
        
        self.model = self._load_model()

    def _load_model(self) -> YOLO:
        """Charge le modèle en mémoire et le bascule sur le device approprié."""
        if not self.path.exists():
            msg = f"❌ Fichier modèle introuvable : {self.path}"
            logger.critical(msg)
            raise FileNotFoundError(msg)

        try:
            logger.info(f"⚖️ Chargement du modèle : {self.path} sur {self.device}...")
            model = YOLO(str(self.path), task="detect")
            
            # Vérification basique (fuse, warm up si nécessaire, mais YOLO le fait souvent lazy)
            return model
        except Exception as e:
            logger.error(f"❌ Erreur critique au chargement du modèle : {e}")
            raise e

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Exécute l'inférence sur une frame unique.

        Args:
            frame (np.ndarray): Image BGR (H, W, 3) issue d'OpenCV.

        Returns:
            np.ndarray: Tableau de détections de forme (N, 6).
                        Chaque ligne : [x1, y1, x2, y2, confidence, class_id]
        """
        # Inférence Ultralytics
        # verbose=False pour éviter de spammer la console à chaque frame
        results = self.model.predict(
            source=frame,
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=self.img_size,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

        if not results:
            return np.empty((0, 6))

        # Récupération du premier résultat (car batch=1)
        result = results[0]
        
        # Copie vers CPU numpy
        # boxes.data contient déjà [x1, y1, x2, y2, conf, cls]
        detections = result.boxes.data.cpu().numpy()
        
        return detections

    @property
    def names(self) -> dict:
        """Retourne le mapping ID -> Nom de classe du modèle."""
        return self.model.names


def main():
    """Test unitaire du détecteur."""
    logging.basicConfig(level=logging.INFO)
    
    # Création d'une image factice (bruit noir)
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Dessin d'un rectangle blanc pour simuler "quelque chose" (peu de chance d'être détecté mais teste le pipeline)
    import cv2
    cv2.rectangle(dummy_frame, (500, 500), (700, 700), (255, 255, 255), -1)

    try:
        logger.info("Initialisation du détecteur...")
        detector = ObjectDetector()
        
        logger.info(f"📸 Test d'inférence sur image {dummy_frame.shape}...")
        detections = detector.detect(dummy_frame)
        
        logger.info(f"✅ Inférence terminée. {len(detections)} objets détectés.")
        logger.info(f"📊 Format de sortie (sample) : \n{detections[:2] if len(detections) > 0 else 'Aucune détection (normal sur image noire)'}")

        # teste sur la première frame réelle
        video_path = Path(SETTINGS.VIDEO_SOURCE)
        if video_path.exists() and video_path.is_file():
            logger.info(f"🎞 Test sur fichier réel : {video_path}")
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            if ret:
                dets = detector.detect(frame)
                logger.info(f"✅ Inférence réelle : {len(dets)} objets trouvés sur la frame 0.")
                # Affichage des classes détectées
                found_classes = [SETTINGS.TARGET_CLASSES.get(int(c), str(c)) for c in dets[:, 5]]
                logger.info(f"🔍 Classes vues : {set(found_classes)}")
            cap.release()

    except Exception as e:
        logger.error(f"❌ Test échoué : {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()