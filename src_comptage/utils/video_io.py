"""
Gestionnaire d'entrée vidéo robuste.
Encapsule OpenCV VideoCapture pour fournir un itérateur sûr avec gestion de contexte.
"""

import cv2
import logging
import time
from pathlib import Path
from typing import Generator, Tuple, Optional, Union
import numpy as np

# Configuration logger module
logger = logging.getLogger(__name__)

class VideoStream:
    """
    Classe wrapper autour de cv2.VideoCapture pour une lecture vidéo sécurisée.
    Supporte l'utilisation via context manager (`with VideoStream(...) as vs:`).
    """

    def __init__(self, source: Union[str, int, Path]):
        """
        Initialise le flux vidéo.
        
        Args:
            source: Chemin fichier (str/Path) ou index caméra (int).
        """
        # Conversion Path -> str pour OpenCV si nécessaire
        self.source = str(source) if isinstance(source, Path) else source
        self.cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        
        self._open_stream()

    def _open_stream(self) -> None:
        """Ouvre ou ré-ouvre le flux vidéo et valide l'accès."""
        logger.info(f"🎥 Ouverture du flux vidéo : {self.source}")
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            msg = f"❌ Impossible d'ouvrir la source vidéo : {self.source}"
            logger.error(msg)
            raise IOError(msg)

        # Récupération des métadonnées
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"✅ Flux ouvert : {self.width}x{self.height} @ {self.fps:.2f}fps ({self.total_frames} frames)")

    def __iter__(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Générateur principal de frames.
        
        Yields:
            Tuple[int, np.ndarray]: (index_frame, image_bgr)
        """
        if self.cap is None or not self.cap.isOpened():
            logger.error("Tentative de lecture sur un flux fermé.")
            return

        self._frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                # Fin de fichier ou erreur de flux
                logger.info("Flux vidéo terminé ou interrompu.")
                break
            
            yield self._frame_count, frame
            self._frame_count += 1

    def __enter__(self):
        """Support du Context Manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Nettoyage automatique des ressources."""
        self.release()

    def release(self) -> None:
        """Libère proprement la ressource OpenCV."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info("🛑 Flux vidéo libéré.")
        self.cap = None

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)


def main():
    """
    Test unitaire manuel : lit une vidéo et affiche les infos.
    N'affiche pas la GUI si exécuté sur un serveur headless (sauf si DISPLAY set).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    
    # Simuler un chemin depuis la config (exemple)
    try:
        # On remonte de utils/ à la racine pour trouver data/
        root = Path(__file__).resolve().parent.parent.parent
        test_video = root / "data" / "video_test" / "Road_traffic_cctv.mp4"
        
        if not test_video.exists():
            logger.warning(f"Fichier de test non trouvé : {test_video}")
            logger.info("Essai avec la webcam (0)...")
            test_video = 0

        # Test du Context Manager
        with VideoStream(test_video) as stream:
            logger.info(f"Lecture en cours... Resolution: {stream.resolution}")
            
            start_time = time.time()
            for idx, frame in stream:
                # Simulation de traitement (juste pour tester la boucle)
                if idx % 30 == 0:
                    logger.info(f"Frame {idx}/{stream.total_frames} traitée.")
                
                # Optionnel : Affichage (commenté pour compatibilité serveur)
                # cv2.imshow("Test Stream", cv2.resize(frame, (640, 360)))
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
            
            duration = time.time() - start_time
            logger.info(f"Terminé. {stream._frame_count} frames lues en {duration:.2f}s ({stream._frame_count/duration:.1f} fps process).")

    except Exception as e:
        logger.error(f"Erreur lors du test : {e}")
        raise e

if __name__ == "__main__":
    main()