"""
Module de visualisation pour le pipeline de comptage.
Responsable unique du dessin des annotations (BBoxes, IDs, Lignes, Statistiques) sur les frames.
Optimisé pour OpenCV.
"""

import cv2
import numpy as np
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Gestion des imports
try:
    from src_comptage.config.settings import SETTINGS
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src_comptage.config.settings import SETTINGS

logger = logging.getLogger(__name__)

class Visualizer:
    """
    Moteur de rendu pour les résultats d'inférence.
    Sépare la logique d'affichage du traitement pour permettre un mode headless.
    """

    def __init__(self):
        """Initialise les ressources graphiques (polices, couleurs, géométrie)."""
        self.line_coords = SETTINGS.LINE_COORDINATES
        self.line_color = SETTINGS.LINE_COLOR
        self.line_thickness = SETTINGS.LINE_THICKNESS
        
        self.text_color = SETTINGS.TEXT_COLOR
        self.text_scale = SETTINGS.TEXT_SCALE
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        self.colors = {} 

    def _get_color(self, tag: int) -> Tuple[int, int, int]:
        """Génère une couleur unique et stable pour un ID ou une classe."""
        if tag not in self.colors:
            np.random.seed(tag)
            self.colors[tag] = tuple(map(int, np.random.randint(0, 255, 3)))
        return self.colors[tag]

    def draw_tracks(self, frame: np.ndarray, tracks: np.ndarray) -> np.ndarray:
        """
        Dessine les bounding boxes et les IDs des véhicules suivis.
        
        Args:
            frame: Image BGR.
            tracks: Array (N, 7) -> [x1, y1, x2, y2, id, conf, cls]
        """
        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            track_id = int(track[4])
            cls_id = int(track[6])
            
            # Récupération du nom de groupe (vehicle, 2-wheels...) ou fallback
            label_text = SETTINGS.TARGET_CLASSES.get(cls_id, str(cls_id))
            
            color = self._get_color(track_id)
            
            # Dessin BBox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Dessin Label (ID + Class)
            caption = f"#{track_id} {label_text}"
            (w, h), _ = cv2.getTextSize(caption, self.font, 0.6, 1)
            
            # Fond du texte pour lisibilité
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(frame, caption, (x1, y1 - 5), self.font, 0.6, (255, 255, 255), 1)
            
            # Point central
            center_x, center_y = int((x1 + x2) / 2), int(y2)
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

        return frame

    def draw_line(self, frame: np.ndarray) -> np.ndarray:
        """Dessine la ligne de comptage virtuelle."""
        pt1, pt2 = self.line_coords
        cv2.line(frame, pt1, pt2, self.line_color, self.line_thickness)
        
        # Marqueurs aux extrémités pour visibilité
        cv2.circle(frame, pt1, 5, self.line_color, -1)
        cv2.circle(frame, pt2, 5, self.line_color, -1)
        
        # Label "Counting Line"
        mid_x = (pt1[0] + pt2[0]) // 2
        mid_y = (pt1[1] + pt2[1]) // 2
        cv2.putText(frame, "Ligne de comptage", (mid_x - 50, mid_y - 10), 
                    self.font, 0.6, self.line_color, 2)
        return frame

    def draw_counts_panel(self, frame: np.ndarray, counts: Dict[str, Dict[str, int]]) -> np.ndarray:
        """
        Affiche un tableau de bord semi-transparent avec les statistiques.
        Structure attendue de counts: {'vehicle': {'in': 10, 'out': 5, 'total': 15}, ...}
        """
        # Dimensions du panneau
        h, w = frame.shape[:2]
        panel_w = 350
        panel_h = 50 + (len(counts) * 40) if counts else 80
        
        # Création overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (20 + panel_w, 20 + panel_h), (0, 0, 0), -1)
        
        # Application transparence (alpha blending)
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Titre
        cv2.putText(frame, "STATISTIQUES TRAFIC", (35, 50), self.font, 0.7, (0, 255, 255), 2)
        
        # Lignes de stats
        y_offset = 90
        if not counts:
            cv2.putText(frame, "En attente...", (35, y_offset), self.font, 0.6, (200, 200, 200), 1)
        
        for category, vals in counts.items():
            # Format: CATEGORY: Total (In / Out)
            text = f"{category.upper()}: {vals['total']} (In: {vals['in']} | Out: {vals['out']})"
            cv2.putText(frame, text, (35, y_offset), self.font, 0.6, (255, 255, 255), 1)
            y_offset += 35
            
        return frame

    def render(self, frame: np.ndarray, tracks: np.ndarray, counts: Dict) -> np.ndarray:
        """Méthode principale : applique toutes les couches de visualisation."""
        frame = self.draw_tracks(frame, tracks)
        frame = self.draw_line(frame)
        frame = self.draw_counts_panel(frame, counts)
        return frame


def main():
    """Test unitaire visuel (génère une image statique)."""
    logging.basicConfig(level=logging.INFO)
    
    # Image noire HD
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Mock Data
    # Tracks: [x1, y1, x2, y2, id, conf, cls]
    mock_tracks = np.array([
        [400, 500, 600, 700, 1, 0.95, 4],   # Van
        [800, 550, 950, 650, 2, 0.88, 3],   # Car
        [1200, 600, 1300, 800, 42, 0.75, 9] # Motor
    ])
    
    mock_counts = {
        "vehicle": {"total": 124, "in": 60, "out": 64},
        "2-wheels": {"total": 15, "in": 5, "out": 10}
    }

    viz = Visualizer()
    
    logger.info("🎨 Rendu d'une frame de test...")
    res_frame = viz.render(frame, mock_tracks, mock_counts)
    
    output_path = Path("test_visu.jpg")
    cv2.imwrite(str(output_path), res_frame)
    logger.info(f"✅ Image sauvegardée : {output_path.absolute()}")
    
    # Si affichage possible (local)
    # cv2.imshow("Test Visualizer", cv2.resize(res_frame, (960, 540)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()