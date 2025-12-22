"""
Gestion des augmentations d'images avancées pour VisDrone.
Intègre Albumentations pour des transformations spécifiques à la vue aérienne (CLAHE, Blur, etc.)
qui complètent les augmentations géométriques natives de YOLO (Mosaic, MixUp).

Peut être utilisé comme module importé par le Trainer ou exécuté seul pour visualiser/debugger
l'impact des augmentations sur une image donnée.

Usage (Debug):
    python src_training/data/processor.py --image path/to/image.jpg --output_dir runs/debug_aug
"""

import argparse
import logging
import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

try:
    import albumentations as A
    from albumentations.core.composition import Compose
except ImportError:
    A = None
    Compose = None

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [DATA-PROC] - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Configuration par défaut des augmentations pour VisDrone
# Note: YOLO gère déjà Mosaic, MixUp, HSV. Ici on vise la texture et la robustesse capteur.
DEFAULT_AUG_CONFIG = {
    "p_seq": 1.0,           # Probabilité globale d'appliquer la séquence
    "clahe_p": 0.5,         # Contrast Limited Adaptive Histogram Equalization (Anti-haze/shadow)
    "blur_p": 0.2,          # Flou de mouvement (vitesse drone)
    "median_blur_p": 0.1,   # Bruit capteur
    "to_gray_p": 0.05,      # Robustesse couleur
    "dropout_p": 0.1,       # Occlusions partielles (CoarseDropout)
}

class VisDroneAugmenter:
    """
    Wrapper pour le pipeline Albumentations optimisé pour la détection aérienne.
    """
    def __init__(self, config: Optional[Dict[str, float]] = None):
        if A is None:
            logger.error("Albumentations n'est pas installé. `pip install albumentations` requis.")
            sys.exit(1)
        
        self.cfg = config or DEFAULT_AUG_CONFIG
        self.transform = self._build_pipeline()
        logger.info(f"Pipeline Albumentations initialisé avec : {self.cfg}")

    def _build_pipeline(self) -> Compose:
        """Construit le graphe de transformation."""
        # Note: BBoxParams format 'yolo' = [x_center, y_center, width, height] normalized
        return A.Compose([
            # 1. Amélioration de contraste (Vital pour ombres/nuages en vue aérienne)
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=self.cfg['clahe_p']),
            
            # 2. Simulation défauts capteur / mouvement
            A.OneOf([
                A.MotionBlur(p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.Blur(blur_limit=3, p=1.0),
            ], p=self.cfg['blur_p']),

            # 3. Robustesse chromatique (rare mais utile)
            A.ToGray(p=self.cfg['to_gray_p']),
            
            # 4. Occlusions (simule objets cachés sous arbres/ponts)
            # max_holes=8, max_height=32, max_width=32 (sur image HD)
            A.CoarseDropout(
                max_holes=8, 
                max_height=32, 
                max_width=32, 
                min_holes=1, 
                min_height=8, 
                min_width=8, 
                fill_value=0, 
                p=self.cfg['dropout_p']
            ),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def __call__(self, image: np.ndarray, bboxes: List[List[float]], class_labels: List[int]) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        Applique les transformations sur une image et ses labels.
        
        Args:
            image: Image OpenCV (H, W, C) BGR ou RGB
            bboxes: Liste de [xc, yc, w, h] normalisés
            class_labels: Liste d'IDs de classes
            
        Returns:
            (augmented_image, augmented_bboxes, augmented_labels)
        """
        try:
            # Albumentations requiert RGB souvent, mais OpenCV est BGR. 
            # On assume l'entrée est cohérente avec la sortie désirée.
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            return transformed['image'], transformed['bboxes'], transformed['class_labels']
        except Exception as e:
            logger.warning(f"Échec augmentation Albumentations: {e}. Retour orignaux.")
            return image, bboxes, class_labels


def load_yolo_label(label_path: Path) -> Tuple[List[List[float]], List[int]]:
    """Lit un fichier label YOLO txt."""
    bboxes = []
    classes = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(parts[0])
                    # x, y, w, h
                    box = [float(x) for x in parts[1:]]
                    classes.append(cls)
                    bboxes.append(box)
    return bboxes, classes

def draw_yolo_boxes(image: np.ndarray, bboxes: List[List[float]], classes: List[int]) -> np.ndarray:
    """Dessine les boîtes normalisées YOLO sur l'image pour visualisation."""
    h, w, _ = image.shape
    vis_img = image.copy()
    
    for box, cls in zip(bboxes, classes):
        xc, yc, bw, bh = box
        # Denormalize
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        
        # Draw rectangle
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw label
        cv2.putText(vis_img, str(cls), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
    return vis_img

def main():
    """Mode DEBUG : Teste les augmentations sur une image spécifique."""
    parser = argparse.ArgumentParser(description="Test/Debug VisDrone Augmentations")
    parser.add_argument("--image", type=Path, required=True, help="Chemin vers une image .jpg")
    parser.add_argument("--label", type=Path, help="Chemin vers le label .txt (optionnel, déduit si absent)")
    parser.add_argument("--output_dir", type=Path, default=Path("runs/debug_aug"), help="Dossier de sortie")
    parser.add_argument("--count", type=int, default=5, help="Nombre de versions augmentées à générer")
    args = parser.parse_args()

    if not args.image.exists():
        logger.error(f"Image introuvable : {args.image}")
        sys.exit(1)

    # Déduction label si non fourni
    label_path = args.label
    if not label_path:
        # Tente ../labels/name.txt ou ./name.txt
        p1 = args.image.parent.parent / "labels" / args.image.parent.name / f"{args.image.stem}.txt"
        p2 = args.image.with_suffix(".txt")
        if p1.exists(): label_path = p1
        elif p2.exists(): label_path = p2
    
    # Chargement
    logger.info(f"Chargement image: {args.image}")
    img_bgr = cv2.imread(str(args.image))
    if img_bgr is None:
        logger.error("Erreur lecture image via OpenCV.")
        sys.exit(1)
    
    # Conversion BGR -> RGB pour Albumentations (recommandé)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    bboxes, classes = [], []
    if label_path and label_path.exists():
        logger.info(f"Chargement labels: {label_path}")
        bboxes, classes = load_yolo_label(label_path)
    else:
        logger.warning("Aucun label trouvé. Augmentation image seule.")

    # Init Augmenter
    augmenter = VisDroneAugmenter()
    
    # Création output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Génération samples
    logger.info(f"Génération de {args.count} exemples dans {args.output_dir}...")
    
    # 1. Sauvegarde Original avec box
    orig_vis = draw_yolo_boxes(img_rgb.copy(), bboxes, classes)
    cv2.imwrite(str(args.output_dir / "original_viz.jpg"), cv2.cvtColor(orig_vis, cv2.COLOR_RGB2BGR))
    
    # 2. Boucle augmentations
    for i in range(args.count):
        aug_img, aug_boxes, aug_cls = augmenter(img_rgb, bboxes, classes)
        
        # Visu
        vis = draw_yolo_boxes(aug_img.copy(), aug_boxes, aug_cls)
        
        # Save (RGB -> BGR)
        out_name = args.output_dir / f"aug_{i:03d}.jpg"
        cv2.imwrite(str(out_name), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
    logger.info("Terminé.")

if __name__ == "__main__":
    main()