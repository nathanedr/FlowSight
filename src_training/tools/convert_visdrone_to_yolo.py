"""
Script de conversion du dataset VisDrone-DET vers le format YOLO (Ultralytics standard).

Usage:
    python src/tools/convert_visdrone_to_yolo.py --data_root ./data/VisDrone-dataset --output_dir ./data/VisDrone_YOLO
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
from tqdm import tqdm
from PIL import Image

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Mapping des classes VisDrone vers YOLO (0-indexé)
VISDRONE_TO_YOLO_MAP: Dict[int, int] = {
    1: 0,   # pedestrian
    2: 1,   # people
    3: 2,   # bicycle
    4: 3,   # car
    5: 4,   # van
    6: 5,   # truck
    7: 6,   # tricycle
    8: 7,   # awning-tricycle
    9: 8,   # bus
    10: 9   # motor
}

IGNORED_CLASSES = {0, 11}

YOLO_CLASSES_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VisDrone-DET dataset to YOLO format.")
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Chemin racine contenant les dossiers VisDrone extraits (ex: VisDrone2019-DET-train)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Dossier de destination pour le dataset converti"
    )
    parser.add_argument(
        "--copy_images",
        action="store_true",
        default=True, 
        help="Copier les images (True) ou créer des symlinks (False, non implémenté ici)"
    )
    return parser.parse_args()

def resolve_visdrone_dirs(input_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Résout les chemins vers images/ et annotations/ en gérant:
      - input_dir/images
      - input_dir/annotations
    ET le cas imbriqué:
      - input_dir/<input_dir.name>/images
      - input_dir/<input_dir.name>/annotations
    """
    candidates = [
        input_dir,
        input_dir / input_dir.name,  # ex: VisDrone2019-DET-train/VisDrone2019-DET-train/
    ]

    img_dir = None
    ann_dir = None

    for base in candidates:
        d_img = base / "images"
        d_ann = base / "annotations"
        if img_dir is None and d_img.exists():
            img_dir = d_img
        if ann_dir is None and d_ann.exists():
            ann_dir = d_ann

    return img_dir, ann_dir

def convert_box(img_size: Tuple[int, int], box: List[float]) -> Optional[List[float]]:
    """
    Convertit une bbox VisDrone (top_left_x, top_left_y, width, height)
    en format YOLO normalisé (center_x, center_y, width, height).
    """
    dw = 1.0 / img_size[0]
    dh = 1.0 / img_size[1]

    x, y, w, h = box

    x_center = x + w / 2.0
    y_center = y + h / 2.0

    x_center *= dw
    w *= dw
    y_center *= dh
    h *= dh

    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    if w < 0.001 or h < 0.001:
        return None

    return [x_center, y_center, w, h]

def process_split(
    split_name: str,
    input_dir: Path,
    output_base: Path,
    is_test: bool = False
) -> int:
    """
    Traite un sous-ensemble (train, val ou test).
    Retourne le nombre d'images traitées.
    """
    img_dir_in, ann_dir_in = resolve_visdrone_dirs(input_dir)

    if img_dir_in is None or not img_dir_in.exists():
        logger.warning(f"Dossier images introuvable pour {split_name} (input_dir={input_dir}). Skip.")
        return 0

    # Pour test-dev, annotations peuvent être absentes
    if (ann_dir_in is None or not ann_dir_in.exists()) and (not is_test):
        logger.warning(f"Dossier annotations introuvable pour {split_name}: {ann_dir_in}. Les labels seront vides.")

    img_dir_out = output_base / "images" / split_name
    label_dir_out = output_base / "labels" / split_name

    img_dir_out.mkdir(parents=True, exist_ok=True)
    label_dir_out.mkdir(parents=True, exist_ok=True)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in img_dir_in.iterdir() if p.is_file() and p.suffix.lower() in valid_extensions]

    logger.info(f"Traitement du split '{split_name}': {len(images)} images détectées ({img_dir_in}).")

    processed_count = 0

    for img_path in tqdm(images, desc=f"Conversion {split_name}"):
        try:
            with Image.open(img_path) as im:
                width, height = im.size

            target_img_path = img_dir_out / img_path.name
            if not target_img_path.exists():
                shutil.copy2(img_path, target_img_path)

            yolo_labels: List[str] = []

            if not is_test and ann_dir_in is not None and ann_dir_in.exists():
                ann_path = ann_dir_in / f"{img_path.stem}.txt"

                if ann_path.exists():
                    with open(ann_path, "r") as f:
                        for raw in f:
                            line = raw.strip()
                            if not line:
                                continue

                            parts = line.split(",")
                            if len(parts) < 8:
                                continue

                            try:
                                bbox_left = float(parts[0])
                                bbox_top = float(parts[1])
                                bbox_w = float(parts[2])
                                bbox_h = float(parts[3])
                                score = float(parts[4])
                                category = int(parts[5])

                                if category in IGNORED_CLASSES:
                                    continue
                                if category not in VISDRONE_TO_YOLO_MAP:
                                    continue
                                if score == 0:
                                    continue

                                yolo_cls = VISDRONE_TO_YOLO_MAP[category]
                                yolo_box = convert_box((width, height), [bbox_left, bbox_top, bbox_w, bbox_h])

                                if yolo_box:
                                    yolo_labels.append(
                                        f"{yolo_cls} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}"
                                    )
                            except ValueError:
                                continue

            if not is_test:
                target_label_path = label_dir_out / f"{img_path.stem}.txt"
                with open(target_label_path, "w") as f:
                    f.write("\n".join(yolo_labels))

            processed_count += 1

        except Exception as e:
            logger.error(f"Erreur sur l'image {img_path.name}: {e}")
            continue

    return processed_count

def create_yaml_config(output_dir: Path):
    yaml_content = {
        "path": str(output_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(YOLO_CLASSES_NAMES)},
    }

    yaml_path = output_dir / "VisDrone.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    logger.info(f"Fichier de configuration généré : {yaml_path}")

def main():
    args = parse_args()

    if not args.data_root.exists():
        logger.error(f"Le dossier racine {args.data_root} n'existe pas.")
        sys.exit(1)

    if args.output_dir.exists():
        logger.warning(f"Le dossier de sortie {args.output_dir} existe déjà. Les fichiers peuvent être écrasés.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits_map = {
        "train": "VisDrone2019-DET-train",
        "val": "VisDrone2019-DET-val",
        "test": "VisDrone2019-DET-test-dev",
    }

    logger.info("Démarrage de la conversion VisDrone -> YOLO...")

    total_imgs = 0
    for split_yolo, split_visdrone in splits_map.items():
        input_split_path = args.data_root / split_visdrone
        if not input_split_path.exists():
            logger.warning(f"Dossier source pour '{split_yolo}' non trouvé: {input_split_path}. Fallback simple.")
            input_split_path = args.data_root / split_visdrone.replace("VisDrone2019-DET-", "")
            if not input_split_path.exists():
                logger.error(f"Impossible de trouver les données pour {split_yolo}. Vérifiez la structure.")
                continue

        is_test_set = (split_yolo == "test")
        count = process_split(split_yolo, input_split_path, args.output_dir, is_test=is_test_set)
        total_imgs += count

    create_yaml_config(args.output_dir)

    logger.info("---")
    logger.info("Conversion terminée avec succès.")
    logger.info(f"Total images traitées : {total_imgs}")
    logger.info(f"Dataset prêt dans : {args.output_dir}")
    logger.info(f"Pour lancer l'entraînement : yolo train data={args.output_dir}/VisDrone.yaml ...")

if __name__ == "__main__":
    main()
