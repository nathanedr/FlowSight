"""
Script d'analyse exploratoire (EDA) pour datasets au format YOLO (VisDrone converti).
Génère des statistiques sur la distribution des classes, la taille des objets et leur densité.
Crucial pour ajuster les hyperparamètres (anchors, imgsz, mixup).

Usage:
    python src/tools/analyze_data.py --data_root ./data/VisDrone_YOLO --output_dir ./runs/analysis
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import Counter
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Configuration Matplotlib
plt.style.use('ggplot')

# Mapping par défaut (sera surchargé si un data.yaml est trouvé, sinon VisDrone standard)
DEFAULT_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

@dataclass
class DatasetStats:
    total_images: int
    total_objects: int
    empty_images: int
    class_counts: Dict[int, int]
    # Stockage optimisé numpy pour les boxes [class_id, x, y, w, h]
    boxes_norm: np.ndarray 

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze YOLO formatted dataset.")
    parser.add_argument("--data_root", type=Path, required=True, help="Racine du dataset YOLO (contenant labels/train, etc.)")
    parser.add_argument("--output_dir", type=Path, default=Path("runs/analysis"), help="Dossier de sortie pour les graphiques et rapports")
    parser.add_argument("--split", type=str, default="train", help="Sous-dossier à analyser (train, val, test)")
    parser.add_argument("--ref_imgsz", type=int, default=640, help="Taille d'image de référence pour estimer les pixels (ex: 640 ou 1920)")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="Nombre de workers pour le chargement")
    return parser.parse_args()

def parse_label_file(file_path: Path) -> List[List[float]]:
    """Lit un fichier label YOLO et retourne une liste de boxes."""
    boxes = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # class x y w h
                    boxes.append([float(x) for x in parts[:5]])
    except Exception:
        pass # Fichier vide ou corrompu traité comme vide
    return boxes

def load_dataset(data_root: Path, split: str, workers: int) -> DatasetStats:
    """Charge les labels en parallèle et compile les stats brutes."""
    label_dir = data_root / "labels" / split
    if not label_dir.exists():
        logger.error(f"Dossier labels introuvable : {label_dir}")
        sys.exit(1)

    label_files = list(label_dir.glob("*.txt"))
    logger.info(f"Analyse de {len(label_files)} fichiers dans {label_dir}...")

    all_boxes = []
    empty_imgs = 0

    # Lecture parallèle
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(executor.map(parse_label_file, label_files), total=len(label_files), desc="Parsing labels"))

    for boxes in results:
        if not boxes:
            empty_imgs += 1
        else:
            all_boxes.extend(boxes)

    # Conversion en numpy pour manipulation vectorisée
    if all_boxes:
        boxes_np = np.array(all_boxes, dtype=np.float32)
    else:
        boxes_np = np.zeros((0, 5), dtype=np.float32)

    # Comptage des classes
    if boxes_np.shape[0] > 0:
        cls_ids = boxes_np[:, 0].astype(np.int64)
        counts = Counter(cls_ids)
        counts = {int(k): int(v) for k, v in counts.items()}
    else:
        counts = {}

    return DatasetStats(
        total_images=len(label_files),
        total_objects=len(all_boxes),
        empty_images=empty_imgs,
        class_counts=counts,
        boxes_norm=boxes_np
    )

def plot_class_distribution(stats: DatasetStats, class_names: List[str], output_dir: Path):
    """Génère un histogramme de la distribution des classes."""
    if not stats.class_counts:
        return

    # Préparation données
    indices = sorted(stats.class_counts.keys())
    counts = [stats.class_counts[i] for i in indices]
    names = [class_names[i] if i < len(class_names) else str(i) for i in indices]

    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, counts, color='#3498db', alpha=0.8)
    plt.title(f"Distribution des Classes (Total: {stats.total_objects})")
    plt.xlabel("Classes")
    plt.ylabel("Nombre d'objets")
    plt.xticks(rotation=45, ha='right')
    
    # Ajout des valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=300)
    plt.close()

def plot_box_stats(stats: DatasetStats, output_dir: Path, ref_size: int):
    """Analyse la géométrie des boîtes (taille, ratio, position)."""
    if stats.boxes_norm.shape[0] == 0:
        return

    # Données: [class, x, y, w, h]
    w = stats.boxes_norm[:, 3]
    h = stats.boxes_norm[:, 4]
    x = stats.boxes_norm[:, 1]
    y = stats.boxes_norm[:, 2]

    # 1. Heatmap des positions (x, y)
    plt.figure(figsize=(8, 8))
    plt.hist2d(x, y, bins=50, cmap="viridis", range=[[0, 1], [0, 1]])
    plt.title("Heatmap: Position des Centres d'Objets (Normalisé)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().invert_yaxis() # Convention image
    plt.colorbar(label="Nombre d'objets")
    plt.savefig(output_dir / "object_heatmap.png", dpi=300)
    plt.close()

    # 2. Scatter plot Width vs Height
    plt.figure(figsize=(8, 8))
    plt.scatter(w, h, alpha=0.1, s=1, c='#e74c3c')
    plt.title(f"Dimensions des Objets (Normalisé)\n(Ref Size: {ref_size}px)")
    plt.xlabel("Width (norm)")
    plt.ylabel("Height (norm)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5) # Ligne ratio 1:1
    plt.savefig(output_dir / "box_sizes.png", dpi=300)
    plt.close()

    # 3. Histogramme des tailles (Area en pixels approx)
    areas_px = (w * ref_size) * (h * ref_size)
    small_obj_thresh = 32 * 32 # COCO definition for small objects
    
    ratio_small = np.sum(areas_px < small_obj_thresh) / len(areas_px)
    
    plt.figure(figsize=(10, 6))
    plt.hist(np.sqrt(areas_px), bins=100, color='#2ecc71', log=True)
    plt.title(f"Distribution de la racine carrée de l'aire (px)\nSmall objects (<32px): {ratio_small:.1%}")
    plt.xlabel("Sqrt(Area) en pixels (base ref_size)")
    plt.ylabel("Log Count")
    plt.axvline(x=32, color='r', linestyle='--', label='COCO Small limit (32px)')
    plt.legend()
    plt.savefig(output_dir / "object_sizes_dist.png", dpi=300)
    plt.close()

    return ratio_small

def generate_report(stats: DatasetStats, ratio_small: float, output_dir: Path):
    """Écrit un résumé textuel."""
    report = {
        "timestamp": str(logging.Formatter().formatTime(logging.makeLogRecord({}))),
        "total_images": stats.total_images,
        "empty_images": stats.empty_images,
        "empty_ratio": stats.empty_images / stats.total_images if stats.total_images else 0,
        "total_objects": stats.total_objects,
        "objects_per_image": stats.total_objects / stats.total_images if stats.total_images else 0,
        "small_objects_ratio": float(ratio_small),
        "class_counts": stats.class_counts
    }

    with open(output_dir / "analysis_report.json", 'w') as f:
        json.dump(report, f, indent=4)
    
    logger.info("--- RAPPORT SYNTHÉTIQUE ---")
    logger.info(f"Images Totales : {stats.total_images}")
    logger.info(f"Images Vides   : {stats.empty_images} ({report['empty_ratio']:.1%})")
    logger.info(f"Objets Totaux  : {stats.total_objects}")
    logger.info(f"Moyenne Obj/Img: {report['objects_per_image']:.2f}")
    logger.info(f"Ratio 'Small'  : {ratio_small:.1%} (< 32x32px sur {report.get('ref_size', 'ref')})")
    logger.info(f"Classes        : {len(stats.class_counts)} détectées.")
    logger.info(f"Rapport JSON   : {output_dir / 'analysis_report.json'}")

def main():
    args = parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Démarrage de l'analyse sur {args.data_root} [{args.split}]")

    # 1. Chargement
    stats = load_dataset(args.data_root, args.split, args.workers)

    if stats.total_images == 0:
        logger.warning("Aucune image trouvée. Fin.")
        return

    # 2. Récupération noms de classes (Tentative lecture YAML, sinon défaut)
    yaml_path = args.data_root / "VisDrone.yaml"
    class_names = DEFAULT_CLASSES
    if yaml_path.exists():
        try:
            import yaml
            with open(yaml_path, 'r') as f:
                data_cfg = yaml.safe_load(f)
                if 'names' in data_cfg:
                    names_cfg = data_cfg['names']
                    # names peut être dict ou list
                    if isinstance(names_cfg, dict):
                        class_names = [names_cfg[i] for i in sorted(names_cfg.keys())]
                    elif isinstance(names_cfg, list):
                        class_names = names_cfg
            logger.info(f"Noms de classes chargés depuis {yaml_path.name}")
        except ImportError:
            logger.warning("PyYAML non installé, utilisation des noms par défaut.")
        except Exception as e:
            logger.warning(f"Erreur lecture YAML ({e}), utilisation défaut.")

    # 3. Visualisation
    logger.info("Génération des graphiques...")
    plot_class_distribution(stats, class_names, args.output_dir)
    ratio_small = plot_box_stats(stats, args.output_dir, args.ref_imgsz)
    
    # 4. Rapport
    generate_report(stats, ratio_small if ratio_small else 0.0, args.output_dir)
    logger.info("Analyse terminée.")

if __name__ == "__main__":
    main()