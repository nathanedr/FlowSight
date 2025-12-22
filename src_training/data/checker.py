"""
Outil de validation d'intégrité de dataset YOLO avant entraînement.
Vérifie la correspondance images/labels, la validité des fichiers et les bornes des annotations.
Essentiel pour éviter les crashs silencieux ou explicites en cours de training.

Usage:
    python src_training/data/checker.py --config ./VisDrone.yaml
"""

import argparse
import logging
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml
from tqdm import tqdm
from PIL import Image

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [DATA-CHECK] - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class DatasetValidator:
    def __init__(self, config_path: Path, verbose: bool = False):
        self.config_path = config_path
        self.verbose = verbose
        self.config = self._load_config()
        self.root_dir = Path(self.config.get('path', config_path.parent))
        
        # Récupération des classes
        names = self.config.get('names', {})
        if isinstance(names, list):
            self.num_classes = len(names)
        elif isinstance(names, dict):
            self.num_classes = len(names)
        else:
            logger.error("Format de 'names' invalide dans le YAML.")
            sys.exit(1)
            
        logger.info(f"Configuration chargée. {self.num_classes} classes définies.")
        logger.info(f"Racine dataset déduite : {self.root_dir}")

    def _load_config(self) -> Dict:
        if not self.config_path.exists():
            logger.error(f"Fichier config introuvable : {self.config_path}")
            sys.exit(1)
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Erreur parsing YAML : {e}")
            sys.exit(1)

    def resolve_path(self, path_str: str) -> Path:
        """Résout les chemins relatifs par rapport au root du dataset."""
        p = Path(path_str)
        if p.is_absolute():
            return p
        return (self.root_dir / p).resolve()

    @staticmethod
    def _check_pair(args: Tuple[Path, Optional[Path], int]) -> Dict:
        """
        Worker function pour validation unitaire (multiprocess safe).
        Args: (img_path, label_path, num_classes)
        Returns: Dict d'erreurs
        """
        img_path, lbl_path, num_classes = args
        errors = []
        stats = {"valid_img": False, "valid_lbl": False, "objects": 0}

        # 1. Verifie les images
        try:
            with Image.open(img_path) as img:
                img.verify() # Vérifie l'intégrité de l'en-tête
            stats["valid_img"] = True
        except Exception:
            errors.append(f"Image corrompue: {img_path.name}")
            return {"file": img_path.name, "errors": errors, "stats": stats}

        # 2. Vérifie les labels
        if lbl_path and lbl_path.exists():
            try:
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                
                valid_lines = 0
                for idx, line in enumerate(lines):
                    parts = line.strip().split()
                    if not parts:
                        continue # Ligne vide OK
                    
                    if len(parts) != 5:
                        errors.append(f"Label malformé (attendu 5 valeurs): ligne {idx+1}")
                        continue
                    
                    try:
                        cls = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                    except ValueError:
                        errors.append(f"Valeurs non numériques: ligne {idx+1}")
                        continue

                    # Vérifie l'ID de classe
                    if not (0 <= cls < num_classes):
                        errors.append(f"Class ID {cls} hors limites (0-{num_classes-1})")

                    # Vérifie les coordonnées (normalisées 0-1)
                    if any(c < 0.0 or c > 1.0 for c in coords):
                        errors.append(f"Coordonnées non normalisées (doit être 0-1): ligne {idx+1}")
                    
                    # Vérifie les dimensions positives
                    if coords[2] <= 0 or coords[3] <= 0: # width, height
                        errors.append(f"Dimensions négatives ou nulles: ligne {idx+1}")

                    valid_lines += 1
                
                stats["valid_lbl"] = True
                stats["objects"] = valid_lines

            except Exception as e:
                errors.append(f"Erreur lecture label: {e}")
        elif lbl_path and not lbl_path.exists():
            # YOLO accepte les images sans label (background images)
            # Mais c'est souvent une erreur d'oubli
            pass 

        return {"file": img_path.name, "errors": errors, "stats": stats}

    def scan_split(self, split_key: str) -> bool:
        """Scanne un split (train/val/test). Retourne True si OK (avec warnings possibles)."""
        rel_path = self.config.get(split_key)
        if not rel_path:
            logger.warning(f"Split '{split_key}' non défini dans le YAML. Skip.")
            return True

        # Gestion format liste ou string
        paths = rel_path if isinstance(rel_path, list) else [rel_path]
        
        all_ok = True
        total_errors = 0
        total_objects = 0
        
        for p_str in paths:
            img_dir = self.resolve_path(p_str)
            if not img_dir.exists():
                logger.error(f"Dossier images introuvable : {img_dir}")
                all_ok = False
                continue

            # Trouver les images
            valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif'}
            images = [p for p in img_dir.iterdir() if p.suffix.lower() in valid_exts]
            
            if not images:
                logger.warning(f"Aucune image trouvée dans {img_dir}")
                continue

            # Déduire dossier labels (convention YOLO: ../images/x -> ../labels/x)
            # Ou au même niveau avec suffixe .txt
            label_dir = img_dir.parent.parent / 'labels' / img_dir.name
            if not label_dir.exists():
                # Fallback: check si labels est au meme niveau
                label_dir = img_dir.parent / 'labels'
            
            logger.info(f"Scanning '{split_key}': {len(images)} images (Labels prévus dans: {label_dir})")

            # Préparation des tâches
            tasks = []
            for img in images:
                lbl = label_dir / f"{img.stem}.txt"
                tasks.append((img, lbl, self.num_classes))

            # Exécution parallèle
            chunk_size = max(1, len(tasks) // (multiprocessing.cpu_count() * 4))
            with ProcessPoolExecutor() as executor:
                results = list(tqdm(
                    executor.map(self._check_pair, tasks, chunksize=chunk_size), 
                    total=len(tasks), 
                    desc=f"Validating {split_key}"
                ))

            # Analyse des résultats
            split_errors = 0
            for res in results:
                total_objects += res["stats"]["objects"]
                if res["errors"]:
                    split_errors += 1
                    total_errors += 1
                    if self.verbose:
                        for err in res["errors"]:
                            logger.warning(f"[{res['file']}] {err}")
            
            if split_errors > 0:
                logger.warning(f"Trouvé {split_errors} fichiers problématiques dans {split_key}.")
                all_ok = False
            else:
                logger.info(f"✅ Split {split_key} clean. ({total_objects} objets trouvés)")

        return all_ok

    def run(self):
        logger.info("Démarrage de la validation globale...")
        
        # Vérification des splits standards YOLO
        splits = ['train', 'val', 'test']
        status = {}
        
        for split in splits:
            status[split] = self.scan_split(split)
            
        logger.info("--- Résumé de l'Audit ---")
        failed = [s for s, ok in status.items() if not ok]
        
        if failed:
            logger.error(f"❌ Échecs détectés sur : {', '.join(failed)}. Voir logs ci-dessus.")
            sys.exit(1)
        else:
            logger.info("🚀 Dataset Sain. Prêt pour entraînement.")
            sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Check YOLO Dataset Integrity")
    parser.add_argument("--config", type=Path, required=True, help="Chemin vers dataset.yaml")
    parser.add_argument("--verbose", action="store_true", help="Afficher toutes les erreurs individuelles")
    
    args = parser.parse_args()
    
    validator = DatasetValidator(args.config, verbose=args.verbose)
    validator.run()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()