# main_train.py
"""
Point d'entrée principal pour l'entraînement du modèle YOLOv11 sur VisDrone.
Orchestre la validation de la configuration, l'intégrité des données et le moteur d'entraînement.

Architecture :
1. Chargement & Validation Configuration (Schema)
2. Audit Data (Checker)
3. Lancement Engine (Trainer)

Usage:
    python main_train.py --data_root ./data/VisDrone_YOLO --epochs 200 --imgsz 1024 --device 0
"""

import argparse
import logging
import os
import sys
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Import modules internes (Architecture src_training/)
try:
    from src_training.config.schema import load_config, TrainingConfig
    from src_training.data.checker import DatasetValidator
    from src_training.engine.trainer import TrainerEngine
    # Les callbacks sont gérés via l'engine ou injectés si l'architecture le permet
except ImportError as e:
    print(f"❌ Erreur d'import critique : {e}")
    print("Assurez-vous d'être à la racine du projet.")
    sys.exit(1)

# --- Setup Inline ---
def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure le logging vers console et fichier."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "main_execution.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [MAIN] - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='a')
        ]
    )
    return logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """Reproductibilité basique avant chargement des modules lourds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# --- Main Logic ---

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VisDrone Training Pipeline (Senior Staff Arch)")
    
    # Paths
    parser.add_argument("--data_root", type=Path, required=True, help="Racine du dataset préparé")
    parser.add_argument("--output_dir", type=Path, default=Path("runs/train"), help="Dossier de sortie")
    parser.add_argument("--config", type=Path, default=Path("src_training/config/default_config.yaml"), help="Config YAML par défaut")
    
    # Overrides critiques
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="Poids initiaux (pt) ou config (yaml)")
    parser.add_argument("--epochs", type=int, default=None, help="Surcharge epochs")
    parser.add_argument("--batch", type=int, default=None, help="Surcharge batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Surcharge taille image (ex: 1024)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device ID ou cpu")
    parser.add_argument("--seed", type=int, default=42, help="Seed globale")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--name", type=str, default="exp", help="Nom de l'expérience")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Init Environnement
    set_seed(args.seed)
    logger = setup_logging(args.output_dir)
    logger.info("🚀 Démarrage de la pipeline d'entraînement VisDrone.")
    logger.info(f"Commande: {' '.join(sys.argv)}")

    # 2. Validation de la Configuration (Fail Fast)
    logger.info("--- [STEP 1] Validation Configuration ---")
    
    # Construction du dictionnaire d'overrides pour le schéma
    overrides = {
        'model': args.model,
        'data': str(args.data_root / "VisDrone.yaml"), # On suppose que le YAML est à la racine du data_root
        'device': args.device,
        'seed': args.seed,
        'project': str(args.output_dir),
        'name': args.name,
        'workers': args.workers
    }
    # Ajout conditionnel des arguments optionnels
    if args.epochs is not None: overrides['epochs'] = args.epochs
    if args.batch is not None: overrides['batch'] = args.batch
    if args.imgsz is not None: overrides['imgsz'] = args.imgsz

    try:
        validated_config = load_config(args.config, overrides=overrides)
        logger.info(f"✅ Configuration validée. Modele: {validated_config.model}, Imgsz: {validated_config.imgsz}")
    except Exception as e:
        logger.error(f"❌ Configuration invalide : {e}")
        sys.exit(1)

    # 3. Audit des Données (Sanity Check)
    logger.info("--- [STEP 2] Audit du Dataset ---")
    data_yaml_path = Path(validated_config.data)
    
    if not data_yaml_path.exists():
        # Tentative de reconstruction du chemin si relatif
        possible_path = args.data_root / "VisDrone.yaml"
        if possible_path.exists():
            data_yaml_path = possible_path
            # Mise à jour de l'arg pour le trainer
            args.data = data_yaml_path 
        else:
            logger.error(f"❌ Fichier dataset YAML introuvable : {data_yaml_path}")
            sys.exit(1)
            
    # Lancement du validateur
    validator = DatasetValidator(data_yaml_path, verbose=False)
    try:
        # On ne stop pas forcément tout le process sur un warning mineur, 
        # mais on audit. Checker.py fait sys.exit(1) si grave.
        validator.run() 
    except SystemExit as e:
        if e.code != 0:
            logger.error("❌ Échec critique de la validation des données.")
            sys.exit(1)
        logger.info("✅ Dataset audité avec succès.")

    # 4. Lancement de l'Entraînement (Engine)
    logger.info("--- [STEP 3] Lancement Engine ---")
    
    # Préparation des arguments pour le TrainerEngine
    # Le TrainerEngine attend un Namespace argparse avec 'data', 'config', etc.
    # Nous mettons à jour args avec les chemins validés
    args.data = data_yaml_path
    
    # Instanciation Moteur
    try:
        trainer = TrainerEngine(args)
        trainer.run()
    except KeyboardInterrupt:
        logger.warning("⚠️ Entraînement interrompu par l'utilisateur.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Erreur non gérée durant l'exécution : {e}", exc_info=True)
        sys.exit(1)

    logger.info("✅ Pipeline terminée.")

if __name__ == "__main__":
    main()