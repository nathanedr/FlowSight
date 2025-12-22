"""
Moteur d'entraînement (Trainer Engine) pour YOLOv11 / VisDrone.
Intègre la gestion des hyperparamètres, le cycle de vie du modèle et les callbacks de production.

Ce module dépend de 'src_training.engine.callbacks' pour le monitoring avancé.

Usage:
    python -m src_training.engine.trainer --data VisDrone.yaml --epochs 100 --device 0
"""

import argparse
import gc
import logging
import os
import sys
import random
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import yaml
from ultralytics import YOLO

# Import conditionnel pour gérer l'exécution standalone vs package
try:
    from src_training.engine.callbacks import get_callbacks
except ImportError:
    # Si exécuté directement sans le path configuré, on tente d'ajouter la racine
    current_path = Path(__file__).resolve()
    root_path = current_path.parent.parent.parent
    sys.path.append(str(root_path))
    try:
        from src_training.engine.callbacks import get_callbacks
    except ImportError:
        print("❌ Erreur critique : Impossible d'importer src_training.engine.callbacks")
        sys.exit(1)

# Configuration du logging (local au module)
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [TRAINER] - %(levelname)s - %(message)s")

class TrainerEngine:
    def __init__(self, args: argparse.Namespace):
        """
        Initialise le moteur d'entraînement.
        
        Args:
            args: Namespace contenant les arguments CLI et hyperparamètres.
        """
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = self._select_device(args.device)
        logger.info(f"Trainer initialisé. Output: {self.output_dir}, Device: {self.device}")

    def _select_device(self, device_arg: str) -> str:
        """Valide et sélectionne le périphérique de calcul."""
        if device_arg == 'cpu':
            return 'cpu'
        
        if torch.cuda.is_available():
            # Vérification simple de l'index
            try:
                if ',' in device_arg: return device_arg # Multi-GPU
                idx = int(device_arg)
                if idx < torch.cuda.device_count():
                    return str(idx)
            except ValueError:
                pass
        
        logger.warning(f"Device demandé '{device_arg}' invalide ou CUDA indisponible. Fallback CPU.")
        return 'cpu'

    def _load_config(self) -> Dict[str, Any]:
        """
        Prépare le dictionnaire de configuration pour YOLO.
        Fusionne le fichier YAML (si fourni) avec les arguments CLI.
        """
        cfg = {}
        
        # 1. Chargement YAML optionnel
        if self.args.config and Path(self.args.config).exists():
            try:
                with open(self.args.config, 'r') as f:
                    cfg = yaml.safe_load(f) or {}
                logger.info(f"Base config chargée depuis {self.args.config}")
            except yaml.YAMLError as e:
                logger.error(f"Erreur lecture YAML: {e}")
                sys.exit(1)
        
        # 2. Surcharge CLI (Priorité)
        # On ne passe que les arguments explicitement définis (non None)
        overrides = {
            'data': str(self.args.data),
            'model': self.args.model,
            'epochs': self.args.epochs,
            'batch': self.args.batch,
            'imgsz': self.args.imgsz,
            'device': self.device,
            'project': str(self.output_dir.parent),
            'name': self.output_dir.name,
            'workers': self.args.workers,
            'seed': self.args.seed,
            'exist_ok': True,
            'val': True,
            'plots': True,
            'save': True
        }
        
        # Nettoyage et fusion
        clean_overrides = {k: v for k, v in overrides.items() if v is not None}
        cfg.update(clean_overrides)
        
        return cfg

    def _setup_reproducibility(self, seed: int):
        """Fixe les graines aléatoires pour garantir la reproductibilité."""
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ultralytics gère aussi le deterministic=True dans args, mais ceci est une sécurité
        os.environ['PYTHONHASHSEED'] = str(seed)

    def run(self):
        """Exécute la pipeline d'entraînement complète."""
        self._setup_reproducibility(self.args.seed)
        
        train_args = self._load_config()
        logger.info(f"Configuration finale: Model={train_args.get('model')}, Imgsz={train_args.get('imgsz')}")

        # 1. Chargement Modèle
        try:
            model = YOLO(train_args['model'])
            logger.info("Modèle YOLO chargé avec succès.")
        except Exception as e:
            logger.error(f"Échec chargement modèle: {e}")
            sys.exit(1)

        # 2. Injection des Callbacks (Monitoring)
        # Récupération des fonctions via le module callbacks.py
        try:
            custom_callbacks = get_callbacks(self.output_dir)
            for event, func in custom_callbacks.items():
                model.add_callback(event, func)
            logger.info(f"{len(custom_callbacks)} callbacks personnalisés enregistrés.")
        except Exception as e:
            logger.warning(f"Erreur lors de l'enregistrement des callbacks: {e}")

        # 3. Lancement Entraînement
        logger.info("🚀 Démarrage de model.train()...")
        try:
            results = model.train(**train_args)
            logger.info(f"Entraînement terminé. Résultats sauvegardés dans {results.save_dir}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("❌ OOM (Out Of Memory). Réduisez --batch ou --imgsz.")
                torch.cuda.empty_cache()
            else:
                logger.error(f"❌ Erreur Runtime: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"❌ Erreur inattendue: {e}")
            sys.exit(1)
        finally:
            self._cleanup()

    def _cleanup(self):
        """Nettoyage mémoire GPU post-run."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv11 Trainer Engine for VisDrone")
    
    # Arguments structurels
    parser.add_argument("--data", type=Path, required=True, help="Chemin vers data.yaml")
    parser.add_argument("--output_dir", type=Path, default=Path("runs/train/exp"), help="Dossier de sortie")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML par défaut")
    
    # Hyperparamètres
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="Poids ou config modèle")
    parser.add_argument("--epochs", type=int, default=None, help="Nombre d'époques")
    parser.add_argument("--batch", type=int, default=None, help="Taille du batch")
    parser.add_argument("--imgsz", type=int, default=None, help="Taille image (ex: 640, 1024)")
    parser.add_argument("--device", type=str, default="0", help="ID GPU (ex: 0) ou cpu")
    parser.add_argument("--workers", type=int, default=8, help="Threads data loader")
    parser.add_argument("--seed", type=int, default=42, help="Seed globale")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Garde-fou existence Data
    if not args.data.exists():
        logger.error(f"Fichier data introuvable : {args.data.absolute()}")
        sys.exit(1)

    engine = TrainerEngine(args)
    engine.run()