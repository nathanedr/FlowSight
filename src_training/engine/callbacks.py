"""
Gestionnaires d'événements (Callbacks) avancés pour l'entraînement YOLO.
Fournit un monitoring système (GPU/RAM), un logging structuré (JSON) et des alertes de performance.
Conçu pour être injecté dans le Trainer Ultralytics via `model.add_callback`.

Usage (dans trainer.py):
    from src_training.engine.callbacks import get_callbacks
    for event, func in get_callbacks(output_dir).items():
        model.add_callback(event, func)
"""

import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Callable, Any, List

import psutil
import torch

# Logger dédié
logger = logging.getLogger(__name__)

class ProductionCallbacks:
    """
    Ensemble de callbacks pour surveiller la santé de l'entraînement et des ressources.
    """
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.metrics_file = log_dir / "metrics_history.jsonl"
        self.system_file = log_dir / "system_stats.csv"
        self.start_time = time.time()
        
        # Initialisation des fichiers logs
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
        with open(self.system_file, 'w') as f:
            f.write("epoch,timestamp,gpu_mem_gb,ram_percent,cpu_percent\n")

    def on_pretrain_routine_start(self, trainer: Any):
        """Audit avant démarrage : vérifie l'espace disque et la config."""
        logger.info("--- [CALLBACK] Initialisation Pre-Train ---")
        
        # 1. Check Disk Space
        total, used, free = shutil.disk_usage(self.log_dir)
        free_gb = free / (1024**3)
        if free_gb < 10.0:
            logger.warning(f"⚠️ Espace disque faible : {free_gb:.2f} GB libres sur {self.log_dir}")
        else:
            logger.info(f"💾 Espace disque OK : {free_gb:.2f} GB libres.")

        # 2. Log Config Snapshot
        args_dump = getattr(trainer, 'args', {})
        if hasattr(args_dump, '__dict__'):
            args_dump = args_dump.__dict__
            
        with open(self.log_dir / "config_snapshot.json", "w") as f:
            json.dump(args_dump, f, indent=4, default=str)

    def on_train_epoch_start(self, trainer: Any):
        """Nettoyage proactif du cache GPU."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_train_epoch_end(self, trainer: Any):
        """Monitoring des ressources système après chaque epoch."""
        epoch = trainer.epoch + 1
        
        # Stats Système
        gpu_mem = 0.0
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_reserved(0) / 1E9
        
        ram_pct = psutil.virtual_memory().percent
        cpu_pct = psutil.cpu_percent()

        # Log CSV System
        with open(self.system_file, 'a') as f:
            f.write(f"{epoch},{datetime.now().isoformat()},{gpu_mem:.2f},{ram_pct},{cpu_pct}\n")
            
        # Alertes seuils critiques
        if ram_pct > 90:
            logger.warning(f"⚠️ Alerte RAM Critique : {ram_pct}% utilisé à l'epoch {epoch}")
        if gpu_mem > 23.0: # Pour une carte 24GB
            logger.warning(f"⚠️ Alerte VRAM Critique : {gpu_mem:.2f} GB réservés")

    def on_fit_epoch_end(self, trainer: Any):
        """Logging structuré des métriques de validation (JSONL)."""
        metrics = trainer.metrics
        epoch = trainer.epoch + 1
        
        # Construction d'un objet de log propre
        log_entry = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            # Metrics principales (Detection)
            "mAP_50": metrics.get("metrics/mAP50(B)", 0),
            "mAP_50_95": metrics.get("metrics/mAP50-95(B)", 0),
            "precision": metrics.get("metrics/precision(B)", 0),
            "recall": metrics.get("metrics/recall(B)", 0),
            # Loss values (Train/Val)
            "train_box_loss": trainer.loss_items[0].item() if len(trainer.loss_items) > 0 else 0,
            "train_cls_loss": trainer.loss_items[1].item() if len(trainer.loss_items) > 1 else 0,
            "lr": trainer.optimizer.param_groups[0]['lr']
        }
        
        # Append JSONL
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
            
        # Log Console Senior (VisDrone Focus : mAP 50-95 est la clé)
        map_val = log_entry["mAP_50_95"]
        best_val = trainer.best_fitness
        is_best = (map_val == best_val) and (epoch > 1)
        
        status_icon = "⭐ BEST" if is_best else ""
        logger.info(f"Epoch {epoch} | mAP@50-95: {map_val:.4f} {status_icon} | VRAM: {torch.cuda.memory_reserved()/1e9:.1f}GB")

    def on_train_end(self, trainer: Any):
        """Résumé final."""
        duration = time.time() - self.start_time
        hours = duration / 3600
        logger.info(f"✅ Entraînement terminé en {hours:.2f} heures.")
        logger.info(f"📁 Logs complets disponibles dans : {self.log_dir}")


def get_callbacks(output_dir: Path) -> Dict[str, Callable]:
    """
    Factory method pour instancier et relier les callbacks au format Ultralytics.
    
    Returns:
        Dictionnaire {event_name: callback_function}
    """
    cb_instance = ProductionCallbacks(output_dir)
    
    return {
        "on_pretrain_routine_start": cb_instance.on_pretrain_routine_start,
        "on_train_epoch_start": cb_instance.on_train_epoch_start,
        "on_train_epoch_end": cb_instance.on_train_epoch_end,
        "on_fit_epoch_end": cb_instance.on_fit_epoch_end,
        "on_train_end": cb_instance.on_train_end
    }

def main():
    """Test unitaire des callbacks (Mocking)."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Test des callbacks en mode Mock...")
    
    temp_dir = Path("runs/test_callbacks")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    # Mock Ultralytics Trainer
    class MockTrainer:
        def __init__(self):
            self.epoch = 0
            self.epochs = 10
            self.args = argparse.Namespace(model="test.pt", imgsz=640)
            self.metrics = {"metrics/mAP50(B)": 0.5, "metrics/mAP50-95(B)": 0.3}
            self.loss_items = [torch.tensor(0.1), torch.tensor(0.2)]
            self.optimizer = argparse.Namespace(param_groups=[{'lr': 0.01}])
            self.best_fitness = 0.3

    trainer = MockTrainer()
    callbacks = get_callbacks(temp_dir)
    
    # Simulation cycle
    callbacks["on_pretrain_routine_start"](trainer)
    callbacks["on_train_epoch_start"](trainer)
    callbacks["on_train_epoch_end"](trainer)
    callbacks["on_fit_epoch_end"](trainer)
    callbacks["on_train_end"](trainer)
    
    logger.info("Test terminé. Vérifier le dossier runs/test_callbacks")

if __name__ == "__main__":
    import argparse # Requis pour le mock
    main()