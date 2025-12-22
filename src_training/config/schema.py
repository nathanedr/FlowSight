"""
Schéma de validation stricte pour la configuration d'entraînement YOLO.
Utilise les dataclasses Python 3.11+ pour garantir le typage et l'intégrité des hyperparamètres.

Usage:
    from src_training.config.schema import load_config
    cfg = load_config("src_training/config/default_config.yaml", overrides={'epochs': 300})
"""

import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

import yaml

# Logger dédié à la config
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - [CONFIG] - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

@dataclass(frozen=True)
class TrainingConfig:
    """
    Configuration immuable représentant l'état complet d'un entraînement YOLO.
    """
    # --- Task & Mode ---
    task: str
    mode: str
    
    # --- Model & Data ---
    model: str
    data: str
    
    # --- Training Mechanics ---
    device: Union[str, int, List[int]]
    epochs: int
    patience: int
    batch: int
    imgsz: int
    save: bool
    save_period: int
    cache: bool
    workers: int
    project: str
    name: str
    exist_ok: bool
    pretrained: bool
    optimizer: str
    seed: int
    deterministic: bool
    single_cls: bool
    rect: bool
    cos_lr: bool
    close_mosaic: int
    resume: bool
    amp: bool
    fraction: float
    profile: bool

    # --- Validation ---
    val: bool
    split: str
    conf: float
    iou: float
    max_det: int
    plots: bool

    # --- Augmentations ---
    hsv_h: float
    hsv_s: float
    hsv_v: float
    degrees: float
    translate: float
    scale: float
    shear: float
    perspective: float
    flipud: float
    fliplr: float
    mosaic: float
    mixup: float
    copy_paste: float

    # --- Catch-all pour paramètres YOLO futurs/inconnus ---
    # Permet la flexibilité si Ultralytics ajoute des params sans casser le schéma strict
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validation logique métier après initialisation."""
        self._validate_mechanics()
        self._validate_augmentations()
        self._validate_paths()

    def _validate_mechanics(self):
        if self.epochs < 1:
            raise ValueError(f"Epochs doit être >= 1, reçu {self.epochs}")
        
        if self.imgsz % 32 != 0:
            raise ValueError(f"imgsz doit être multiple de 32 (stride YOLO), reçu {self.imgsz}")
        
        if self.batch < 1 and self.batch != -1: # -1 = autobatch
            raise ValueError(f"Batch size invalide: {self.batch}")
            
        if not (0.0 < self.fraction <= 1.0):
             raise ValueError(f"Fraction dataset doit être entre 0.0 et 1.0, reçu {self.fraction}")

    def _validate_augmentations(self):
        """Vérifie la cohérence des augments pour la vue aérienne (VisDrone)."""
        probs = [
            ('hsv_h', self.hsv_h), ('hsv_s', self.hsv_s), ('hsv_v', self.hsv_v),
            ('translate', self.translate), ('scale', self.scale),
            ('flipud', self.flipud), ('fliplr', self.fliplr),
            ('mosaic', self.mosaic), ('mixup', self.mixup)
        ]
        
        for name, val in probs:
            if not (0.0 <= val <= 1.0) and name not in ['degrees', 'shear', 'perspective']: 
                # degrees/shear peuvent être > 1, mais les probs/ratios doivent être [0,1]
                # Note: Dans YOLO config, scale est un gain (ex: 0.5 = +/- 50%), pas une proba.
                # flipud/lr/mosaic/mixup sont des probabilités.
                if name in ['flipud', 'fliplr', 'mosaic', 'mixup']:
                     if not (0.0 <= val <= 1.0):
                        raise ValueError(f"{name} doit être une probabilité [0,1], reçu {val}")

        # Règle métier VisDrone : Pas de flip vertical (les voitures ne volent pas)
        if self.flipud > 0.05:
            logger.warning(f"⚠️ Attention: flipud={self.flipud} > 0. Pour VisDrone (vue aérienne), cela devrait être proche de 0.")

    def _validate_paths(self):
        # Vérification simple de l'extension du modèle
        if not str(self.model).endswith(('.pt', '.yaml', '.yml')):
            raise ValueError(f"Format de modèle inconnu: {self.model}. Attendu .pt ou .yaml")

    def to_dict(self) -> Dict[str, Any]:
        """Exporte la config en dictionnaire plat pour Ultralytics."""
        base = asdict(self)
        extras = base.pop('extra_params')
        return {**base, **extras}


def load_config(yaml_path: Union[str, Path], overrides: Optional[Dict[str, Any]] = None) -> TrainingConfig:
    """
    Charge, merge et valide la configuration.
    
    Args:
        yaml_path: Chemin vers le fichier YAML par défaut.
        overrides: Dictionnaire d'arguments (ex: venant de argparse) qui écrasent le YAML.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier de config introuvable : {path.absolute()}")

    with open(path, 'r') as f:
        try:
            raw_cfg = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Erreur de syntaxe YAML dans {path}: {e}")

    # Fusion des overrides
    if overrides:
        # Nettoyage des None (arguments CLI non définis)
        clean_overrides = {k: v for k, v in overrides.items() if v is not None}
        raw_cfg.update(clean_overrides)

    # Séparation des champs connus et inconnus
    known_fields = TrainingConfig.__annotations__.keys()
    known_args = {}
    extra_args = {}

    for k, v in raw_cfg.items():
        if k in known_fields:
            known_args[k] = v
        else:
            extra_args[k] = v

    # Instanciation (déclenche __post_init__)
    try:
        cfg = TrainingConfig(**known_args, extra_params=extra_args)
        logger.info(f"✅ Configuration chargée et validée depuis {path.name}")
        return cfg
    except TypeError as e:
        # Souvent causé par un champ manquant requis
        missing = set(known_fields) - set(known_args.keys()) - {'extra_params'}
        raise ValueError(f"Configuration incomplète. Champs manquants : {missing}. Détail: {e}")
    except ValueError as e:
        raise ValueError(f"Validation métier échouée : {e}")


if __name__ == "__main__":
    # Test simple : création d'un fichier dummy et validation
    dummy_yaml = Path("temp_test_config.yaml")
    
    # Simulation d'un contenu YAML valide pour VisDrone
    content = """
task: detect
mode: train
model: yolo11n.pt
data: VisDrone.yaml
device: 0
epochs: 100
patience: 50
batch: 16
imgsz: 640
save: true
save_period: 10
cache: false
workers: 4
project: runs/test
name: exp
exist_ok: true
pretrained: true
optimizer: auto
seed: 42
deterministic: true
single_cls: false
rect: false
cos_lr: true
close_mosaic: 10
resume: false
amp: true
fraction: 1.0
profile: false
val: true
split: val
conf: 0.25
iou: 0.7
max_det: 300
plots: true
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.1
copy_paste: 0.0
"""
    with open(dummy_yaml, "w") as f:
        f.write(content)

    try:
        # Test 1: Chargement valide
        logger.info("--- Test 1: Config Valide ---")
        cfg = load_config(dummy_yaml)
        print(f"Loaded imgsz: {cfg.imgsz}")

        # Test 2: Validation erreur (imgsz incorrect)
        logger.info("--- Test 2: Validation Error (imgsz) ---")
        try:
            load_config(dummy_yaml, overrides={'imgsz': 630}) # Pas multiple de 32
        except ValueError as e:
            logger.info(f"Succès, erreur attrapée : {e}")

        # Test 3: Warning métier (flipud)
        logger.info("--- Test 3: Warning Métier (FlipUD) ---")
        load_config(dummy_yaml, overrides={'flipud': 0.5})

    finally:
        if dummy_yaml.exists():
            dummy_yaml.unlink()