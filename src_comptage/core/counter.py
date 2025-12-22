"""
Module de logique pour le comptage.
Gère la "Machine à États" des véhicules : suivi des trajectoires, détection de franchissement de ligne
et agrégation des statistiques par classe/groupe.
"""

import sys
import logging
from collections import deque, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import numpy as np

# Gestion des imports
try:
    from src_comptage.config.settings import SETTINGS
    from src_comptage.utils.geometry import is_segment_crossing, get_crossing_direction
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src_comptage.config.settings import SETTINGS
    from src_comptage.utils.geometry import is_segment_crossing, get_crossing_direction

logger = logging.getLogger(__name__)

class VehicleCounter:
    """
    Gère la logique de comptage par franchissement de ligne.
    """

    def __init__(self):
        self.line = SETTINGS.LINE_COORDINATES
        self.class_mapping = SETTINGS.TARGET_CLASSES
        
        # On augmente la mémoire pour permettre de regarder plus loin en arrière
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=50))
        self.counted_ids: Set[int] = set()
        self.counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "in": 0, "out": 0})

        logger.info(f"✅ Compteur initialisé (Mode Centre de gravité). Ligne: {self.line}")

    def update(self, tracks: np.ndarray) -> Dict[str, Dict[str, int]]:
        """
        Met à jour l'état du compteur avec une logique plus robuste.
        """
        for track in tracks:
            x1, y1, x2, y2 = track[:4]
            track_id = int(track[4])
            cls_id = int(track[6])
            
            if cls_id not in self.class_mapping:
                continue
                
            group_name = self.class_mapping[cls_id]

            # UTILISER le centre absolu
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            current_point = (center_x, center_y)

            self.track_history[track_id].append(current_point)
            
            HISTORY_STRIDE = 3 # Regarder 3 frames en arrière
            
            if len(self.track_history[track_id]) > HISTORY_STRIDE:
                # Point actuel
                curr = self.track_history[track_id][-1]
                # Point il y a quelques frames
                prev = self.track_history[track_id][-(HISTORY_STRIDE + 1)]
                
                if track_id not in self.counted_ids:
                    # Test mathématique du franchissement
                    if is_segment_crossing(prev, curr, self.line[0], self.line[1]):
                        
                        direction_code = get_crossing_direction(prev, curr, self.line[0], self.line[1])
                        
                        if direction_code != "none":
                            semantic_dir = "in" if direction_code == "left_to_right" else "out"
                            self._increment_count(group_name, semantic_dir, track_id)
                            # logger.info(f"🚗 ID: {track_id} ({group_name}) -> {semantic_dir}")

        return dict(self.counts)

    def _increment_count(self, group: str, direction: str, track_id: int):
        self.counts[group]["total"] += 1
        self.counts[group][direction] += 1
        self.counted_ids.add(track_id)

    def get_counts_formatted(self) -> str:
        summary = []
        for group, vals in self.counts.items():
            summary.append(f"{group}: {vals['total']}")
        return " | ".join(summary)