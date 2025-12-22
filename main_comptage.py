"""
Orchestrateur principal du pipeline de comptage de véhicules.
Charge les modules, lance la boucle d'inférence, gère les I/O et sauvegarde les résultats.
"""

import argparse
import cv2
import logging
import sys
import time
from pathlib import Path
from tqdm import tqdm

# Ajout du dossier parent au path pour les imports si exécuté depuis la racine
current_file = Path(__file__).resolve()
project_root = current_file.parent
sys.path.append(str(project_root))

from src_comptage.config.settings import SETTINGS
from src_comptage.utils.video_io import VideoStream
from src_comptage.utils.visualizer import Visualizer
from src_comptage.core.tracker import ObjectTracker
from src_comptage.core.counter import VehicleCounter

# Configuration Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("inference.log", mode="w")
    ]
)
logger = logging.getLogger("MainPipeline")

def parse_arguments():
    """Gère les arguments en ligne de commande pour surcharger la config."""
    parser = argparse.ArgumentParser(description="Pipeline de Comptage de Véhicules YOLOv11")
    
    parser.add_argument("--source", type=str, default=None, 
                        help="Chemin vidéo source (écrase settings.VIDEO_SOURCE)")
    parser.add_argument("--model", type=str, default=None, 
                        help="Chemin modèle .pt (écrase settings.MODEL_PATH)")
    parser.add_argument("--headless", action="store_true", 
                        help="Désactive l'affichage cv2.imshow (utile pour serveurs)")
    parser.add_argument("--output", type=str, default=None, 
                        help="Nom du fichier de sortie (dans le dossier output)")
    
    return parser.parse_args()

def setup_video_writer(output_path: Path, width: int, height: int, fps: float):
    """Initialise l'écriture vidéo."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        logger.error(f"❌ Impossible de créer le fichier sortie : {output_path}")
        raise IOError("Erreur VideoWriter")
    return writer

def run_pipeline(args):
    """Boucle principale d'inférence."""
    
    # 1. Configuration
    source = args.source if args.source else SETTINGS.VIDEO_SOURCE
    model_path = Path(args.model) if args.model else SETTINGS.MODEL_PATH
    output_dir = SETTINGS.OUTPUT_DIR
    
    logger.info("🚀 Démarrage du pipeline de comptage")
    logger.info(f"📂 Source : {source}")
    logger.info(f"🧠 Modèle : {model_path}")
    logger.info(f"💾 Sortie : {output_dir}")

    # 2. Initialisation des modules
    try:
        tracker = ObjectTracker(model_path=model_path)
        counter = VehicleCounter()
        visualizer = Visualizer()
    except Exception as e:
        logger.critical(f"❌ Échec initialisation modules : {e}")
        return

    # 3. Traitement Vidéo
    with VideoStream(source) as stream:
        width, height = stream.resolution
        fps = stream.cap.get(cv2.CAP_PROP_FPS)
        total_frames = stream.total_frames
        
        # Préparation sortie
        output_filename = args.output if args.output else f"result_{Path(str(source)).name}"
        if not output_filename.endswith('.mp4'):
            output_filename += '.mp4'
        save_path = output_dir / output_filename
        
        writer = setup_video_writer(save_path, width, height, fps)
        
        # Métriques performance
        start_time = time.time()
        frame_count = 0
        
        # Barre de progression
        pbar = tqdm(total=total_frames, desc="Traitement", unit="frame")
        
        try:
            for idx, frame in stream:
                loop_start = time.time()
                
                # A. Tracking (Détection + Association)
                tracks = tracker.update(frame)
                
                # B. Comptage (Logique Métier)
                counts = counter.update(tracks)
                
                # C. Visualisation (Dessin)
                annotated_frame = visualizer.render(frame, tracks, counts)
                
                # D. Sauvegarde & Affichage
                writer.write(annotated_frame)
                
                if not args.headless:
                    # Redimensionnement pour affichage écran si trop grand
                    display_frame = annotated_frame
                    if width > 1280:
                        scale = 1280 / width
                        display_frame = cv2.resize(annotated_frame, (0, 0), fx=scale, fy=scale)
                    
                    cv2.imshow("FlowSight Monitor", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("🛑 Interruption utilisateur (Q).")
                        break
                
                frame_count += 1
                pbar.update(1)
                
        except KeyboardInterrupt:
            logger.info("🛑 Interruption clavier (Ctrl+C).")
        except Exception as e:
            logger.error(f"❌ Erreur en cours de boucle : {e}", exc_info=True)
        finally:
            writer.release()
            pbar.close()
            if not args.headless:
                cv2.destroyAllWindows()

    # 4. Rapport Final
    end_time = time.time()
    duration = end_time - start_time
    avg_fps = frame_count / duration if duration > 0 else 0
    
    logger.info("-" * 40)
    logger.info("✅ Traitement terminé avec succès.")
    logger.info(f"⏱  Durée : {duration:.2f}s ({avg_fps:.1f} FPS moyen)")
    logger.info(f"📊 Statistiques Finales :")
    logger.info(f"   {counter.get_counts_formatted()}")
    logger.info(f"💾 Vidéo sauvegardée : {save_path}")
    logger.info("-" * 40)

if __name__ == "__main__":
    cli_args = parse_arguments()
    run_pipeline(cli_args)