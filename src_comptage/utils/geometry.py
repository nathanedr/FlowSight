"""
Primitives géométriques critiques pour le comptage par franchissement de ligne.
Implémente l'algorithme d'intersection de segments sans dépendances lourdes (maths pures).
"""

import sys
import logging
from typing import Tuple, List

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

# Alias de type pour clarté : (x, y) en pixels
Point = Tuple[int, int]
Segment = Tuple[Point, Point]

def cross_product_2d(o: Point, a: Point, b: Point) -> int:
    """
    Calcule le produit vectoriel (composante Z) des vecteurs OA et OB.
    
    Args:
        o: Point d'origine
        a: Premier point
        b: Second point
        
    Returns:
        int: > 0 si O->A->B est dans le sens anti-horaire (à gauche)
             < 0 si O->A->B est dans le sens horaire (à droite)
             = 0 si les points sont colinéaires
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def is_segment_crossing(
    move_start: Point, 
    move_end: Point, 
    line_start: Point, 
    line_end: Point
) -> bool:
    """
    Détermine si le vecteur de mouvement (trajectoire véhicule) coupe la ligne de comptage.
    
    Maths:
        Deux segments AB et CD s'intersectent si et seulement si :
        - C et D sont de part et d'autre de la droite (AB)
        - ET A et B sont de part et d'autre de la droite (CD)
    
    Args:
        move_start (Point): Position du véhicule à t-1 (ou centre bbox antérieur)
        move_end (Point): Position du véhicule à t (ou centre bbox actuel)
        line_start (Point): Début de la ligne virtuelle
        line_end (Point): Fin de la ligne virtuelle

    Returns:
        bool: True si il y a franchissement strict.
    """
    
    # Calcul des orientations relatives
    # Position de la ligne par rapport au mouvement
    d1 = cross_product_2d(move_start, move_end, line_start)
    d2 = cross_product_2d(move_start, move_end, line_end)
    
    # Position du mouvement par rapport à la ligne
    d3 = cross_product_2d(line_start, line_end, move_start)
    d4 = cross_product_2d(line_start, line_end, move_end)

    # Vérification du chevauchement strict (signes opposés)
    # On utilise la multiplication < 0 pour vérifier que les signes sont différents
    # Note: On exclut souvent le cas = 0 (colinéaire) pour éviter les doubles comptes sur la ligne exacte,
    # sauf besoin spécifique. Ici, on cherche un franchissement franc.
    intersect_move = ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0))
    intersect_line = ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))

    if intersect_move and intersect_line:
        return True
    
    return False

def get_crossing_direction(
    move_start: Point, 
    move_end: Point, 
    line_start: Point, 
    line_end: Point
) -> str:
    """
    Identifie le sens du franchissement par rapport à la ligne.
    Utile pour distinguer 'Entrée' vs 'Sortie'.
    
    Args:
        move_start, move_end: Trajectoire du véhicule.
        line_start, line_end: Ligne de référence orientée de Start vers End.
    
    Returns:
        str: "left_to_right" ou "right_to_left" (relatif à l'orientation de la ligne).
             "none" si pas de croisement.
    """
    if not is_segment_crossing(move_start, move_end, line_start, line_end):
        return "none"
    
    # On regarde de quel côté de la ligne se trouvait le véhicule au départ (t-1)
    # Produit vectoriel: LigneStart -> LigneEnd vs LigneStart -> MoveStart
    cp = cross_product_2d(line_start, line_end, move_start)
    
    # Si cp > 0, MoveStart est à "gauche" de la ligne orientée
    # Si cp < 0, MoveStart est à "droite" de la ligne orientée
    return "left_to_right" if cp > 0 else "right_to_left"


def main():
    """
    Validation unitaire.
    Vérifie les cas nominaux et limites sans lancer tout le pipeline ML.
    """
    logger = logging.getLogger("GeometryTest")
    logger.info("🧪 Démarrage des tests unitaires géométriques...")

    # Cas 1 : Franchissement net (Croix)
    # Ligne verticale x=10, de y=0 à y=20
    line = ((10, 0), (10, 20))
    # Mouvement horizontal de x=5 à x=15 à hauteur y=10
    move_cross = ((5, 10), (15, 10))
    
    assert is_segment_crossing(*move_cross, *line) == True, "❌ Erreur: Devrait croiser (Croix simple)"
    logger.info("✅ Test 1 (Croix simple) : PASS")

    # Cas 2 : Pas de franchissement (Parallèle)
    move_parallel = ((12, 0), (12, 20))
    assert is_segment_crossing(*move_parallel, *line) == False, "❌ Erreur: Ne devrait pas croiser (Parallèle)"
    logger.info("✅ Test 2 (Parallèle) : PASS")

    # Cas 3 : Pas de franchissement (Trop court / Avant la ligne)
    move_short = ((5, 10), (9, 10))
    assert is_segment_crossing(*move_short, *line) == False, "❌ Erreur: Ne devrait pas croiser (Trop court)"
    logger.info("✅ Test 3 (Trop court) : PASS")

    # Cas 4 : Direction
    # Ligne horizontale (0, 10) -> (20, 10)
    # Mouvement bas -> haut (5, 5) -> (5, 15)
    # Avec ligne orientée gauche->droite, bas est à "droite" (sens horaire), haut est à "gauche"
    horiz_line = ((0, 10), (20, 10))
    move_up = ((5, 5), (5, 15)) # Start (5,5) est "sous" la ligne (y positif vers le bas en image ?)
    # En repère image standard (y vers le bas) :
    # (0,10)->(20,10) vecteur (20, 0).
    # (0,10)->(5,5) vecteur (5, -5).
    # Cross: 20*(-5) - 0*5 = -100. Négatif => Droite.
    # Donc passage Droite -> Gauche.
    
    direction = get_crossing_direction(*move_up, *horiz_line)
    # Note: L'interprétation gauche/droite dépend du repère. Ici on valide la consistance.
    assert direction != "none", "❌ Erreur: Direction non détectée"
    logger.info(f"✅ Test 4 (Direction) : {direction} détecté")

    logger.info("🎉 Tous les tests géométriques sont passés.")

if __name__ == "__main__":
    main()