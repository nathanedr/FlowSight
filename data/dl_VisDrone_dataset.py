import kagglehub
import os
import shutil

# Télécharger et extraire le dataset UA-DETRAC depuis Kaggle
path = kagglehub.dataset_download("kushagrapandya/visdrone-dataset")

# Déplacer les fichiers extraits dans le répertoire souhaité
destination = "data/VisDrone-dataset"
if not os.path.exists(destination):
    os.makedirs(destination)
for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(destination, item)
    if os.path.isdir(s):
        shutil.move(s, d)
    else:
        shutil.move(s, d)
shutil.rmtree(path)
print(f"Dataset VisDrone téléchargé et extrait dans {destination}")

