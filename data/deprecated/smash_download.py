
import requests
import os
import zipfile
from typing import Optional

def telecharger_donnees_smash(url: str, chemin_destination: str) -> bool:
    """
    Télécharge les données depuis un lien Smash et les sauvegarde localement.

    Args:
        url (str): L'URL du lien Smash.
        chemin_destination (str): Le chemin où sauvegarder le fichier téléchargé.

    Returns:
        bool: True si le téléchargement a réussi, False sinon.
    """
    try:
        # Effectuer la requête GET pour obtenir les informations du fichier
        reponse = requests.get(url)
        reponse.raise_for_status()

        # Extraire l'URL de téléchargement direct
        donnees_json = reponse.json()
        url_telechargement: Optional[str] = donnees_json.get("download", {}).get("url")

        if not url_telechargement:
            print("L'URL de téléchargement n'a pas été trouvée dans la réponse.")
            return False

        # Télécharger le fichier
        with requests.get(url_telechargement, stream=True) as r:
            r.raise_for_status()
            with open(chemin_destination, "wb") as f:
                for morceau in r.iter_content(chunk_size=8192):
                    f.write(morceau)

        print(f"Téléchargement terminé. Fichier sauvegardé à : {chemin_destination}")
        return True

    except requests.RequestException as e:
        print(f"Erreur lors de la requête HTTP : {e}")
    except IOError as e:
        print(f"Erreur lors de l'écriture du fichier : {e}")
    except Exception as e:
        print(f"Une erreur inattendue est survenue : {e}")
    
    return False

def dezipper_fichier(chemin_zip: str, dossier_extraction: str) -> bool:
    """
    Dézippe un fichier ZIP dans un dossier spécifié.

    Args:
        chemin_zip (str): Le chemin du fichier ZIP à extraire.
        dossier_extraction (str): Le dossier où extraire les fichiers.

    Returns:
        bool: True si l'extraction a réussi, False sinon.
    """
    try:
        with zipfile.ZipFile(chemin_zip, "r") as zip_ref:
            zip_ref.extractall(dossier_extraction)
        print(f"Extraction terminée dans : {dossier_extraction}")
        return True
    except zipfile.BadZipFile:
        print(f"Le fichier {chemin_zip} n'est pas un fichier ZIP valide.")
    except IOError as e:
        print(f"Erreur lors de l'extraction : {e}")
    return False

def renommer_dossier(ancien_nom: str, nouveau_nom: str) -> bool:
    """
    Renomme un dossier.

    Args:
        ancien_nom (str): Le nom actuel du dossier.
        nouveau_nom (str): Le nouveau nom du dossier.

    Returns:
        bool: True si le renommage a réussi, False sinon.
    """
    try:
        os.rename(ancien_nom, nouveau_nom)
        print(f"Dossier renommé de {ancien_nom} à {nouveau_nom}")
        return True
    except OSError as e:
        print(f"Erreur lors du renommage du dossier : {e}")
        return False

# URL du lien Smash fourni
url_smash = "https://fromsmash.com/ATuA9urSUG-ct?e=amFuaXMuYWlhZEBwb2x5dGVjaG5pcXVlLmVkdQ%3D%3D"

# Créer le dossier de destination s'il n'existe pas
dossier_destination = "databento"
os.makedirs(dossier_destination, exist_ok=True)

# Chemin de destination pour le fichier téléchargé
chemin_fichier = os.path.join(dossier_destination, "donnees_telechargees.zip")

# Appeler la fonction pour télécharger les données
if telecharger_donnees_smash(url_smash, chemin_fichier):
    # Dézipper le fichier téléchargé
    dossier_extraction = os.path.join(dossier_destination, "extraction_temp")
    if dezipper_fichier(chemin_fichier, dossier_extraction):
        # Renommer le dossier extrait
        nouveau_nom_dossier = os.path.join(dossier_destination, "data_smash")
        if renommer_dossier(dossier_extraction, nouveau_nom_dossier):
            # Supprimer le fichier ZIP après extraction
            try:
                os.remove(chemin_fichier)
                print(f"Fichier ZIP supprimé : {chemin_fichier}")
            except OSError as e:
                print(f"Erreur lors de la suppression du fichier ZIP : {e}")
        else:
            print("Le renommage du dossier a échoué. Le fichier ZIP n'a pas été supprimé.")
    else:
        print("L'extraction a échoué. Le fichier ZIP n'a pas été supprimé.")
else:
    print("Le téléchargement a échoué. Aucune autre action n'a été effectuée.")
