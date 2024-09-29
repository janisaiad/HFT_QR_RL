
import requests
import os
from typing import Optional

def telecharger_donnees_smash(url: str, chemin_destination: str) -> None:
    """
    Télécharge les données depuis un lien Smash et les sauvegarde localement.

    Args:
        url (str): L'URL du lien Smash.
        chemin_destination (str): Le chemin où sauvegarder le fichier téléchargé.

    Raises:
        Exception: Si une erreur survient pendant le téléchargement ou l'écriture du fichier.
    """
    try:
        # Effectuer la requête GET pour obtenir les informations du fichier
        reponse = requests.get(url)
        reponse.raise_for_status()

        # Extraire l'URL de téléchargement direct
        donnees_json = reponse.json()
        url_telechargement: Optional[str] = donnees_json.get('download', {}).get('url')

        if not url_telechargement:
            raise ValueError("L'URL de téléchargement n'a pas été trouvée dans la réponse.")

        # Télécharger le fichier
        with requests.get(url_telechargement, stream=True) as r:
            r.raise_for_status()
            with open(chemin_destination, 'wb') as f:
                for morceau in r.iter_content(chunk_size=8192):
                    f.write(morceau)

        print(f"Téléchargement terminé. Fichier sauvegardé à : {chemin_destination}")

    except requests.RequestException as e:
        print(f"Erreur lors de la requête HTTP : {e}")
    except IOError as e:
        print(f"Erreur lors de l'écriture du fichier : {e}")
    except Exception as e:
        print(f"Une erreur inattendue est survenue : {e}")

# URL du lien Smash fourni
url_smash = "https://fromsmash.com/ATuA9urSUG-ct?e=amFuaXMuYWlhZEBwb2x5dGVjaG5pcXVlLmVkdQ%3D%3D"

# Créer le dossier de destination s'il n'existe pas
dossier_destination = "databento"
os.makedirs(dossier_destination, exist_ok=True)

# Chemin de destination pour le fichier téléchargé
chemin_fichier = os.path.join(dossier_destination, "donnees_telechargees.zip")

# Appeler la fonction pour télécharger les données
telecharger_donnees_smash(url_smash, chemin_fichier)
