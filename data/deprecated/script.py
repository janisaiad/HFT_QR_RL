import requests
import pandas as pd
import os
import time
from typing import Dict, List, Optional, Any
from requests.exceptions import RequestException, HTTPError

# Créer le dossier databento s'il n'existe pas
os.makedirs("databento", exist_ok=True)

# Liste des symboles boursiers
symbols: List[str] = [
    "YMM", "APLT", "ERIC", "ANGO", "IE", 
    "UA", "RIOT", "ASAI", "CGAU", "PBI", 
    "CX", "ITUB", "HL", "DBI"
]

def download_databento_data(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Télécharge les données de Databento pour un symbole donné.
    
    Args:
        symbol (str): Le symbole boursier à télécharger.
    
    Returns:
        Optional[Dict[str, Any]]: Les données au format JSON si le téléchargement réussit, None sinon.
    """
    url: str = f"https://api.databento.com/v1/equities/{symbol}/basic"
    max_retries: int = 3
    retry_delay: int = 5
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except HTTPError as e:
            if e.response.status_code == 404:
                print(f"Le symbole {symbol} n'a pas été trouvé sur Databento.")
                return None
            print(f"Erreur HTTP lors du téléchargement des données pour {symbol}: {e}")
        except RequestException as e:
            print(f"Erreur lors du téléchargement des données pour {symbol}: {e}")
        
        if attempt < max_retries - 1:
            print(f"Nouvelle tentative dans {retry_delay} secondes...")
            time.sleep(retry_delay)
        else:
            print(f"Échec du téléchargement après {max_retries} tentatives.")
    return None

# Dictionnaire pour stocker les données
data: Dict[str, Optional[Dict[str, Any]]] = {}

# Télécharger les données pour chaque symbole
for symbol in symbols:
    data[symbol] = download_databento_data(symbol)
    if data[symbol] is None:
        print(f"Impossible de télécharger les données pour {symbol}. Passage au symbole suivant.")

# Filtrer les données non nulles et les convertir en DataFrame
valid_data: Dict[str, Dict[str, Any]] = {k: v for k, v in data.items() if v is not None}

if valid_data:
    df = pd.DataFrame.from_dict(valid_data, orient='index')
    df.to_csv("databento/databento_equities_basic.csv", index=True)
    print("Téléchargement terminé et données sauvegardées dans 'databento/databento_equities_basic.csv'.")
else:
    print("Aucune donnée valide n'a été téléchargée. Vérifiez les symboles et l'accès à l'API.")
