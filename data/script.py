import requests
import pandas as pd
import os

# Créer le dossier databento s'il n'existe pas
os.makedirs("databento", exist_ok=True)

# Liste des symboles boursiers
symbols = [
    "YMM", "APLT", "ERIC", "ANGO", "IE", 
    "UA", "RIOT", "ASAI", "CGAU", "PBI", 
    "CX", "ITUB", "HL", "DBI"
]

# Fonction pour télécharger les données de Databento
def download_databento_data(symbol):
    url = f"https://api.databento.com/v1/equities/{symbol}/basic"  # Remplacez par l'URL correcte de l'API
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()  # Retourne les données au format JSON
    else:
        print(f"Erreur lors du téléchargement des données pour {symbol}: {response.status_code}")
        return None

# Dictionnaire pour stocker les données
data = {}

# Télécharger les données pour chaque symbole
for symbol in symbols:
    data[symbol] = download_databento_data(symbol)

# Convertir les données en DataFrame et sauvegarder dans un fichier CSV dans le dossier databento
df = pd.DataFrame(data)
df.to_csv("databento/databento_equities_basic.csv", index=False)

print("Téléchargement terminé et données sauvegardées dans 'databento/databento_equities_basic.csv'.")
