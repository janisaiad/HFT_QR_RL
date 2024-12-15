# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# %%
import json
import pandas as pd
# %matplotlib notebook

# %%
# Fonction pour charger les données JSON
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Charger les fichiers JSON
condition = load_json('/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash2/data/dbn/condition.json')
manifest = load_json('/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash2/data/dbn/manifest.json')
metadata = load_json('/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash2/data/csv/metadata.json')

# Fonction pour charger les données CSV
def load_csv(stock, date):
    file_path = f'/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash2/data/csv/{stock}/{date}.csv'
    return pd.read_csv(file_path)

# Spécifier les dates et les stocks
dates = ["20240624", "20240625", "20240626", "20240627", "20240628", "20240701", "20240702", "20240703", "20240705", "20240708", "20240709", "20240710", "20240711", "20240712", "20240715", "20240716", "20240717", "20240718", "20240719", "20240722", "20240723", "20240724", "20240725", "20240726", "20240729", "20240730", "20240731", "20240801", "20240802", "20240805", "20240806", "20240807", "20240808"]
stocks = ["HL",'ASAI','RIOT','CGAU']
# Charger les données pour chaque stock et chaque date dans des datasets différents
data_dict = {}
for stock in stocks:
    data_dict[stock] = {}
    for date in dates:
        data_dict[stock][date] = load_csv(stock, date).sample(frac=0.1, random_state=1)
# Concaténer toutes les données
data_list = [data_dict[stock][date] for stock in stocks for date in dates]
data = pd.concat(data_list, ignore_index=True)

# Filtrer par publisher_id = 39
data = data[data['publisher_id'] == 39]

# Convertir ts_event en datetime
data['ts_event'] = pd.to_datetime(data['ts_event'], utc=True)
data = data.sort_values(by='ts_event')


# %%
from joblib import Parallel, delayed

# Calcul des intensités de Poisson pour chaque stock et chaque jour
def calculate_poisson_intensity(events, total_time):
    return len(events) / total_time

# Dictionnaire pour stocker les intensités
intensities = {stock: {} for stock in stocks}

def process_stock_date(stock, date):
    stock_data = data_dict[stock][date]
    
    # Calculer la durée totale de la journée en secondes
    start_time = pd.to_datetime(stock_data['ts_event'].min(), utc=True)
    end_time = pd.to_datetime(stock_data['ts_event'].max(), utc=True)
    total_time = (end_time - start_time).total_seconds()
    
    # Initialiser les dictionnaires pour stocker les événements
    cancel_events = {i: [] for i in range(1, 6)}  # Pour les 5 meilleurs niveaux
    add_events = {i: [] for i in range(1, 6)}
    market_order_events = []
    
    # Parcourir les données pour collecter les événements
    for _, row in stock_data.iterrows():
        action = row['action']
        side = row['side']
        depth = row['depth']
        
        if action == 'Cancel' and side == 'Bid' and depth <= 5:  # Cancel bid
            cancel_events[depth].append(row['ts_event'])
        elif action == 'Add' and side == 'Bid' and depth <= 5:  # Add bid
            add_events[depth].append(row['ts_event'])
        elif action == 'Trade' and side == 'Ask':  # Trade ask (market order)
            market_order_events.append(row['ts_event'])
    
    # Calculer les intensités pour chaque type d'événement
    intensities[stock][date] = {
        'cancel': {level: calculate_poisson_intensity(events, total_time) for level, events in cancel_events.items()},
        'add': {level: calculate_poisson_intensity(events, total_time) for level, events in add_events.items()},
        'market_order': calculate_poisson_intensity(market_order_events, total_time)
    }

# Utiliser joblib pour paralléliser le traitement
Parallel(n_jobs=-1)(delayed(process_stock_date)(stock, date) for stock in stocks for date in dates)

# Afficher un résumé des intensités calculées
for stock in stocks:
    print(f"\nIntensités pour {stock}:")
    for date in dates:
        print(f"  Date: {date}")
        print(f"    Annulations: {intensities[stock][date]['cancel']}")
        print(f"    Ajouts: {intensities[stock][date]['add']}")
        print(f"    Ordres de marché: {intensities[stock][date]['market_order']}")

# Fonction pour calculer l'intensité en fonction de la taille de la file d'attente
def queue_size_intensity(stock_data, action, side='Bid'):
    queue_sizes = stock_data[stock_data['side'] == side]['size']
    events = stock_data[(stock_data['action'] == action) & (stock_data['side'] == side)]
    
    # Vérifier si les événements ne sont pas vides
    if events.empty:
        return pd.Series(dtype=float)
    
    # Grouper les événements par taille de file d'attente
    grouped_events = events.groupby(pd.cut(events['size'], bins=10))
    
    # Calculer l'intensité pour chaque groupe
    intensities = grouped_events.size() / len(stock_data)
    return intensities

# Calculer les intensités en fonction de la taille de la file d'attente pour chaque stock et date
queue_size_intensities = {stock: {} for stock in stocks}

def process_queue_size_intensity(stock, date):
    stock_data = data_dict[stock][date]
    queue_size_intensities[stock][date] = {
        'cancel': queue_size_intensity(stock_data, 'Cancel'),
        'add': queue_size_intensity(stock_data, 'Add'),
        'market_order': queue_size_intensity(stock_data, 'Trade', 'Ask')
    }

# Utiliser joblib pour paralléliser le traitement
Parallel(n_jobs=-1)(delayed(process_queue_size_intensity)(stock, date) for stock in stocks for date in dates)

# Afficher un résumé des intensités en fonction de la taille de la file d'attente
for stock in stocks:
    print(f"\nIntensités en fonction de la taille de la file d'attente pour {stock}:")
    for date in dates:
        print(f"  Date: {date}")
        print(f"    Annulations: {queue_size_intensities[stock][date]['cancel']}")
        print(f"    Ajouts: {queue_size_intensities[stock][date]['add']}")
        print(f"    Ordres de marché: {queue_size_intensities[stock][date]['market_order']}")


# %%
