import os
import json
import pandas as pd
import plotly.graph_objects as go

# Fonction pour charger les données JSON
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Charger les fichiers JSON
condition = load_json('/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash2/data/dbn/condition.json')
manifest = load_json('/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash2/data/dbn/manifest.json')
metadata = load_json('/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash2/data/csv/metadata.json')

# Fonction pour charger les données CSV
def load_csv(stock):
    file_path = f'/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash2/data/csv/{stock}/20240624.csv'
    return pd.read_csv(file_path)

# Spécifier le stock
stock = 'ASAI'
data = load_csv(stock).sample(frac=0.03, random_state=1)

# Convertir ts_event en datetime
data['ts_event'] = pd.to_datetime(data['ts_event'], unit='ns')

# Créer une figure
fig = go.Figure()

# Tracer le prix moyen
fig.add_trace(go.Scatter(x=data['ts_event'], y=data['price'], mode='lines', name='Mid Price', line=dict(color='black')))

# Tracer la taille de la meilleure offre
fig.add_trace(go.Scatter(x=data['ts_event'], y=data['bid_px_00'], mode='markers',
                         marker=dict(size=data['bid_sz_00']**0.5 / 2, color='blue', opacity=0.5),
                         name='Best Bid Size'))

# Tracer la taille de la meilleure demande
fig.add_trace(go.Scatter(x=data['ts_event'], y=data['ask_px_00'], mode='markers',
                         marker=dict(size=data['ask_sz_00']**0.5 / 2, color='red', opacity=0.5),
                         name='Best Ask Size'))

# Mettre à jour la mise en page
fig.update_layout(title=f"Mid Price, Best Bid Size, and Best Ask Size for {stock}", xaxis_title="Time", yaxis_title="Value")

# Afficher le graphique
fig.show()
