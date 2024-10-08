import os
import json
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

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
data = load_csv(stock).sample(frac=0.1, random_state=1)

# Filtrer par publisher_id = 39
data = data[data['publisher_id'] == 39]

# Convertir ts_event en datetime
data['ts_event'] = pd.to_datetime(data['ts_event'], unit='ns')

# Créer une figure avec Plotly
fig = go.Figure()

# Couleurs pour les tailles de la meilleure offre et demande
bid_colors = ['red', 'darkred', 'brown', 'firebrick', 'maroon', 'darkorange', 'orange', 'gold', 'yellow', 'black']
ask_colors = ['blue', 'darkblue', 'navy', 'royalblue', 'mediumblue', 'dodgerblue', 'deepskyblue', 'skyblue', 'lightblue', 'black']

# Tracer les tailles des meilleures offres avec une courbe claire pour suivre la couleur de l'offre
for i in range(10):
    bid_px_col = f'bid_px_0{i}'
    bid_sz_col = f'bid_sz_0{i}'
    fig.add_trace(go.Scatter(x=data['ts_event'], y=data[bid_px_col], mode='markers',
                             line=dict(color=bid_colors[i], width=2),
                             marker=dict(size=data[bid_sz_col]**0.5 / 2, color=bid_colors[i], opacity=0.5),
                             name=f'Best Bid Size {i}'))

# Tracer les tailles des meilleures demandes avec une courbe claire pour suivre la couleur de la demande
for i in range(10):
    ask_px_col = f'ask_px_0{i}'
    ask_sz_col = f'ask_sz_0{i}'
    fig.add_trace(go.Scatter(x=data['ts_event'], y=data[ask_px_col], mode='markers',
                             line=dict(color=ask_colors[i], width=2),
                             marker=dict(size=data[ask_sz_col]**0.5 / 2, color=ask_colors[i], opacity=0.5),
                             name=f'Best Ask Size {i}'))
# Tracer les courbes pour le prix de bid 00 et ask 00 avec des points de dispersion croissants dans le temps
fig.add_trace(go.Scatter(x=data['ts_event'], y=data['bid_px_00'], mode='lines+markers', name='Bid Price 00', line=dict(color='green', width=2), marker=dict(size=5)))
fig.add_trace(go.Scatter(x=data['ts_event'], y=data['ask_px_00'], mode='lines+markers', name='Ask Price 00', line=dict(color='purple', width=2), marker=dict(size=5)))

# Mettre à jour la mise en page
fig.update_layout(title=f"Mid Price, Best Bid Size, and Best Ask Size for {stock}", xaxis_title="Time", yaxis_title="Value")

# Afficher le graphique
fig.show()

"""
# Créer une figure avec Matplotlib
plt.figure(figsize=(14, 7))

# Tracer le prix moyen
plt.plot(data['ts_event'], data['price'], label='Mid Price', color='black')

# Tracer la taille de la meilleure offre
plt.scatter(data['ts_event'], data['bid_px_00'], s=data['bid_sz_00']**0.5 / 2, c='blue', alpha=0.5, label='Best Bid Size')

# Tracer la taille de la meilleure demande
plt.scatter(data['ts_event'], data['ask_px_00'], s=data['ask_sz_00']**0.5 / 2, c='red', alpha=0.5, label='Best Ask Size')

# Mettre à jour la mise en page
plt.title(f"Mid Price, Best Bid Size, and Best Ask Size for {stock}")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()

# Afficher le graphique
plt.show()
"""
