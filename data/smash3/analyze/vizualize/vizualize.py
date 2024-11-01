import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# Charger les données pour chaque stock
stocks = ['ASAI', 'CGAU', 'HL', 'RIOT']
data = {stock: load_csv(stock).sample(frac=0.1, random_state=1) for stock in stocks}

# Créer une figure avec des sous-graphiques pour chaque stock
fig = make_subplots(rows=2, cols=2, subplot_titles=stocks)

for i, stock in enumerate(stocks):
    df = data[stock]
    row = i // 2 + 1
    col = i % 2 + 1

    # Convertir ts_event en datetime
    df['ts_event'] = pd.to_datetime(df['ts_event'], unit='ns')

    # Calculer le prix moyen
    df['mid_price'] = (df['bid_px_00'] + df['ask_px_00']) / 2

    # Tracer le prix moyen
    fig.add_trace(go.Scatter(x=df['ts_event'], y=df['mid_price'], mode='lines', name=f'{stock} Mid Price'),
                  row=row, col=col)

    # Tracer le prix de la meilleure offre
    fig.add_trace(go.Scatter(x=df['ts_event'], y=df['bid_px_00'], mode='lines', name=f'{stock} Best Bid Price'),
                  row=row, col=col)

    # Tracer le prix de la meilleure demande
    fig.add_trace(go.Scatter(x=df['ts_event'], y=df['ask_px_00'], mode='lines', name=f'{stock} Best Ask Price'),
                  row=row, col=col)

    # Tracer la taille de la meilleure offre
    fig.add_trace(go.Scatter(x=df['ts_event'], y=df['bid_sz_00'], mode='markers',
                             marker=dict(size=df['bid_sz_00']**0.5 / 10, color='blue', opacity=0.5),
                             name=f'{stock} Best Bid Size'), row=row, col=col)

    # Tracer la taille de la meilleure demande
    fig.add_trace(go.Scatter(x=df['ts_event'], y=df['ask_sz_00'], mode='markers',
                             marker=dict(size=df['ask_sz_00']**0.5 / 10, color='red', opacity=0.5),
                             name=f'{stock} Best Ask Size'), row=row, col=col)

    # Mise à jour des axes
    fig.update_xaxes(title_text="Time", row=row, col=col)
    fig.update_yaxes(title_text="Price", row=row, col=col)

# Mise à jour de la mise en page
fig.update_layout(height=1000, width=1200, title_text="MBP10 Visualization for Stocks")

# Afficher le graphique
fig.show()
