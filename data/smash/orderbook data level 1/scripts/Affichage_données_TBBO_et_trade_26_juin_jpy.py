# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: EA
#     language: python
#     name: ea
# ---

# %% [markdown]
# # ÉTUDE RAPIDE DATA DU 26 JUIN 2024 ET VÉRIFICATION DE LA COHÉRENCE ENTRE LES DIFFÉRENTS DATAFRAMES

# %%
import os
import zstandard as zstd
import json
import databento as dbn  
import pandas as pd
import plotly.graph_objects as go
import numpy as np


# %% [markdown]
# On va étudier la journée du 24 juin 2024 et regarder dans un premier temps si les informations entre les différents datafrqmes sont cohérentes ou non.

# %%

def decompress_zst(zst_path, extract_path):
    """Décompresse un fichier zst comme un boss."""
    with open(zst_path, 'rb') as zst_file:
        dctx = zstd.ZstdDecompressor()
        with open(extract_path, 'wb') as out_file:
            dctx.copy_stream(zst_file, out_file)
    print(f"Bam ! Fichier décompressé dans : {extract_path}")

def load_metadata(metadata_path):
    """Charge les métadonnées du fichier JSON, fastoche."""
    with open(metadata_path, 'r') as f:
        return json.load(f)

def load_symbology(symbology_path):
    """Charge la symbologie du fichier JSON, rien de ouf."""
    with open(symbology_path, 'r') as f:
        return json.load(f)
    
    
    
def visualize_orderbook(dbn_file, symbol):
    """Visualise le carnet d'ordres pour un symbole donné en utilisant @databento."""
    try:
        # On crée un DBNStore à partir du chemin du fichier
        store = dbn.DBNStore.from_file(dbn_file)
        
        # On récupère les données pour le symbole spécifié
        df = store.to_df( # Assurez-vous que c'est le bon schéma pour vos données
        )
    except FileNotFoundError:
        print(f"Erreur : Le fichier {dbn_file} n'a pas été trouvé.")
        return
    except Exception as e:
        print(f"Une erreur s'est produite lors de la lecture du fichier : {str(e)}")
        return
    
    # On vérifie si des données ont été récupérées
    if df.empty:
        print(f"Aucune donnée n'a été trouvée pour le symbole {symbol}")
        return
    # On filtre les données pour ne garder que celles correspondant au symbole spécifié
    df = df[df['symbol'] == symbol]
    
    # On vérifie à nouveau si des données ont été trouvées après le filtrage
    if df.empty:
        print(f"Aucune donnée n'a été trouvée pour le symbole {symbol}")
        return
    # On extrait les données pertinentes
    
    df['timestamp'] = pd.to_datetime(df['ts_event'], unit='ns')
    
def main():
    
    # TBBO
    # Le chemin vers le dossier avec tous les trucs dedans
    folder_path = "data_TBBO"
    
    # On décompresse le fichier zst s'il y en a un
    zst_files = [f for f in os.listdir(folder_path) if f.endswith('.zst')]
    if zst_files:
        zst_path = os.path.join(folder_path, zst_files[0])
        extract_path = os.path.join(folder_path, zst_files[0][:-4])  # On enlève l'extension .zst
        decompress_zst(zst_path, extract_path)
    
    # On charge les métadonnées et la symbologie
    metadata = load_metadata(os.path.join(folder_path, "metadata.json"))
    symbology = load_symbology(os.path.join(folder_path, "symbology.json"))
    
    # On cherche le fichier DBN, ça devrait pas trop galère
    dbn_files = [f for f in os.listdir(folder_path) if f.endswith('.dbn')]
    if not dbn_files:
        print("Merde, pas de fichier DBN dans le dossier.")
        return
    
    dbn_file = os.path.join(folder_path, dbn_files[0])
    
    # On visualise le carnet d'ordres pour chaque symbole
    for symbol in symbology['symbols']:
        visualize_orderbook(dbn_file, symbol)
    
    
    # Pour les trades
    
    # Le chemin vers le dossier avec tous les trucs dedans
    folder_path = "data_Trade"
    
    # On décompresse le fichier zst s'il y en a un
    zst_files = [f for f in os.listdir(folder_path) if f.endswith('.zst')]
    if zst_files:
        zst_path = os.path.join(folder_path, zst_files[0])
        extract_path = os.path.join(folder_path, zst_files[0][:-4])  # On enlève l'extension .zst
        decompress_zst(zst_path, extract_path)
    
    # On charge les métadonnées et la symbologie
    metadata = load_metadata(os.path.join(folder_path, "metadata.json"))
    symbology = load_symbology(os.path.join(folder_path, "symbology.json"))
    
    # On cherche le fichier DBN, ça devrait pas trop galère
    dbn_files = [f for f in os.listdir(folder_path) if f.endswith('.dbn')]
    if not dbn_files:
        print("Merde, pas de fichier DBN dans le dossier.")
        return
    
    dbn_file = os.path.join(folder_path, dbn_files[0])
    
    # On visualise le carnet d'ordres pour chaque symbole
    for symbol in symbology['symbols']:
        visualize_orderbook(dbn_file, symbol)
        
    # Status
    
    # Le chemin vers le dossier avec tous les trucs dedans
    folder_path = "data_status"
    
    # On décompresse le fichier zst s'il y en a un
    zst_files = [f for f in os.listdir(folder_path) if f.endswith('.zst')]
    if zst_files:
        zst_path = os.path.join(folder_path, zst_files[0])
        extract_path = os.path.join(folder_path, zst_files[0][:-4])  # On enlève l'extension .zst
        decompress_zst(zst_path, extract_path)
    
    # On charge les métadonnées et la symbologie
    metadata = load_metadata(os.path.join(folder_path, "metadata.json"))
    symbology = load_symbology(os.path.join(folder_path, "symbology.json"))
    
    # On cherche le fichier DBN, ça devrait pas trop galère
    dbn_files = [f for f in os.listdir(folder_path) if f.endswith('.dbn')]
    if not dbn_files:
        print("Merde, pas de fichier DBN dans le dossier.")
        return
    
    dbn_file = os.path.join(folder_path, dbn_files[0])
    
    # On visualise le carnet d'ordres pour chaque symbole
    for symbol in symbology['symbols']:
        visualize_orderbook(dbn_file, symbol)

# main()

# %%
store_TBBO = dbn.DBNStore.from_file("data_TBBO/dbeq-basic-20240624.tbbo.dbn")
df_TBBO = store_TBBO.to_df().reset_index()
print(df_TBBO.shape)

# %%
df_TBBO.head()

# %%
store_trade = dbn.DBNStore.from_file("data_Trade/dbeq-basic-20240624.trades.dbn")
df_Trades = store_trade.to_df().reset_index()
print(df_Trades.shape)

# %%
df_Trades.head()

# %%
store_status = dbn.DBNStore.from_file("data_status/dbeq-basic-20240624.status.dbn")
df_status = store_status.to_df().reset_index()
print(df_status.shape)

# %%
df_status.head()

# %%
# on regarde les les trades sont les mêmes dans les deux df Trades et TBBO

colonnes_communes = df_TBBO.columns.intersection(df_Trades.columns)
print(colonnes_communes)
booleen = True
tab_difference = []
for colonnes in colonnes_communes:
    booleen = df_Trades[colonnes].equals(df_TBBO[colonnes])
    if not booleen:
        tab_difference.append(colonnes)

print('Les deux dataframes sont cohérents : ', len(tab_difference)==1)
print('Les différences sont dans les colonnes : ', tab_difference)

# %% [markdown]
# À partir de maintenant on va s'intérresser à certains actifs seulement qui sont des cas extrêmes dans les dataframes. On va essayer de vérifier la validité des informations données par le dataframe status.

# %%
actif_le_plus_frequent = df_status['symbol'].value_counts().idxmax()

# on filtres les dataframes en prenant
# l'actif qui a subit le plus de changements de status

df_trades_filtered = df_Trades[df_Trades['symbol']==actif_le_plus_frequent]
print(df_trades_filtered.shape)
df_TBBO_filtered = df_TBBO[df_TBBO['symbol']==actif_le_plus_frequent]
print(df_TBBO_filtered.shape)
df_status_filtered = df_status[df_status['symbol']==actif_le_plus_frequent]
print(df_status_filtered.shape)


# %%
df_trades_filtered.head()

# %%
df_TBBO_filtered.head()

# %%
df_status_filtered.head()

# %%
print(df_status_filtered.iloc[:,6])

# %% [markdown]
# Le status 0 correspond à aucun changement donc on devrait avoir des trades en quasi continu.

# %%
prices = df_trades_filtered['price'].to_numpy()
time = pd.to_datetime(df_trades_filtered['ts_event'].to_numpy())

fig = go.Figure()
fig.add_trace(go.Scatter(x = time, y = prices, mode ='lines', showlegend = False))
fig.update_layout(title=f'Évolution du prix de {actif_le_plus_frequent}', xaxis_title='Temps', yaxis_title='Prix', showlegend=True)
fig.show()

# %% [markdown]
# Pas très concluant mais on voit plus de trades sur la fin de la session. ON peut représenter le bid ask ssur les dates qui ont en un

# %%
prices = df_TBBO_filtered['price'].to_numpy()
time = pd.to_datetime(df_TBBO_filtered['ts_event'].to_numpy())

bid = df_TBBO_filtered['bid_px_00'].to_numpy()
ask = df_TBBO_filtered['ask_px_00'].to_numpy()

fig = go.Figure()
fig.add_trace(go.Scatter(x = time, y = prices, mode ='lines', name ='Prix', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = bid, mode ='lines', name ='Bid', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = ask, mode ='lines', name = 'Ask', showlegend = True))
fig.update_layout(title=f'Évolution du prix de {actif_le_plus_frequent} avec le bid-ask', xaxis_title='Temps', yaxis_title='Prix', showlegend=True)
fig.show()

# %% [markdown]
# Déjà le prix est bien compris dans le spread Bid-Ask. On voit cependant de grosses variations du spread. Probablement de la manipulation de marché...

# %%
spread = ask-bid

fig = go.Figure()
fig.add_trace(go.Scatter(x = time, y = spread, mode ='lines', name ='', showlegend = False))
fig.update_layout(title=f'Spread Bid-Ask de {actif_le_plus_frequent}', xaxis_title='Temps', yaxis_title='Prix', showlegend=True)
fig.show()

# %%
prices = df_TBBO_filtered['price'].to_numpy()
time = pd.to_datetime(df_TBBO_filtered['ts_event'])
bid = df_TBBO_filtered['bid_px_00'].to_numpy()
ask = df_TBBO_filtered['ask_px_00'].to_numpy()
ids = df_TBBO_filtered['publisher_id'].to_numpy()

# Créer une figure
fig = go.Figure()

unique_ids = np.unique(ids)
print('Il y a',(unique_ids), 'acteurs')
colors = {id_: f'rgb({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)})' for id_ in unique_ids}

for id_ in unique_ids:
    mask = ids == id_
    fig.add_trace(go.Scatter(
        x=time[mask],
        y=prices[mask],
        mode='markers',
        marker=dict(color=colors[id_]),
        showlegend=False
    ))
fig.update_layout(title=f'Trades de {actif_le_plus_frequent} par acheteurs', xaxis_title='Temps', yaxis_title='Prix', showlegend=True)
fig.show()

# %% [markdown]
# On va maintenant s'intéressé a des actifs qui ont subit des interruptions de marché.
#
# Différents status possibles:
# - 3 quoting but not trading
# - 8 trading halt
# - 9 trading paused
# - 10 suspended
# - 15 not available for trading
#
# 1 is not enough restrictive for now, we will not consider it
#
# Malheureusement aucun cas de ces arrêts n'est dans le dataframe

# %%
print('---Liste des états---\n', np.unique(df_status["action"].to_numpy()))

# %% [markdown]
# - 0 no change
# - 1 pre-open period
# - 3 the instrument is quoting but not trading.
# - 7 The instrument is trading.
# - 12 Trading in the instrument has closed.
# - 14 A change in short-selling restrictions.
#

# %% [markdown]
# Afin de vérifier la cohérence des dataframes, on regarde le 3 on devrait avoir un spread bid-ask mais pas de trades. La durée n'est pas indiquée???

# %%
states = [3]
actif_le_plus_frequent = df_status[df_status['action'].isin(states)]['symbol'].value_counts().idxmax()
#actif_le_plus_frequent = df_status['symbol'].value_counts().idxmax()

# on filtre les dataframes en prenant
# l'actif qui a subit le plus de changements de status

df_trades_filtered = df_Trades[df_Trades['symbol']==actif_le_plus_frequent]
#print(df_trades_filtered.shape)
df_TBBO_filtered = df_TBBO[df_TBBO['symbol']==actif_le_plus_frequent]
#print(df_TBBO_filtered.shape)
df_status_filtered = df_status[df_status['symbol']==actif_le_plus_frequent]
#print(df_status_filtered.shape)
prices = df_trades_filtered['price'].to_numpy()
time = pd.to_datetime(df_trades_filtered['ts_event'].to_numpy())

fig = go.Figure()
fig.add_trace(go.Scatter(x = time, y = prices, mode ='lines', showlegend = False))
fig.update_layout(title=f'Évolution du prix de {actif_le_plus_frequent}', xaxis_title='Temps', yaxis_title='Prix', showlegend=True)
fig.show()

prices = df_TBBO_filtered['price'].to_numpy()
time = pd.to_datetime(df_TBBO_filtered['ts_event'].to_numpy())

# date de l'event 
date_debut = '2024-06-24 10:30:00.001339990+00:00'
date_fin = '2024-06-24 10:30:00.001341861+00:00'

bid = df_TBBO_filtered['bid_px_00'].to_numpy()
ask = df_TBBO_filtered['ask_px_00'].to_numpy()

fig = go.Figure()
fig.add_trace(go.Scatter(x = time, y = prices, mode ='lines', name ='Prix', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = bid, mode ='lines', name ='Bid', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = ask, mode ='lines', name = 'Ask', showlegend = True))
fig.update_layout(title=f'Évolution du prix de {actif_le_plus_frequent} avec le bid-ask', xaxis_title='Temps', yaxis_title='Prix', showlegend=True)
fig.add_shape(type='rect',
    x0=date_debut, y0=0,
    x1=date_fin, y1=df_TBBO_filtered['price'].max(),
    fillcolor='red',
    opacity=0.5,
    layer='below',
    line_width=0,
)
fig.show()

prices = df_TBBO_filtered['price'].to_numpy()
time = pd.to_datetime(df_TBBO_filtered['ts_event'])
bid = df_TBBO_filtered['bid_px_00'].to_numpy()
ask = df_TBBO_filtered['ask_px_00'].to_numpy()
ids = df_TBBO_filtered['publisher_id'].to_numpy()

fig = go.Figure()

unique_ids = np.unique(ids)
print('Il y a',(unique_ids), 'acteurs')
colors = {id_: f'rgb({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)})' for id_ in unique_ids}

for id_ in unique_ids:
    mask = ids == id_
    fig.add_trace(go.Scatter(
        x=time[mask],
        y=prices[mask],
        mode='markers', 
        marker=dict(color=colors[id_]),
        showlegend=False
    ))
fig.update_layout(title=f'Trades de {actif_le_plus_frequent} par acheteurs', xaxis_title='Temps', yaxis_title='Prix', showlegend=True)
fig.show()

# %% [markdown]
# L'event a eu lieu hors-ouverture??

# %%
df_status_filtered[df_status_filtered['action']==3]

# %% [markdown]
# On va maintenant regarder les news (raisons)

# %%
print('---Liste des news---\n', np.unique(df_status["reason"].to_numpy()))

# %% [markdown]
# Pas de news intéressantes...
