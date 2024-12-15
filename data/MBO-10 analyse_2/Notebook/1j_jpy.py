# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import pandas as pd # ca on va devoir bcp l'uiliser
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)

# %%
import databento as db

store = db.DBNStore.from_file('/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/dbn/NASDAQ/xnas-itch-20240927.mbp-10.dbn')
df = store.to_df(pretty_ts=True, pretty_px=True, map_symbols=True)

# df = df[df['symbol'] == 'GOOGL'] 
# df = df[df['depth'] == 0]
df.head(100)

# %%
size_bid = df['bid_sz_00'].to_numpy()
size_ask = df['ask_sz_00'].to_numpy()
df['ts_event'] = pd.to_datetime(df['ts_event'])
time = df['ts_event']

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x = time, y = size_bid, mode ='lines', name ='Bid', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = size_ask, mode ='lines', name = 'Ask', showlegend = True))
fig.update_layout(title='Taille de queues', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()

# %%
price = df[df['action'] =='T']['price'].to_numpy()
time_price = pd.to_datetime(df[df['action'] =='T']['ts_event'])
bid_px_00 = df['bid_px_00'].to_numpy()
ask_px_00 = df['ask_px_00'].to_numpy()
bid_px_01 = df['bid_px_01'].to_numpy()
ask_px_01 = df['ask_px_01'].to_numpy()
bid_px_02 = df['bid_px_02'].to_numpy()
ask_px_02 = df['ask_px_02'].to_numpy()
bid_px_03 = df['bid_px_03'].to_numpy()
ask_px_03 = df['ask_px_03'].to_numpy()
time = pd.to_datetime(df['ts_event'])
len(price)


# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x = time_price, y = price, mode ='lines', name ='Prix', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = bid_px_00, mode ='lines', name = 'Bid 0', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = ask_px_00, mode ='lines', name = 'Ask 0', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = bid_px_01, mode ='lines', name = 'Bid 1', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = ask_px_01, mode ='lines', name = 'Ask 1', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = bid_px_02, mode ='lines', name = 'Bid 2', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = ask_px_02, mode ='lines', name = 'Ask 2', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = bid_px_03, mode ='lines', name = 'Bid 3', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = ask_px_03, mode ='lines', name = 'Ask 3', showlegend = True))
fig.update_layout(title='Bid-ask', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()

# %%
df = df.replace(614, 0)
depth = 0
df = df[df['symbol'] == 'GOOGL']
df['ts_event'] = pd.to_datetime(df['ts_event'])
df[f'bid_sz_0{depth}_diff'] = df[f'bid_sz_0{depth}'].diff()
df[f'ask_sz_0{depth}_diff'] = df[f'ask_sz_0{depth}'].diff()

df = df[df['depth'] == depth]
#df['action'].value_counts()
# df_ = df[['action','side','size','bid_sz_00','ask_sz_00']]

# df_.head(1)


# %%

condition_T = (
    (df['action'] == 'T') &
    (
        ((df['side'] == 'B') & (df[f'bid_sz_0{depth}_diff'] == -df['size'])) |
        ((df['side'] == 'A') & (df[f'ask_sz_0{depth}_diff'] == -df['size']))
    )
)

# Condition pour 'A'
condition_A = (
    (df['action'] == 'A') &
    (
        ((df['side'] == 'B') & (df[f'bid_sz_0{depth}_diff'] == df['size'])) |
        ((df['side'] == 'A') & (df[f'ask_sz_0{depth}_diff'] == df['size']))
    )
)

# Condition pour 'C'
condition_C = (
    (df['action'] == 'C') &
    (
        ((df['side'] == 'B') & (df[f'bid_sz_0{depth}_diff'] == -df['size'])) |
        ((df['side'] == 'A') & (df[f'ask_sz_0{depth}_diff'] == -df['size']))
    )
)

# Appliquer 'OK' ou 'NOK' en fonction des conditions respectées
df['status'] = np.where(condition_T | condition_A | condition_C, 'OK', 'NOK')
#df = df['status' == 'OK']
df['status_N'] = np.where(condition_T | condition_A | condition_C, 3, 2)
df['status_diff'] = df['status_N'].diff()
df_ = df[['action','side','size','bid_sz_00','ask_sz_00','status']]#,'status_N','status_diff']]
df_.head(100)


# %%
df_.head(10)

# %%
df_['status'].value_counts()

# %%
df['size'].value_counts()

# %%
#df = df[df['depth'] == 0]
df['bid_sz_00_diff'] = df['bid_sz_00'].diff()
df['ask_sz_00_diff'] = df['ask_sz_00'].diff()
# MBO_filtered_depth_0_ = df[
#     ~(
#         (df['action'] == 'C')&
#         (
#             ((df['side'] == 'B')&(df['bid_sz_00_diff'] == 0)) |
#             ((df['side'] == 'A')&(df['ask_sz_00_diff'] == 0))
#         )
#     )
# ]

# df = df[
#     ~(
#         (df['action'] == 'A')&
#         (
#             ((df['side'] == 'B')&(df['bid_sz_00_diff'] != df['size'])) |
#             ((df['side'] == 'A')&(df['ask_sz_00_diff'] != df['size']))
#         )
#     )
# ]
# df = df[
#     ~(
#         (df['action'] == 'T')&
#         (
#             ((df['side'] == 'B')&(df['bid_sz_00_diff'] != -df['size'])) |
#             ((df['side'] == 'A')&(df['ask_sz_00_diff'] != -df['size']))
#         )
#     )
# ]

df.head(30)

# %%
import polars as pl

# Lire le fichier CSV avec Polars
df = pl.read_csv('/Users/edouard/Desktop/EA p1  HFT/HFT_QR_RL_save/Sans titre/HFT_QR_RL/data/MBO-10 analyse_2/CSV_dezippe_nasdaq/xnas-itch-20240927.mbp-10.csv')

# Ajouter une colonne d'index 'index_col'
df = df.with_columns(pl.arange(0, df.height).alias('index_col'))

# Filtrer le DataFrame pour ne garder que les lignes avec 'symbol' == 'GOOGL'
df = df.filter(pl.col('symbol') == 'GOOGL')

# Convertir 'ts_event' en datetime
df = df.with_columns(
    pl.col('ts_event').str.to_datetime().alias('ts_event')
)

# Calculer la différence des indices
df = df.with_columns(
    pl.col('index_col').diff().alias('index_diff')
)

# Calculer la différence de temps entre événements consécutifs
df = df.with_columns(
    pl.when(pl.col('index_diff') == 1)  # Si les indices sont consécutifs
    .then(pl.col('ts_event').diff().dt.nanoseconds() / 1e9)  # Calculer la différence en secondes
    .otherwise(0)  # Sinon, mettre 0
    .alias('temps_ecoule_secondes')
)

# Supprimer les lignes où l'intervalle de temps est 0
df = df.filter(pl.col('temps_ecoule_secondes') != 0)

# Afficher le DataFrame final
print(df)


# %%
df.replace(614, 0, inplace=True)
df['ts_event'] = pd.to_datetime(df['ts_event'])
df = df[df['side'].isin(['B','A'])]
df['temps_ecoule'] = df['ts_event'].diff()
df['temps_ecoule_secondes'] = df['temps_ecoule'].dt.total_seconds()

# df['ask_size'] = df['ask_sz_00']
# df['bid_size'] = df['ask_sz_00']
# df['bid_sz_00_shifted'] = df['bid_sz_00'].shift(1)
# df['bid_sz_shifted'].iloc[0] = 0

df = df[
    ~(
        (df['action'] == 'A')&
        (
            ((df['side'] == 'B')&(df['bid_sz_00_diff'] != df['size'])) |
            ((df['side'] == 'A')&(df['ask_sz_00_diff'] != df['size']))
        )
    )
]

df = df[
    ~(
        (df['action'] == 'T')&
        (
            ((df['side'] == 'B')&(df['bid_sz_00_diff'] != -df['size'])) |
            ((df['side'] == 'A')&(df['ask_sz_00_diff'] != -df['size']))
        )
    )
]

df = df[
    ~(
        (df['action'] == 'C')&
        (
            ((df['side'] == 'B')&(df['bid_sz_00_diff'] != -df['size'])) |
            ((df['side'] == 'A')&(df['ask_sz_00_diff'] != -df['size']))
        )
    )
]

#df.head(30)
#df = df[df['action'] == 'T']
df_filtered = df[['action','side','size','bid_sz_00','ask_sz_00']]
# # Initialiser la variable pour la ligne précédente à None
# previous_row = None

# # Itérer sur les lignes du DataFrame
# # Supposons que df_filtered est votre DataFrame original
# # Ajoutez des colonnes 'ask_size' et 'bid_size' initialisées à 0
# df_filtered['ask_size'] = 0
# df_filtered['bid_size'] = 0

# Initialiser la variable pour la ligne précédente à None
# previous_row = None

# # Itérer sur les lignes du DataFrame en utilisant iterrows()
# for index, row in tqdm(df_filtered.iterrows()):
#     if previous_row is not None:
#         # Vérifier la valeur de la colonne 'side'
#         if row['side'] == 'A':
#             # Mettre à jour 'ask_size' en fonction de l'action
#             if row['action'] == 'A':
#                 df_filtered.at[index, 'ask_size'] = previous_row['ask_size'] + row['size']
#             elif row['action'] == 'C':
#                 df_filtered.at[index, 'ask_size'] = previous_row['ask_size'] - row['size']
#             elif row['action'] == 'T':
#                 df_filtered.at[index, 'ask_size'] = previous_row['ask_size'] - row['size']
#             # Garder la même valeur pour 'bid_size'
#             df_filtered.at[index, 'bid_size'] = previous_row['bid_size']
            
#         elif row['side'] == 'B':
#             # Mettre à jour 'bid_size' en fonction de l'action
#             if row['action'] == 'A':
#                 df_filtered.at[index, 'bid_size'] = previous_row['bid_size'] + row['size']
#             elif row['action'] == 'C':
#                 df_filtered.at[index, 'bid_size'] = previous_row['bid_size'] - row['size']
#             elif row['action'] == 'T':
#                 df_filtered.at[index, 'bid_size'] = previous_row['bid_size'] - row['size']
#             # Garder la même valeur pour 'ask_size'
#             df_filtered.at[index, 'ask_size'] = previous_row['ask_size']
#     else:
#         # Si c'est la première ligne, initialiser 'ask_size' et 'bid_size' avec la valeur de 'size'
#         if row['side'] == 'A':
#             df_filtered.at[index, 'ask_size'] = row['size']
#         elif row['side'] == 'B':
#             df_filtered.at[index, 'bid_size'] = row['size']

#     # Mettre à jour la ligne précédente
#     previous_row = df_filtered.loc[index]
df_filtered.head(100)


# %%
df = pd.read_csv('/Users/edouard/Desktop/EA p1  HFT/HFT_QR_RL_save/Sans titre/HFT_QR_RL/data/MBO-10 analyse_2/CSV_dezippe_nasdaq/xnas-itch-20240927.mbp-10.csv')
df = df[df['symbol'] == 'GOOGL']

# %%
price = df[df['action'] =='T']['price'].to_numpy()
time_price = pd.to_datetime(df[df['action'] =='T']['ts_event'])
bid_px_00 = df['bid_px_00'].to_numpy()
ask_px_00 = df['ask_px_00'].to_numpy()
time = pd.to_datetime(df['ts_event'])
len(price)

