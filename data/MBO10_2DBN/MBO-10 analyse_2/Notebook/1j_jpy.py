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
import polars as pl
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)
from tqdm import tqdm
import os
import glob
from collections import Counter

# %%
df = pl.read_parquet('/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ/KHC/20240927.parquet')
df.head(20)

# %%
size_bid = df.get_column('bid_sz_00').to_numpy()
size_ask = df.get_column('ask_sz_00').to_numpy()
df = df.with_columns(pl.col('ts_event').cast(pl.Datetime))
time = df.get_column('ts_event')

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x = time, y = size_bid, mode ='lines', name ='Bid', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = size_ask, mode ='lines', name = f'Ask', showlegend = True))
fig.update_layout(title=f'Taille de queues', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()

# %%
price = df.filter(pl.col('action') == 'T').get_column('price').to_numpy()
time_price = df.filter(pl.col('action') == 'T').get_column('ts_event')
bid_px_00 = df.get_column('bid_px_00').to_numpy()
ask_px_00 = df.get_column('ask_px_00').to_numpy()
bid_px_01 = df.get_column('bid_px_01').to_numpy()
ask_px_01 = df.get_column('ask_px_01').to_numpy()
bid_px_02 = df.get_column('bid_px_02').to_numpy()
ask_px_02 = df.get_column('ask_px_02').to_numpy()
bid_px_03 = df.get_column('bid_px_03').to_numpy()
ask_px_03 = df.get_column('ask_px_03').to_numpy()
time = df.get_column('ts_event')
len(price)


# %%
# Take every 10th point to reduce data
step = 10
fig = go.Figure()
fig.add_trace(go.Scatter(x = time_price[::step], y = price[::step], mode ='lines', name ='Prix', showlegend = True))
fig.add_trace(go.Scatter(x = time[::step], y = bid_px_00[::step], mode ='lines', name = f'Bid 0', showlegend = True))
fig.add_trace(go.Scatter(x = time[::step], y = ask_px_00[::step], mode ='lines', name = f'Ask 0', showlegend = True))
fig.add_trace(go.Scatter(x = time[::step], y = bid_px_01[::step], mode ='lines', name = f'Bid 1', showlegend = True))
fig.add_trace(go.Scatter(x = time[::step], y = ask_px_01[::step], mode ='lines', name = f'Ask 1', showlegend = True))
fig.add_trace(go.Scatter(x = time[::step], y = bid_px_02[::step], mode ='lines', name = f'Bid 2', showlegend = True))
fig.add_trace(go.Scatter(x = time[::step], y = ask_px_02[::step], mode ='lines', name = f'Ask 2', showlegend = True))
fig.add_trace(go.Scatter(x = time[::step], y = bid_px_03[::step], mode ='lines', name = f'Bid 3', showlegend = True))
fig.add_trace(go.Scatter(x = time[::step], y = ask_px_03[::step], mode ='lines', name = f'Ask 3', showlegend = True))
fig.update_layout(title=f'Bid-ask', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()

# %%
depth = 0
df = df.with_columns(pl.col('bid_sz_00').replace(614, 0))
df = df.filter(pl.col('symbol') == 'GOOGL')
df = df.with_columns([
    pl.col(f'bid_sz_0{depth}').diff().alias(f'bid_sz_0{depth}_diff'),
    pl.col(f'ask_sz_0{depth}').diff().alias(f'ask_sz_0{depth}_diff')
])

df = df.filter(pl.col('depth') == depth)
#df.groupby('action').count()
# df = df.select(['action', 'side', 'size', 'bid_sz_00', 'ask_sz_00'])

# df.head(1)


# %%
# Ensure required columns exist before applying conditions
required_cols = ['action', 'side', 'size', f'bid_sz_0{depth}_diff', f'ask_sz_0{depth}_diff']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataframe")

# Cast size and diff columns to i64 to support negative values
df = df.with_columns([
    pl.col('size').cast(pl.Int64),
    pl.col(f'bid_sz_0{depth}_diff').cast(pl.Int64),
    pl.col(f'ask_sz_0{depth}_diff').cast(pl.Int64)
])

# Create conditions for each action type
condition_T = (
    (pl.col('action') == 'T') &
    (
        ((pl.col('side') == 'B') & (pl.col(f'bid_sz_0{depth}_diff') == -pl.col('size'))) |
        ((pl.col('side') == 'A') & (pl.col(f'ask_sz_0{depth}_diff') == -pl.col('size')))
    )
)

condition_A = (
    (pl.col('action') == 'A') &
    (
        ((pl.col('side') == 'B') & (pl.col(f'bid_sz_0{depth}_diff') == pl.col('size'))) |
        ((pl.col('side') == 'A') & (pl.col(f'ask_sz_0{depth}_diff') == pl.col('size')))
    )
)

condition_C = (
    (pl.col('action') == 'C') &
    (
        ((pl.col('side') == 'B') & (pl.col(f'bid_sz_0{depth}_diff') == -pl.col('size'))) |
        ((pl.col('side') == 'A') & (pl.col(f'ask_sz_0{depth}_diff') == -pl.col('size')))
    )
)

# First create the status columns
df = df.with_columns([
    pl.when(condition_T | condition_A | condition_C)
    .then(pl.lit('OK'))
    .otherwise(pl.lit('NOK'))
    .alias('status')
])

df = df.with_columns([
    pl.when(condition_T | condition_A | condition_C)
    .then(pl.lit(3))
    .otherwise(pl.lit(2))
    .alias('status_N')
])

# Then add the status difference column
df = df.with_columns([
    pl.col('status_N').diff().alias('status_diff')
])

# Select final columns
select_cols = ['ts_event', 'rtype', 'publisher_id', 'instrument_id', 'action', 'side', 'size', 'bid_sz_00', 'ask_sz_00', 'status']
df_ = df.select(select_cols)
df_.head(100)


# %%
df_.head(10)

# %%
df_['status'].value_counts()

# %%
df['size'].value_counts()

# %%
#df = df[df['depth'] == 0]
df = df.with_columns([
    pl.col('bid_sz_00').diff().alias('bid_sz_00_diff'),
    pl.col('ask_sz_00').diff().alias('ask_sz_00_diff')
])
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

# %% [markdown]
# polars clean upward
#

# %%

# %%

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

