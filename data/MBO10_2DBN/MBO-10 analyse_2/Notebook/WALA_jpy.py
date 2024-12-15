# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: IA_m1
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
import polars as pl

# %%
actif = 'GOOGL'
limite = 0


# %%
df = pl.read_parquet(f'/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ/{actif}/20240927.parquet')

# %%
df = df.filter(
    (pl.col('symbol') == actif) &
    (pl.col('depth') == limite) &
    (pl.col('side').is_in(['A','B']))
)

# No need to parse ts_event since it's already datetime[ns, UTC]
df = df.filter(
    (pl.col('ts_event').dt.hour() >= 14) & 
    (pl.col('ts_event').dt.hour() < 19)
)

# %%
# import pandas as pd

# # Assurez-vous que le DataFrame est trié par 'ts_event'
# df_sorted = df_.sort_values(by='ts_event').reset_index(drop=True)

# # Spécifiez le timestamp cible
# target_timestamp = pd.Timestamp('2024-09-27 14:00:00.707592949+00:00')

# # Localiser l'index de la ligne qui correspond au timestamp cible
# target_index = df_sorted[df_sorted['ts_event'] == target_timestamp].index

# # Initialiser une liste pour stocker les événements avant et après
# events_around_target = []

# # Si le timestamp cible existe dans le DataFrame, récupérez les événements avant et après
# if len(target_index) > 0:
#     target_index = target_index[0]  # Prendre le premier match si plusieurs existent

#     # Événement juste avant
#     if target_index > 0:
#         event_before = df_sorted.iloc[target_index - 1]
#         events_around_target.append(event_before)

#     # Événement juste après
#     if target_index < len(df_sorted) - 1:
#         event_after = df_sorted.iloc[target_index + 1]
#         events_around_target.append(event_after)

#     # Convertir les événements en DataFrame
#     df_events_around_target = pd.DataFrame(events_around_target)
#     print("Événements juste avant et juste après le timestamp cible:")
# else:
#     print("Le timestamp cible n'est pas présent dans le DataFrame.")


# %%
# import pandas as pd

# # Assurez-vous que le DataFrame est trié par 'ts_event'
# df_sorted = df.sort_values(by='ts_event').reset_index(drop=True)

# # Spécifiez le timestamp cible
# target_timestamp = pd.Timestamp('2024-09-27 14:00:00.707592949+00:00')

# # Filtrer pour obtenir l'événement immédiatement après le timestamp cible
# event_after = df_sorted[df_sorted['ts_event'] > target_timestamp].head(1)

# event_after.head()


# %%
# df_events_around_target.head()

# %%
# df[df['ts_event'] == pd.Timestamp('2024-09-27 14:00:00.707592949+00:00')].head(10)

# %%
def visu_1():
    # Get numpy arrays for plotting
    size_bid = df['bid_sz_00'].to_numpy()
    size_ask = df['ask_sz_00'].to_numpy()
    time = df['ts_event'].cast(pl.Datetime).to_numpy()

    # First plot - Queue sizes
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=size_bid, mode='lines', name='Bid', showlegend=True))
    fig.add_trace(go.Scatter(x=time, y=size_ask, mode='lines', name='Ask', showlegend=True))
    fig.update_layout(title='Taille de queues', xaxis_title='size', yaxis_title='intensity', showlegend=True)
    fig.show()

    # Get data for second plot
    trades_df = df.filter(pl.col('action') == 'T')
    price_1 = trades_df['price'].to_numpy() 
    time_price_1 = trades_df['ts_event'].cast(pl.Datetime).to_numpy()
    
    # Get price levels
    bid_ask_cols = ['bid_px_00', 'ask_px_00', 'bid_px_01', 'ask_px_01', 
                    'bid_px_02', 'ask_px_02', 'bid_px_03', 'ask_px_03']
    price_arrays = {col: df[col].to_numpy() for col in bid_ask_cols}

    # Second plot - Prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_price_1, y=price_1, mode='lines', name='Prix', showlegend=True))
    
    for i, (name, values) in enumerate(price_arrays.items()):
        side = 'Bid' if 'bid' in name else 'Ask'
        level = name[-2:]
        fig.add_trace(go.Scatter(x=time, y=values, mode='lines', 
                               name=f'{side} {level}', showlegend=True))
    
    fig.update_layout(title='Bid-ask', xaxis_title='size', yaxis_title='intensity', showlegend=True)
    fig.show()



# %%
visu_1()

# %%
df_ = df.select([
    'ts_event', 'action', 'side', 'size', 'price',
    f'bid_px_0{limite}', f'ask_px_0{limite}', 
    f'bid_sz_0{limite}', f'ask_sz_0{limite}',
    f'bid_ct_0{limite}', f'ask_ct_0{limite}',
    f'bid_px_0{limite+1}', f'ask_px_0{limite+1}', 
    f'bid_sz_0{limite+1}', f'ask_sz_0{limite+1}',
    f'bid_ct_0{limite+1}', f'ask_ct_0{limite+1}'
])

# %%

df_ = df_.filter(pl.col('bid_sz_00') > 1000)
df_ = df_.filter(pl.col('action') == 'T')
df_.head(100)

# %%
df_ = df_.with_columns([
    pl.col('ts_event').diff().dt.total_seconds().alias('time_diff')
])
df_.head(100)

# %%
# Check if required columns exist
required_columns = [
    'side', f'ask_px_0{limite}', f'bid_px_0{limite}', 
    f'ask_sz_0{limite}', f'bid_sz_0{limite}',
    f'ask_ct_0{limite}', f'bid_ct_0{limite}',
    'price', 'action', 'size'
]

for col in required_columns:
    if col not in df_.columns:
        print(f"Missing required column: {col}")

# First add all columns except those depending on diff_price
df_ = df_.with_columns([
    pl.when(pl.col('side') == 'A')
      .then(pl.col(f'ask_px_0{limite}'))
      .otherwise(pl.col(f'bid_px_0{limite}'))
      .alias('price_same'),
    pl.when(pl.col('side') == 'A')
      .then(pl.col(f'bid_px_0{limite}'))
      .otherwise(pl.col(f'ask_px_0{limite}'))
      .alias('price_opposite'),
    pl.when(pl.col('side') == 'A')
      .then(pl.col(f'ask_sz_0{limite}'))
      .otherwise(pl.col(f'bid_sz_0{limite}'))
      .alias('size_same'),
    pl.when(pl.col('side') == 'A')
      .then(pl.col(f'bid_sz_0{limite}'))
      .otherwise(pl.col(f'ask_sz_0{limite}'))
      .alias('size_opposite'),
    pl.when(pl.col('side') == 'A')
      .then(pl.col(f'ask_ct_0{limite}'))
      .otherwise(pl.col(f'bid_ct_0{limite}'))
      .alias('nb_ppl_same'),
    pl.when(pl.col('side') == 'A')
      .then(pl.col(f'bid_ct_0{limite}'))
      .otherwise(pl.col(f'ask_ct_0{limite}'))
      .alias('nb_ppl_opposite'),
    ((pl.col(f'ask_sz_0{limite}') - pl.col(f'bid_sz_0{limite}')) / 
     (pl.col(f'ask_sz_0{limite}') + pl.col(f'bid_sz_0{limite}'))).alias('imbalance'),
    pl.arange(0, pl.len()).alias('indice'),
    pl.col(f'bid_sz_0{limite}').diff().alias(f'bid_sz_0{limite}_diff'),
    pl.col(f'ask_sz_0{limite}').diff().alias(f'ask_sz_0{limite}_diff')
])

# Check if price column exists before adding diff_price
if 'price' not in df_.columns:
    print("Missing price column for diff_price calculation")
else:
    df_ = df_.with_columns([
        pl.col('price').diff().alias('diff_price')
    ])

# Check if diff_price exists before calculating Mean_price_diff  
if 'diff_price' not in df_.columns:
    print("Missing diff_price column for Mean_price_diff calculation")
else:
    df_ = df_.with_columns([
        pl.col('diff_price').rolling_mean(10).shift(1).alias('Mean_price_diff')
    ])

# Check columns needed for status calculation
status_columns = ['action', 'side', f'bid_sz_0{limite}_diff', f'ask_sz_0{limite}_diff', 'size']
if all(col in df_.columns for col in status_columns):
    df_ = df_.with_columns([
        pl.when(
            ((pl.col('action') == 'T') & 
             ((pl.col('side') == 'B') & (pl.col(f'bid_sz_0{limite}_diff') == -pl.col('size')) |
              (pl.col('side') == 'A') & (pl.col(f'ask_sz_0{limite}_diff') == -pl.col('size')))) |
            ((pl.col('action') == 'A') &
             ((pl.col('side') == 'B') & (pl.col(f'bid_sz_0{limite}_diff') == pl.col('size')) |
              (pl.col('side') == 'A') & (pl.col(f'ask_sz_0{limite}_diff') == pl.col('size')))) |
            ((pl.col('action') == 'C') &
             ((pl.col('side') == 'B') & (pl.col(f'bid_sz_0{limite}_diff') == -pl.col('size')) |
              (pl.col('side') == 'A') & (pl.col(f'ask_sz_0{limite}_diff') == -pl.col('size'))))
        ).then('OK').otherwise('NOK').alias('status')
    ])
else:
    print("Missing columns needed for status calculation")

# Check columns needed for new_limite and price_middle
if all(col in df_.columns for col in [f'bid_px_0{limite}', f'ask_px_0{limite}']):
    df_ = df_.with_columns([
        pl.when(
            (pl.col(f'bid_px_0{limite}').diff() > 0) | 
            (pl.col(f'ask_px_0{limite}').diff() > 0)
        ).then('new_limite').otherwise('n').alias('new_limite'),
        ((pl.col(f'ask_px_0{limite}') - pl.col(f'bid_px_0{limite}'))/2).alias('price_middle')
    ])
else:
    print("Missing columns needed for new_limite and price_middle calculations")

df_.head(1)


# %%
import pandas as pd

df_ = df_.reset_index(drop=True)

new_rows = []
lims = []
i = 0
total_rows = len(df_)
while i < total_rows-1:

    if i % (total_rows//10) == 0:
        print(f"Progression du code de merde: {int((i/total_rows)*100)}%")
    current_timestamp = df_.loc[i, 'ts_event']
    indices_group = [i]
    j = i+1
    
    while j < total_rows and df_.loc[j, 'ts_event'] == current_timestamp:
        indices_group.append(j)
        j += 1

    if len(indices_group) > 1:
        events_group = df_.iloc[indices_group]
        all_trades_then_cancel = all(events_group['action'].iloc[:-1] == 'T') and events_group['action'].iloc[-1] == 'C'
        all_trades = all(events_group['action'] == 'T')
        complex_trades_cancels = (
            events_group['action'].iloc[-1] == 'C' and
            all(
                events_group['action'].iloc[start:k].eq('T').all()
                for k, action in enumerate(events_group['action'])
                if action == 'C' and (start := events_group['action'].iloc[:k].last_valid_index()) is not None
            )
        )
        no_new_limite = 'new_limite' not in events_group['new_limite'].values
        
        complex_trades_cancels_not_ended_by_cancel = (
            all(
                events_group['action'].iloc[start:k].eq('T').all()
                for k, action in enumerate(events_group['action'])
                if action == 'C' and (start := events_group['action'].iloc[:k].last_valid_index()) is not None
            )
        )

        if all_trades_then_cancel and complex_trades_cancels and no_new_limite:
            total_size = events_group.loc[events_group['action'] == 'T', 'size'].sum()
            new_row = events_group.iloc[-1].copy()
            new_row['size'] = total_size
            new_row['action'] = 'T'
            new_rows.append(new_row)
            lims.extend((np.unique(events_group['new_limite'].to_numpy())))

        elif complex_trades_cancels and no_new_limite:
            total_size = events_group.loc[events_group['action'] == 'T', 'size'].sum()
            new_row = events_group.iloc[-1].copy()
            new_row['size'] = total_size
            new_row['action'] = 'T'
            new_rows.append(new_row)
            lims.extend((np.unique(events_group['new_limite'].to_numpy())))

        elif all_trades_then_cancel and not no_new_limite:
            limite_ = events_group['new_limite'].values
            bonne_limite = [0]+[l for l in range(len(limite_)) if limite_[l] == 'new_limite']
            for k in range(len(bonne_limite) - 1):
                start_index = bonne_limite[k]
                end_index = bonne_limite[k + 1]
                total_size = events_group[start_index:end_index].query("action == 'T'")['size'].sum()
                new_row = events_group.iloc[-1].copy()
                new_row['size'] = total_size
                new_row['action'] = 'T'
                new_row['new_limite'] = 'limite_épuisée'
                if new_row['side'] == 'B':
                    new_row[f'bid_sz_0{limite}'] = 0
                elif new_row['side'] == 'A':
                    new_row[f'ask_sz_0{limite}'] = 0
                new_rows.append(new_row)
            total_size = events_group.loc[bonne_limite[-1]:].query("action == 'T'")['size'].sum()
            new_row = events_group.iloc[-1].copy()
            new_row['size'] = total_size
            new_row['action'] = 'T'
            new_row['new_limite'] = 'new_limite'
            new_rows.append(new_row)

        elif complex_trades_cancels and not no_new_limite:
            limite_ = events_group['new_limite'].values
            bonne_limite = [0]+[l for l in range(len(limite_)) if limite_[l] == 'new_limite']
            for k in range(len(bonne_limite) - 1):
                start_index = bonne_limite[k]
                end_index = bonne_limite[k + 1]
                total_size = events_group[start_index:end_index].query("action == 'T'")['size'].sum()
                new_row = events_group.iloc[-1].copy()
                new_row['size'] = total_size
                new_row['action'] = 'T'
                new_row['new_limite'] = 'limite_épuisée'
                if new_row['side'] == 'B':
                    new_row[f'bid_sz_0{limite}'] = 0
                elif new_row['side'] == 'A':
                    new_row[f'ask_sz_0{limite}'] = 0
                new_rows.append(new_row)
            total_size = events_group.loc[bonne_limite[-1]:].query("action == 'T'")['size'].sum()
            new_row = events_group.iloc[-1].copy()
            new_row['size'] = total_size
            new_row['action'] = 'T'
            new_row['new_limite'] = 'new_limite'
            new_rows.append(new_row)

        elif all_trades:
            if df_.iloc[j]['new_limite'] == 'new_limite':
                total_size = events_group.loc[events_group['action'] == 'T', 'size'].sum()
                new_row = events_group.iloc[-1].copy()
                new_row['size'] = 0
                new_row['action'] = 'T'
                new_rows.append(new_row)
            else:
                total_size = events_group.loc[events_group['action'] == 'T', 'size'].sum()
                new_row = events_group.iloc[-1].copy()
                new_row['size'] = total_size
                new_row['action'] = 'T'
                new_rows.append(new_row)

        elif complex_trades_cancels_not_ended_by_cancel:
            if df_.iloc[j]['new_limite'] == 'new_limite':
                total_size = events_group.loc[events_group['action'] == 'T', 'size'].sum()
                new_row = events_group.iloc[-1].copy()
                new_row['size'] = 0
                new_row['action'] = 'T'
                new_rows.append(new_row)
            else:
                total_size = events_group.loc[events_group['action'] == 'T', 'size'].sum()
                new_row = events_group.iloc[-1].copy()
                new_row['size'] = total_size
                new_row['action'] = 'T'
                new_rows.append(new_row)

        else:
            new_rows.extend(events_group.to_dict(orient='records'))

        i = j
    else:
        new_rows.append(df_.iloc[i].to_dict())
        i += 1

print("Traitement terminé à 100% MGL")
# lims = np.array(lims).flatten()
# print(len(lims))
# print(lims)

# %%
standardized_series_list = []
for item in tqdm(new_rows):
    if isinstance(item, dict):
        standardized_series_list.append(pd.Series(item))
    elif isinstance(item, pd.Series):
        standardized_series_list.append(item)
    else:
        raise ValueError("L'élément de la liste n'est ni un dictionnaire ni une série.")

df__ = pd.concat(standardized_series_list, axis=1).T.reset_index(drop=True)


# %%
df__.head(2)

# %%
#df__['ts_diff'] = pd.to_datetime(df__['ts_event']).diff().dt.total_seconds()
df__['price_middle'] = (df__['ask_px_00']-df__['bid_px_00'])/2
df__['price_same'] = np.where(df__['side'] == 'A', df__[f'ask_px_0{limite}'],df__[f'bid_px_0{limite}'])
df__['price_opposite'] = np.where(df__['side'] == 'A', df__[f'bid_px_0{limite}'], df__[f'ask_px_0{limite}'])
df__['size_same'] = np.where(df__['side'] == 'A', df__[f'ask_sz_0{limite}'],df__[f'bid_sz_0{limite}'])
df__['size_opposite'] = np.where(df__['side'] == 'A', df__[f'bid_sz_0{limite}'], df__[f'ask_sz_0{limite}'])
df__['nb_ppl_same'] = np.where(df__['side'] == 'A', df__[f'ask_ct_0{limite}'],df__[f'bid_ct_0{limite}'])
df__['nb_ppl_opposite'] = np.where(df__['side'] == 'A', df__[f'bid_ct_0{limite}'], df__[f'ask_ct_0{limite}'])
#df__.drop(columns=[f'bid_px_0{limite}', f'ask_px_0{limite}', f'bid_sz_0{limite}', f'ask_sz_0{limite}', f'bid_ct_0{limite}', f'ask_ct_0{limite}'], axis=1, inplace=True)
df__['diff_price'] = df__['price_middle'].diff()
df__['time_diff'] = df__['ts_event'].diff().dt.total_seconds()



df__['indice'] = range(len(df__))
df__[f'bid_sz_0{limite}_diff'] = df__[f'bid_sz_0{limite}'].diff()
df__[f'ask_sz_0{limite}_diff'] = df__[f'ask_sz_0{limite}'].diff()
condition_T = (
    (df__['action'] == 'T') &
    (
        ((df__['side'] == 'B') & (df__[f'bid_sz_0{limite}_diff'] == -df__['size'])) |
        ((df__['side'] == 'A') & (df__[f'ask_sz_0{limite}_diff'] == -df__['size']))
    )
)

# Condition pour 'A'
condition_A = (
    (df__['action'] == 'A') &
    (
        ((df__['side'] == 'B') & (df__[f'bid_sz_0{limite}_diff'] == df__['size'])) |
        ((df__['side'] == 'A') & (df__[f'ask_sz_0{limite}_diff'] == df__['size']))
    )
)

# Condition pour 'C'
condition_C = (
    (df__['action'] == 'C') &
    (
        ((df__['side'] == 'B') & (df__[f'bid_sz_0{limite}_diff'] == -df__['size'])) |
        ((df__['side'] == 'A') & (df__[f'ask_sz_0{limite}_diff'] == -df__['size']))
    )
)
df__['status'] = np.where(condition_T | condition_A | condition_C, 'OK', 'NOK')
df__.loc[df__['new_limite'] == 'new_limite', 'time_diff'] = np.nan
df__ = df__[df__['time_diff']>0]
df__ = df__[df__['time_diff'] != np.nan]
df_final = df__[df__['status'] != 'NOK']
df__['Mean_price_diff'] = df__['diff_price'].rolling(window=50).mean().shift(1)
df__['imbalance'] = (df__[f'ask_sz_0{limite}']-df__[f'bid_sz_0{limite}'])/(df__[f'ask_sz_0{limite}']+df__[f'bid_sz_0{limite}'])
#df__['imbalance'] = (df__['size_same']-df__['size_opposite'])/(df__['size_same']+df__['size_opposite'])


df__.loc[df__['new_limite'] == 'new_limite', 'time_diff'] = np.nan
df__.head(1)

# %%
df_final = df__

# %%
# Vérifier s'il y a des NaN dans la colonne 'price'
has_nan_in_price = df__['Mean_price_diff'].isna().any()

if has_nan_in_price:
    print("La colonne 'price' contient des valeurs NaN.")
else:
    print("La colonne 'price' ne contient pas de valeurs NaN.")


# %%
df_trades = df_final[101:]#[df_final['action'] =='T']
imb = df_trades ['imbalance'].to_numpy()
price = df_trades ['Mean_price_diff'].to_numpy()#/df_trades ['time_diff'].to_numpy()
indices_trie = np.argsort(imb)

# Application du tri aux deux tableaux
imb_trie = imb[indices_trie]
price_trie = price[indices_trie]

group_size = 20000

imb_trie_groups = [imb_trie[i:i + group_size] for i in range(0, len(imb_trie), group_size)]
price_trie_groups = [price_trie[i:i + group_size] for i in range(0, len(price_trie), group_size)]


# Calculer la moyenne de chaque groupe
imb_trie_means = [np.mean(group) for group in imb_trie_groups]
price_trie_means = [np.mean(group) for group in price_trie_groups]
print(price_trie_means)
fig = go.Figure()
fig.add_trace(go.Scatter(x = imb_trie_means, y = price_trie_means, mode ='lines', name ='Prix', showlegend = True))

fig.update_layout(title=f'Bid-ask', xaxis_title='imbalance', yaxis_title='delta_price', showlegend=True)
fig.show()

# %%
df_trades = df_final[101:]#[df_final['action'] =='T']

# Calculer la moyenne de chaque groupe
imb_trie_means = [np.mean(group) for group in imb_trie_groups]
price_trie_means = [np.mean(group) for group in price_trie_groups]
print(price_trie_means)
fig = go.Figure()
fig.add_trace(go.Scatter(x = imb_trie_means, y = price_trie_means, mode ='lines', name ='Prix', showlegend = True))

fig.update_layout(title=f'Bid-ask', xaxis_title='imbalance', yaxis_title='delta_price', showlegend=True)
fig.show()

# %%
df___.head(4)

# %%
df_.head(4)

# %%
df_['price_same'] = np.where(df_['side'] == 'A', df_[f'ask_px_0{limite}'],df[f'bid_px_0{limite}'])
df_['price_opposite'] = np.where(df_['side'] == 'A', df[f'bid_px_0{limite}'], df_[f'ask_px_0{limite}'])
df_['size_same'] = np.where(df_['side'] == 'A', df_[f'ask_sz_0{limite}'],df[f'bid_sz_0{limite}'])
df_['size_opposite'] = np.where(df_['side'] == 'A', df[f'bid_sz_0{limite}'], df_[f'ask_sz_0{limite}'])
df_['nb_ppl_same'] = np.where(df_['side'] == 'A', df_[f'ask_ct_0{limite}'],df[f'bid_ct_0{limite}'])
df_['nb_ppl_opposite'] = np.where(df_['side'] == 'A', df[f'bid_ct_0{limite}'], df_[f'ask_ct_0{limite}'])
df_.drop(columns=[f'bid_px_0{limite}', f'ask_px_0{limite}', f'bid_sz_0{limite}', f'ask_sz_0{limite}', f'bid_ct_0{limite}', f'ask_ct_0{limite}'], axis=1, inplace=True)
df_['diff_price'] = df_['price'].diff()
df_['Mean_price_diff'] = df_['diff_price'].rolling(window=10).mean().shift(1)
df_['imbalance'] = (df_['size_same']-df_['size_opposite'])/(df_['size_same']+df_['size_opposite'])
df_['ts_event'] = pd.to_datetime(df_['ts_event'])
df_['time_diff'] = df_['ts_event'].diff().dt.total_seconds()
df_['indice'] = range(len(df_))

df_.head(2)

# %%
df_['trii'] = np.where(df_['time_diff'] == 0, df_['indice'] - 1, df_['indice'])
df_.head(20)

# %%
df_trades = df_[df_['action'] =='T']
imb = df_trades ['imbalance'].to_numpy()
price = df_trades ['Mean_price_diff'].to_numpy()#/df_trades ['time_diff'].to_numpy()
indices_trie = np.argsort(imb)

# Application du tri aux deux tableaux
imb_trie = imb[indices_trie]
price_trie = price[indices_trie]

fig = go.Figure()
fig.add_trace(go.Scatter(x = imb_trie, y = price_trie, mode ='lines', name ='Prix', showlegend = True))

fig.update_layout(title=f'Bid-ask', xaxis_title='imbalance', yaxis_title='delta_price', showlegend=True)
fig.show()

# %%
df_trades = df_[df_['action'] =='T']
price = df_trades['price']
bid_ask = 0.02
df_trades = df_trades[np.abs(df_trades['price_same'] - df_trades['price_opposite']) <= bid_ask]
sze_same = df_trades['size_same'].to_numpy()
sze_opposite = df_trades['size_opposite'].to_numpy()
time = df_trades['ts_event']

# %%

# %%


fig = go.Figure()
fig.add_trace(go.Scatter(x = time, y = sze_same, mode ='lines', name ='opposite', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = sze_opposite, mode ='lines', name ='same', showlegend = True))
fig.update_layout(title=f'Trades queue size', xaxis_title='time', yaxis_title='queue size', showlegend=True)
fig.show()
fig = go.Figure()
fig.add_trace(go.Scatter(x = time, y = sze_same-sze_opposite, mode ='lines', showlegend = True))
fig.update_layout(title=f'Trades queue size difference with bid-ask smaller than {bid_ask}', xaxis_title='time', yaxis_title='queue size difference', showlegend=True)
fig.show()

# %%
