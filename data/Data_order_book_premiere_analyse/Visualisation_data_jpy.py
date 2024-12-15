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
#     name: python3
# ---

# %%
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


# %%
Trades = pd.read_csv("Data_csv/Trade.csv")
Tbbo = pd.read_csv("Data_csv/TBBO.csv")
Status = pd.read_csv("Data_csv/Status.csv")
print(Trades.shape,Tbbo.shape,Status.shape)

# %%
Trades.head()

# %%
actif_le_plus_frequent = Status['symbol'].value_counts().idxmax()

# on filtres les dataframes en prenant
# l'actif qui a subit le plus de changements de status

df_trades_filtered = Trades[Trades['symbol']==actif_le_plus_frequent]
print(df_trades_filtered.shape)
df_TBBO_filtered = Tbbo[Tbbo['symbol']==actif_le_plus_frequent]
print(df_TBBO_filtered.shape)
df_status_filtered = Status[Status['symbol']==actif_le_plus_frequent]
print(df_status_filtered.shape)

# %%
df_trades_filtered.head()

# %%
df_TBBO_filtered.head()

# %%
df_status_filtered.head()

# %%
prices = df_trades_filtered['price'].to_numpy()
df_trades_filtered['ts_event'] = pd.to_datetime(df_trades_filtered['ts_event'])
df_trades_filtered = df_trades_filtered.sort_values(by='ts_event')
time = df_trades_filtered['ts_event'].to_numpy()

fig = go.Figure()
fig.add_trace(go.Scatter(x = time, y = prices, mode ='lines', showlegend = False))
fig.update_layout(title=f'Évolution du prix de {actif_le_plus_frequent}', xaxis_title='Temps', yaxis_title='Prix', showlegend=True)
fig.show()

# %%
print('Les différents états du marché dans le dataframe des données sont :',np.unique(Status['action'].to_numpy()))

# %% [markdown]
# Pour rappel:
# - 0 no change
# - 1 The instrument is in a pre-open period
# - 3 quoting not trading
# - 7 The instrument is trading
# - 12 Trading in the instrument has closed
# - 14 A change in short-selling restrictions
#
# Là encore pas très interressant... On va regarder du côté des raisons.

# %%
print("Les différents raisons de changement d'état du marché du marché dans le dataframe des données sont :",np.unique(Status['reason'].to_numpy()))

# %% [markdown]
# Là encore pas du tout intéressant à part peut-être 60 : An operational issue occurred with the venue. On peut voir si cela à un impact.
#

# %%
df_transition = Status[Status['reason']==60]
actif_le_plus_frequent = df_transition['symbol'].value_counts().idxmax()
df_transition.head()

# %%
df_trades_filtered = Trades[Trades['symbol']==actif_le_plus_frequent]
print(df_trades_filtered.shape)
df_TBBO_filtered = Tbbo[Tbbo['symbol']==actif_le_plus_frequent]
print(df_TBBO_filtered.shape)
df_status_filtered = Status[Status['symbol']==actif_le_plus_frequent]
print(df_status_filtered.shape)

# %%
df_trades_filtered.head()

# %%
prices = df_trades_filtered['price'].to_numpy()
df_trades_filtered['ts_event'] = pd.to_datetime(df_trades_filtered['ts_event'])
df_trades_filtered = df_trades_filtered.sort_values(by='ts_event')
time = df_trades_filtered['ts_event'].to_numpy()

fig = go.Figure()
fig.add_trace(go.Scatter(x = time, y = prices, mode ='lines', showlegend = False))
fig.update_layout(title=f'Évolution du prix de {actif_le_plus_frequent}', xaxis_title='Temps', yaxis_title='Prix', showlegend=True)
fig.show()

# %%
print('Il y a ',np.count_nonzero(df_status_filtered['reason'].to_numpy()==60),'raisons 60')

# %%
df_filtre = df_status_filtered[df_status_filtered['reason']==60]
df_filtre = df_filtre[['ts_recv','reason']]
df_filtre['ts_recv'] = pd.to_datetime(df_filtre['ts_recv'])
df_grouped = df_filtre.groupby(pd.Grouper(key='ts_recv', freq='D')).agg(occurrences=('reason', 'size'))
df_grouped.head()
print('Le nombre max de code 60 en une journée est :',df_grouped['occurrences'].max())

# %% [markdown]
# ## Étude de la première limite

# %% [markdown]
# On va prendre un actif avec le plus de transactions

# %%
actif_le_plus_frequent = Tbbo['symbol'].value_counts().idxmax()

# on filtres les dataframes en prenant
# l'actif qui a le plus de trades

df_trades_filtered = Trades[Trades['symbol']==actif_le_plus_frequent]
df_TBBO_filtered = Tbbo[Tbbo['symbol']==actif_le_plus_frequent]
df_status_filtered = Status[Status['symbol']==actif_le_plus_frequent]

print("L'actif",actif_le_plus_frequent, 'a eu', df_trades_filtered.shape[0], "trades")

# %%
prices = df_trades_filtered['price'].to_numpy()
df_trades_filtered['ts_event'] = pd.to_datetime(df_trades_filtered['ts_event'])
df_trades_filtered = df_trades_filtered.sort_values(by='ts_event')
time = df_trades_filtered['ts_event'].to_numpy()

fig = go.Figure()
fig.add_trace(go.Scatter(x = time, y = prices, mode ='lines', showlegend = False))
fig.update_layout(title=f'Évolution du prix de {actif_le_plus_frequent}', xaxis_title='Temps', yaxis_title='Prix', showlegend=True)
fig.show()

# %%
df_TBBO_filtered['ts_event'] = pd.to_datetime(df_TBBO_filtered['ts_event'])
df_TBBO_filtered = df_TBBO_filtered.sort_values(by='ts_event')
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

# %%
df_TBBO_filtered['ts_event'] = pd.to_datetime(df_TBBO_filtered['ts_event'])
df_TBBO_filtered = df_TBBO_filtered.sort_values(by='ts_event')
df_TBBO_one_day = df_TBBO_filtered[df_TBBO_filtered['ts_event']<pd.to_datetime('2024-06-24 23:30:14.504507540+00:00')]
df_TBBO_filtered.head()

# %%
prices = df_TBBO_one_day['price'].to_numpy()
time = pd.to_datetime(df_TBBO_one_day['ts_event'].to_numpy())

bid = df_TBBO_one_day['bid_px_00'].to_numpy()
ask = df_TBBO_one_day['ask_px_00'].to_numpy()

fig = go.Figure()
fig.add_trace(go.Scatter(x = time, y = prices, mode ='lines', name ='Prix', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = bid, mode ='lines', name ='Bid', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = ask, mode ='lines', name = 'Ask', showlegend = True))
fig.update_layout(title=f'Évolution du prix de {actif_le_plus_frequent} avec le bid-ask sur une journée', xaxis_title='Temps', yaxis_title='Prix', showlegend=True)
fig.show()

# %% [markdown]
# Le bid ask est trop grand on va chercher un actif avec un bid ask ayant un spread plus petit

# %%
Tbbo['spread_bid_ask'] = Tbbo['bid_px_00']-Tbbo['ask_px_00']
df_grouped = Tbbo.groupby(pd.Grouper(key='symbol')).agg(variance_spread=('spread_bid_ask', 'var'))
df_grouped = df_grouped.reset_index()
df_grouped.head()

# %%
var_MIN = df_grouped['variance_spread'].min()
best_actif = df_grouped.loc[df_grouped['variance_spread'].idxmin(), 'symbol']
print("L'actif avec le petit spread de variance est", best_actif, 'avec une variance de', var_MIN)

# %%
df_trades_filtered = Trades[Trades['symbol']==best_actif]
df_TBBO_filtered = Tbbo[Tbbo['symbol']==best_actif]
df_status_filtered = Status[Status['symbol']==best_actif]

df_TBBO_filtered['ts_event'] = pd.to_datetime(df_TBBO_filtered['ts_event'])
df_TBBO_filtered = df_TBBO_filtered.sort_values(by='ts_event')
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

# %%
df_TBBO_filtered['ts_event'] = pd.to_datetime(df_TBBO_filtered['ts_event'])
df_TBBO_filtered = df_TBBO_filtered.sort_values(by='ts_event')
df_TBBO_one_day = df_TBBO_filtered[df_TBBO_filtered['ts_event']<pd.to_datetime('2024-06-24 23:30:14.504507540+00:00')]

prices = df_TBBO_one_day['price'].to_numpy()
time = pd.to_datetime(df_TBBO_one_day['ts_event'].to_numpy())

bid = df_TBBO_one_day['bid_px_00'].to_numpy()
ask = df_TBBO_one_day['ask_px_00'].to_numpy()

fig = go.Figure()
fig.add_trace(go.Scatter(x = time, y = prices, mode ='lines', name ='Prix', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = bid, mode ='lines', name ='Bid', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = ask, mode ='lines', name = 'Ask', showlegend = True))
fig.update_layout(title=f'Évolution du prix de {actif_le_plus_frequent} avec le bid-ask sur une journée', xaxis_title='Temps', yaxis_title='Prix', showlegend=True)
fig.show()

# %%
df_TBBO_filtered_bid_ask = df_TBBO_filtered.dropna(subset=['bid_px_00', 'ask_px_00'], how='any')
df_grouped = df_TBBO_filtered_bid_ask.groupby(pd.Grouper(key='symbol')).agg(variance_spread=('spread_bid_ask', 'var'))
df_TBBO_filtered_bid_ask.head()

# %%
df_TBBO_filtered_bid_ask.head()

# %% [markdown]
# On commence par étudier juste les trades

# %%
df_TBBO_filtered_trades = df_TBBO_filtered.groupby('size').agg(nombre=('size', 'size')).reset_index().sort_values(by=('size'))
df_TBBO_filtered_trades.head()

# %%
nombre = df_TBBO_filtered_trades['nombre'].to_numpy()
size = df_TBBO_filtered_trades['size'].to_numpy()
avrg_event_size = nombre*size/len(size)
size = size/avrg_event_size
fig = go.Figure()
fig.add_trace(go.Scatter(x = size, y = nombre, mode ='markers', showlegend = False))
fig.update_layout(title=f'Market by order of{best_actif} (not intensity)', xaxis_title='Size of the trade per average trade size', yaxis_title='Number', showlegend=True)
fig.show()

# %% [markdown]
# On représente le temps en continu sachant que les heures d'ouverture et de fermeture sont 13:30 et 20:00.

# %%
df_TBBO_filtered = df_TBBO_filtered.sort_values(by='ts_recv')
df_TBBO_filtered['ts_recv'] = pd.to_datetime(df_TBBO_filtered['ts_recv'])
df_TBBO_filtered['hour'] = df_TBBO_filtered['ts_recv'].dt.hour
df_TBBO_filtered['minute'] = df_TBBO_filtered['ts_recv'].dt.minute
df_filtered = df_TBBO_filtered[((df_TBBO_filtered['hour'] == 13) & (df_TBBO_filtered['minute'] >= 30))|((df_TBBO_filtered['hour'] > 13) & (df_TBBO_filtered['hour'] < 20))|(df_TBBO_filtered['hour'] == 20)]
df_filtered['time_diff'] = df_filtered['ts_recv'].diff().fillna(pd.Timedelta(seconds=0))
df_filtered.loc[df_filtered['time_diff'] > pd.Timedelta(hours=16.5), 'time_diff'] = pd.Timedelta(seconds=0)
df_filtered['hours_continuous'] = df_filtered['time_diff'].cumsum().dt.total_seconds()/3600
df_filtered.drop(columns=['hour', 'minute'], inplace=True)
df_filtered.head()


# %%
prices = df_filtered['price'].to_numpy()
time = df_filtered['hours_continuous'].to_numpy()

bid = df_filtered['bid_px_00'].to_numpy()
ask = df_filtered['ask_px_00'].to_numpy()

fig = go.Figure()
fig.add_trace(go.Scatter(x = time, y = prices, mode ='lines', name ='Prix', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = bid, mode ='lines', name ='Bid', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = ask, mode ='lines', name = 'Ask', showlegend = True))
fig.update_layout(title=f'Évolution du prix de {actif_le_plus_frequent} avec le bid-ask', xaxis_title='Temps', yaxis_title='Prix', showlegend=True)
fig.show()

# %% [markdown]
# On peut alors calculer l'intensité

# %%
nombre = df_TBBO_filtered_trades['nombre'].to_numpy()
size = df_TBBO_filtered_trades['size'].to_numpy()
#avrg_event_size = nombre*size/(len(size)*np.sum(nombre))
avrg_event_size = (nombre * size).sum() / nombre.sum()
size = size/avrg_event_size
nombre = nombre/df_filtered['hours_continuous'].to_numpy()[-1]/3600
fig = go.Figure()
fig.add_trace(go.Scatter(x = size, y = nombre, mode ='markers', showlegend = False))
fig.update_layout(title=f'Market by order of {best_actif}', xaxis_title='Size of the trade per average trade size', yaxis_title='Intensity??', showlegend=True)
fig.show()

# %% [markdown]
# ## Étude des queues size (première limite)

# %%
df_filtered.head()

# %%
sizes = df_filtered['ask_sz_00'].to_numpy()
tab = np.unique(sizes)
values = []
tab_utile = []
values_haut = []
values_bas = []
avr = np.sum(sizes)/len(sizes)
for i in range(len(tab)):
    df_tempo = df_filtered[df_filtered['ask_sz_00']==tab[i]]
    if len(df_tempo['size'].to_numpy())>100: # on veut qu'il y ait au min 100 trades
        tab_utile.append(tab[i])
        val = np.sum(df_tempo['size'].to_numpy())/(df_tempo['hours_continuous'].to_numpy()[-1]-df_tempo['hours_continuous'].to_numpy()[0])
        tps = df_tempo['hours_continuous'].to_numpy()
        dt = np.array([(tps[i+1]-tps[i]) for i in range(len(tps)-1)])
        var = np.std(dt)
        values_haut.append(val/3600+1.96*var/np.sqrt(len(dt))/60)
        values_bas.append(max(0,val/3600-1.96*var/np.sqrt(len(dt))/60))
        values.append(val/3600) #passage en secondes

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=tab_utile / avr,y=values_bas,mode='lines',line=dict(width=0),fillcolor='rgba(128, 128, 128, 0.3)',fill=None,showlegend=False))
fig.add_trace(go.Scatter(x=tab_utile / avr,y=values_haut,mode='lines',line=dict(color='gray'),fill='tonexty',fillcolor='rgba(128, 128, 128, 0.3)',name = 'Intervalle à 95%',showlegend=True))
fig.add_trace(go.Scatter(x=tab_utile / avr,y=values_haut,mode='lines',line=dict(color='gray'),showlegend=False))
fig.add_trace(go.Scatter(x=tab_utile / avr,y=values_bas,line=dict(color='gray'),mode='lines',showlegend=False))
fig.add_trace(go.Scatter(x = tab_utile/avr, y = values, mode ='markers+lines', line=dict(color='blue'), name = 'Estimation', showlegend = True))
fig.update_layout(title=f'Intensity of market by order of {best_actif}', xaxis_title='Queue size (per average queue size)', yaxis_title='Intensity', showlegend=True)
fig.show()


# %% [markdown]
# On peut ensuite tracer ce graphe sur les différents actifs du dataset de base

# %%
Symboles = np.unique(Tbbo['symbol'].to_numpy())
for j in range (len(Symboles)):
    df_TBBO_filtered = Tbbo[Tbbo['symbol']==Symboles[j]]
    df_TBBO_filtered = df_TBBO_filtered.sort_values(by='ts_recv')
    df_TBBO_filtered['ts_recv'] = pd.to_datetime(df_TBBO_filtered['ts_recv'])
    df_TBBO_filtered['hour'] = df_TBBO_filtered['ts_recv'].dt.hour
    df_TBBO_filtered['minute'] = df_TBBO_filtered['ts_recv'].dt.minute
    
    df_filtered = df_TBBO_filtered[((df_TBBO_filtered['hour'] == 13) & (df_TBBO_filtered['minute'] >= 30))|((df_TBBO_filtered['hour'] > 13) & (df_TBBO_filtered['hour'] < 20))|(df_TBBO_filtered['hour'] == 20)]
    df_filtered['time_diff'] = df_filtered['ts_recv'].diff().fillna(pd.Timedelta(seconds=0))
    df_filtered.loc[df_filtered['time_diff'] > pd.Timedelta(hours=16.5), 'time_diff'] = pd.Timedelta(seconds=0)
    df_filtered['hours_continuous'] = df_filtered['time_diff'].cumsum().dt.total_seconds()/3600
    df_filtered.drop(columns=['hour', 'minute'], inplace=True)
    
    sizes = df_filtered['ask_sz_00'].to_numpy()
    tab = np.unique(sizes)
    values = []
    tab_utile = []
    avr = np.sum(sizes)/len(sizes)
    
    for i in range(len(tab)):
        df_tempo = df_filtered[df_filtered['ask_sz_00']==tab[i]]
        if len(df_tempo['size'].to_numpy())>300: # on veut qu'il y ait au min 100 trades
            tab_utile.append(tab[i])
            val = np.sum(df_tempo['size'].to_numpy())/(df_tempo['hours_continuous'].to_numpy()[-1]-df_tempo['hours_continuous'].to_numpy()[0])
            tps = df_tempo['hours_continuous'].to_numpy()
            dt = np.array([(tps[i+1]-tps[i]) for i in range(len(tps)-1)])
            var = np.std(dt)
            values.append(val/3600) #passage en secondes

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = tab_utile/avr, y = values, mode ='markers+lines', name = Symboles[j], showlegend = True))
    fig.update_layout(title=f'Intensité des market by order de {Symboles[j]}', xaxis_title='Queue Size (per average queue size)', yaxis_title='Intensity', showlegend=True)
    fig.show()

# %% [markdown]
# On regarde la distribution du tirage lorsque l'on a eu queue de taille $n$. Afin d'avoir la meilleur estimation, on prend celle où l'on a le plus de valeurs.

# %%
best_actif = Tbbo['symbol'].value_counts().idxmax()
df_TBBO_filtered = Tbbo[Tbbo['symbol']==best_actif]
df_TBBO_filtered = df_TBBO_filtered.sort_values(by='ts_recv')
df_TBBO_filtered['ts_recv'] = pd.to_datetime(df_TBBO_filtered['ts_recv'])
df_TBBO_filtered['hour'] = df_TBBO_filtered['ts_recv'].dt.hour
df_TBBO_filtered['minute'] = df_TBBO_filtered['ts_recv'].dt.minute
    
df_filtered = df_TBBO_filtered[((df_TBBO_filtered['hour'] == 13) & (df_TBBO_filtered['minute'] >= 30))|((df_TBBO_filtered['hour'] > 13) & (df_TBBO_filtered['hour'] < 20))|(df_TBBO_filtered['hour'] == 20)]
df_filtered['time_diff'] = df_filtered['ts_recv'].diff().fillna(pd.Timedelta(seconds=0))
df_filtered.loc[df_filtered['time_diff'] > pd.Timedelta(hours=16.5), 'time_diff'] = pd.Timedelta(seconds=0)
df_filtered['hours_continuous'] = df_filtered['time_diff'].cumsum().dt.total_seconds()/3600
df_filtered.drop(columns=['hour', 'minute'], inplace=True)
    
sizes = df_filtered['ask_sz_00'].to_numpy()
tab = np.unique(sizes)
values = []
tab_utile = []
avr = np.sum(sizes)/len(sizes)
    
best_Queue_size = []
for i in range(len(tab)):
    df_tempo = df_filtered[df_filtered['ask_sz_00']==tab[i]]
    if len(df_tempo['size'].to_numpy())>300: # on veut qu'il y ait au min 100 trades
        tab_utile.append(tab[i])
        val = np.sum(df_tempo['size'].to_numpy())/(df_tempo['hours_continuous'].to_numpy()[-1]-df_tempo['hours_continuous'].to_numpy()[0])
        tps = df_tempo['hours_continuous'].to_numpy()
        dt  = np.array([(tps[i+1]-tps[i]) for i in range(len(tps)-1)])
        var = np.std(dt)
        values.append(val/3600) #passage en secondes
        best_Queue_size.append(len(df_tempo['size'].to_numpy()))

fig = go.Figure()
fig.add_trace(go.Scatter(x = tab_utile/avr, y = values, mode ='markers+lines', name = best_actif, showlegend = True))
fig.update_layout(title=f'Intensité des market by order de {best_actif}', xaxis_title='Queue Size (per average queue size)', yaxis_title='Intensity', showlegend=True)
fig.show()

# %%
best_size = tab_utile[np.argmax(best_Queue_size)]
df_best_size = df_filtered[df_filtered['ask_sz_00']==best_size]
df_TBBO_filtered_trades = df_best_size.groupby('size').agg(nombre=('size', 'size')).reset_index().sort_values(by=('size'))
df_TBBO_filtered_trades.head()

# %%

size = df_TBBO_filtered_trades['size'].to_numpy()
nombre = df_TBBO_filtered_trades['nombre'].to_numpy()/np.sum(df_TBBO_filtered_trades['nombre'].to_numpy())
fig = go.Figure()
fig.add_trace(go.Scatter(x = size, y = nombre, mode ='markers+lines', name = best_actif, showlegend = True))
fig.update_layout(title=f'Densité du Market order de {best_actif} lorsque la queue est de taille {best_size}', xaxis_title="Nombre d'actifs tradés", yaxis_title='Quantité', showlegend=True)
fig.show()

df_TBBO_filtered_trades = df_TBBO_filtered_trades[df_TBBO_filtered_trades["size"]<500]
size = df_TBBO_filtered_trades['size'].to_numpy()
nombre = df_TBBO_filtered_trades['nombre'].to_numpy()/np.sum(df_TBBO_filtered_trades['nombre'].to_numpy())
fig = go.Figure()
fig.add_trace(go.Scatter(x = size, y = nombre, mode ='markers+lines', name = best_actif, showlegend = True))
fig.update_layout(title=f'Densité du Market order de {best_actif} lorsque la queue est de taille {best_size} (tailles de paquets <500)', xaxis_title="Nombre d'actifs tradés", yaxis_title='Quantité', showlegend=True)
fig.show()

df_TBBO_filtered_trades = df_TBBO_filtered_trades[df_TBBO_filtered_trades["size"]<99]
size = df_TBBO_filtered_trades['size'].to_numpy()
nombre = df_TBBO_filtered_trades['nombre'].to_numpy()/np.sum(df_TBBO_filtered_trades['nombre'].to_numpy())
fig = go.Figure()
fig.add_trace(go.Scatter(x = size, y = nombre, mode ='markers+lines', name = best_actif, showlegend = True))
fig.update_layout(title=f'Densité du Market order de {best_actif} lorsque la queue est de taille {best_size} (tailles de paquets <100)', xaxis_title="Nombre d'actifs tradés", yaxis_title='Quantité', showlegend=True)
fig.show()

# %% [markdown]
# On voit bien que l'on a une intensité beaucoup plus grande pour les valeurs 10, 50, 100, 250, 500, ect... Cela est drastiquement différent d'une loi de poisson usuel et l'on peut donc rafiner le modèle en supposant une loi de poisson selon la taille de la queue et de la $\textbf{taille du paquet d'actif}$ acheté ou vendu.
#
# Le problème est alors d'avoir assez de données pour calibrer le modèle ($\sim 10^6$)
