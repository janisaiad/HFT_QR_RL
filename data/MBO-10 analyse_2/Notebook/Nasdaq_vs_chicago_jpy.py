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
import pandas as pd # ca on va devoir bcp l'uiliser
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)

# %%
df = pd.read_csv('/Users/edouard/Desktop/EA p1  HFT/HFT_QR_RL_save/Sans titre/HFT_QR_RL/data/MBO-10 analyse_2/CSV_dezippe_nasdaq/xnas-itch-20240927.mbp-10.csv')
df.head()
df = df[df['symbol'] == 'LCID']
df = df[df['depth'] == 0]
size_bid = df['bid_sz_00'].to_numpy()
size_ask = df['ask_sz_00'].to_numpy()
df['ts_event'] = pd.to_datetime(df['ts_event'])
time = df['ts_event']

fig = go.Figure()
fig.add_trace(go.Scatter(x = time, y = size_bid, mode ='lines', name ='Bid', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = size_ask, mode ='lines', name = 'Ask', showlegend = True))
fig.update_layout(title='Taille de queues', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()

price_1 = df[df['action'] =='T']['price'].to_numpy()
time_price_1 = pd.to_datetime(df[df['action'] =='T']['ts_event'])
bid_px_00 = df['bid_px_00'].to_numpy()
ask_px_00 = df['ask_px_00'].to_numpy()
bid_px_01 = df['bid_px_01'].to_numpy()
ask_px_01 = df['ask_px_01'].to_numpy()
bid_px_02 = df['bid_px_02'].to_numpy()
ask_px_02 = df['ask_px_02'].to_numpy()
bid_px_03 = df['bid_px_03'].to_numpy()
ask_px_03 = df['ask_px_03'].to_numpy()
time = pd.to_datetime(df['ts_event'])
len(price_1)

fig = go.Figure()
fig.add_trace(go.Scatter(x = time_price_1, y = price_1, mode ='lines', name ='Prix', showlegend = True))
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
df = pd.read_csv('/Users/edouard/Desktop/EA p1  HFT/HFT_QR_RL_save/Sans titre/HFT_QR_RL/data/MBO-10 analyse_2/CSV_dezzipes_chicago/dbeq-basic-20240927.mbp-10.csv')
df.head()
df = df[df['publisher_id'] == 39]
df = df[df['symbol'] == 'LCID']
df = df[df['depth'] == 0]
size_bid = df['bid_sz_00'].to_numpy()
size_ask = df['ask_sz_00'].to_numpy()
df['ts_event'] = pd.to_datetime(df['ts_event'])
time = df['ts_event']

fig = go.Figure()
fig.add_trace(go.Scatter(x = time, y = size_bid, mode ='lines', name ='Bid', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = size_ask, mode ='lines', name = 'Ask', showlegend = True))
fig.update_layout(title='Taille de queues', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()

price2 = df[df['action'] =='T']['price'].to_numpy()
time_price2 = pd.to_datetime(df[df['action'] =='T']['ts_event'])
bid_px_00 = df['bid_px_00'].to_numpy()
ask_px_00 = df['ask_px_00'].to_numpy()
bid_px_01 = df['bid_px_01'].to_numpy()
ask_px_01 = df['ask_px_01'].to_numpy()
bid_px_02 = df['bid_px_02'].to_numpy()
ask_px_02 = df['ask_px_02'].to_numpy()
bid_px_03 = df['bid_px_03'].to_numpy()
ask_px_03 = df['ask_px_03'].to_numpy()
time = pd.to_datetime(df['ts_event'])

fig = go.Figure()
fig.add_trace(go.Scatter(x = time_price2, y = price2, mode ='lines', name ='Prix', showlegend = True))
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

fig = go.Figure()
fig.add_trace(go.Scatter(x = time_price_1, y = price_1, mode ='lines', name ='Prix_Nasdaq', showlegend = True))
fig.add_trace(go.Scatter(x = time_price2, y = price2, mode ='lines', name ='Prix_Chicago', showlegend = True))
fig.update_layout(title='Chicago vs Nasdaq', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()

# %%
df = pd.read_csv('/Users/edouard/Desktop/EA p1  HFT/HFT_QR_RL_save/Sans titre/HFT_QR_RL/data/MBO-10 analyse_2/CSV_dezzipes_chicago/dbeq-basic-20240927.mbp-10.csv')
df = df[df['depth'] == 0]
df = df[df['action'] =='T']
df = df[df['publisher_id'] == 41]
df.head()
df_bid = df[df['price'] == df['bid_px_00']]
df_ask = df[df['price'] == df['ask_px_00']]
time = pd.to_datetime(df_bid['ts_event'])

size_bid = df_bid['bid_sz_00']
size_ask = df_bid['ask_sz_00']

fig = go.Figure()
fig.add_trace(go.Scatter(x = time, y = size_bid , mode ='lines', name ='bid', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = size_ask , mode ='lines', name ='Ask', showlegend = True))
fig.update_layout(title='Trade du bid queue ask et bid', xaxis_title='size', yaxis_title='taille de queue', showlegend=True)
fig.show()


size_bid = df_ask['bid_sz_00']
size_ask = df_ask['ask_sz_00']

fig = go.Figure()
fig.add_trace(go.Scatter(x = time, y = size_bid , mode ='lines', name ='bid', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = size_ask , mode ='lines', name ='Ask', showlegend = True))
fig.update_layout(title='Trade du ask queue ask et bid', xaxis_title='size', yaxis_title='taille de queue', showlegend=True)
fig.show()

# %%
df = pd.read_csv('/Users/edouard/Desktop/EA p1  HFT/HFT_QR_RL_save/Sans titre/HFT_QR_RL/data/MBO-10 analyse_2/CSV_dezzipes_chicago/dbeq-basic-20240927.mbp-10.csv')
df = df[df['depth'] == 0]
df = df[df['action'] =='T']
df = df[df['publisher_id'] == 39]
df.head()
df_bid = df[df['price'] == df['bid_px_00']]
df_ask = df[df['price'] == df['ask_px_00']]
time = pd.to_datetime(df_bid['ts_event'])

size_bid = df_bid['bid_sz_00']
size_ask = df_bid['ask_sz_00']

fig = go.Figure()
fig.add_trace(go.Scatter(x = time, y = size_bid , mode ='lines', name ='bid', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = size_ask , mode ='lines', name ='Ask', showlegend = True))
fig.update_layout(title='Trade du bid queue ask et bid', xaxis_title='size', yaxis_title='taille de queue', showlegend=True)
fig.show()


size_bid = df_ask['bid_sz_00']
size_ask = df_ask['ask_sz_00']

fig = go.Figure()
fig.add_trace(go.Scatter(x = time, y = size_bid , mode ='lines', name ='bid', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = size_ask , mode ='lines', name ='Ask', showlegend = True))
fig.update_layout(title='Trade du ask queue ask et bid', xaxis_title='size', yaxis_title='taille de queue', showlegend=True)
fig.show()

# %%
df = pd.read_csv('/Users/edouard/Desktop/EA p1  HFT/HFT_QR_RL_save/Sans titre/HFT_QR_RL/data/MBO-10 analyse_2/CSV_dezzipes_chicago/dbeq-basic-20240927.mbp-10.csv')
df = df[df['depth'] == 0]
df = df[df['action'] =='T']
df = df[df['publisher_id'] == 40]
df.head()
df_bid = df[df['price'] == df['bid_px_00']]
df_ask = df[df['price'] == df['ask_px_00']]
time = pd.to_datetime(df_bid['ts_event'])

size_bid = df_bid['bid_sz_00']
size_ask = df_bid['ask_sz_00']

fig = go.Figure()
fig.add_trace(go.Scatter(x = time, y = size_bid , mode ='lines', name ='bid', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = size_ask , mode ='lines', name ='Ask', showlegend = True))
fig.update_layout(title='Trade du bid queue ask et bid', xaxis_title='size', yaxis_title='taille de queue', showlegend=True)
fig.show()


size_bid = df_ask['bid_sz_00']
size_ask = df_ask['ask_sz_00']

fig = go.Figure()
fig.add_trace(go.Scatter(x = time, y = size_bid , mode ='lines', name ='bid', showlegend = True))
fig.add_trace(go.Scatter(x = time, y = size_ask , mode ='lines', name ='Ask', showlegend = True))
fig.update_layout(title='Trade du ask queue ask et bid', xaxis_title='size', yaxis_title='taille de queue', showlegend=True)
fig.show()
