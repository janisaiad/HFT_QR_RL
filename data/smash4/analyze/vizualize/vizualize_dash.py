# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# +
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import cudf
import cupy as cp
import numpy as np
from dash.exceptions import PreventUpdate
import os

# Configuration de la mémoire GPU
mempool = cp.get_default_memory_pool()
mempool.set_limit(size=0.8 * 1024**3)  # 80% de la VRAM

# Fonction helper pour conversion sûre en numpy
def safe_to_numpy(series, sample_rate):
    return series[::sample_rate].fillna(0).to_numpy()

# Chargement des données
def load_parquet(stock, date):
    file_path = f'/home/janis/3A/EA/HFT_QR_RL/data/smash4/DB_MBP_10/{stock}/{stock}_{date}.parquet'
    return cudf.read_parquet(file_path)

# Création de l'app Dash
app = dash.Dash(__name__)

# Layout de l'application
app.layout = html.Div([
    html.H1("Order Book Visualization"),
    
    # Contrôles
    html.Div([
        dcc.Dropdown(
            id='stock-selector',
            options=[{'label': 'CSX', 'value': 'CSX'}],
            value='CSX'
        ),
        dcc.Dropdown(
            id='date-selector',
            options=[{'label': '2024-08-12', 'value': '2024-08-12'}],
            value='2024-08-12'
        ),
        dcc.Slider(
            id='sample-slider',
            min=1000,
            max=50000,
            value=20000,
            marks={i: str(i) for i in range(1000, 50001, 10000)},
            step=1000
        ),
    ], style={'width': '50%', 'margin': '20px'}),
    
    # Graph
    dcc.Graph(id='orderbook-graph'),
    
    # Interval pour mise à jour automatique
    dcc.Interval(
        id='interval-component',
        interval=30*1000,  # en millisecondes
        n_intervals=0
    ),
    
    # Store pour les données
    dcc.Store(id='data-store')
])

# Callback pour charger les données
@app.callback(
    Output('data-store', 'data'),
    [Input('stock-selector', 'value'),
     Input('date-selector', 'value')]
)
def load_data(stock, date):
    if not stock or not date:
        raise PreventUpdate
    
    # Charger les données sur GPU
    df_gpu = load_parquet(stock, date)
    df_gpu = df_gpu.sample(frac=0.1, random_state=42)
    df_gpu = df_gpu.sort_values('ts_event')
    
    # Calculer mid price
    df_gpu['mid_price'] = (df_gpu['bid_px_00'] + df_gpu['ask_px_00']) / 2
    
    return {
        'stock': stock,
        'date': date,
        'len': len(df_gpu)
    }

# Callback pour mettre à jour le graphique
@app.callback(
    Output('orderbook-graph', 'figure'),
    [Input('data-store', 'data'),
     Input('sample-slider', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_graph(data, n_points, n_intervals):
    if not data:
        raise PreventUpdate
    
    # Recharger les données
    stock = data['stock']
    date = data['date']
    df_gpu = load_parquet(stock, date)
    
    # Downsampling
    sample_rate = max(1, len(df_gpu) // n_points)
    
    # Convertir les données GPU en numpy
    ts_event = safe_to_numpy(df_gpu['ts_event'], sample_rate)
    mid_price = safe_to_numpy(df_gpu['mid_price'], sample_rate)
    bid_px_00 = safe_to_numpy(df_gpu['bid_px_00'], sample_rate)
    ask_px_00 = safe_to_numpy(df_gpu['ask_px_00'], sample_rate)
    bid_sz_00 = safe_to_numpy(df_gpu['bid_sz_00'], sample_rate)
    ask_sz_00 = safe_to_numpy(df_gpu['ask_sz_00'], sample_rate)
    
    # Créer la figure
    fig = go.Figure()
    
    # Mid price
    fig.add_trace(go.Scatter(
        x=ts_event,
        y=mid_price,
        mode='lines',
        name='Mid Price',
        line=dict(color='black', width=1)
    ))
    
    # Best bid
    fig.add_trace(go.Scatter(
        x=ts_event,
        y=bid_px_00,
        mode='lines+markers',
        name='Best Bid',
        line=dict(color='green', width=1),
        marker=dict(
            size=bid_sz_00,
            sizeref=2.*float(bid_sz_00.max())/17**2,
            sizemode='area',
            color='green',
            opacity=0.3
        )
    ))
    
    # Best ask
    fig.add_trace(go.Scatter(
        x=ts_event,
        y=ask_px_00,
        mode='lines+markers',
        name='Best Ask',
        line=dict(color='red', width=1),
        marker=dict(
            size=ask_sz_00,
            sizeref=2.*float(ask_sz_00.max())/17**2,
            sizemode='area',
            color='red',
            opacity=0.3
        )
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Order Book for {stock} on {date}",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_white",
        hovermode=False,
        uirevision='constant'  # Maintient le zoom entre les updates
    )
    
    # Libérer la mémoire GPU
    del df_gpu
    cp.get_default_memory_pool().free_all_blocks()
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
