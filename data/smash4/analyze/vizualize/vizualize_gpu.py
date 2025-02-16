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
import cudf
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
import os

# Fonction pour charger les données parquet
def load_parquet(stock, date):
    file_path = f'/home/janis/3A/EA/HFT_QR_RL/data/smash4/DB_MBP_10/{stock}/{stock}_{date}.parquet'
    return cudf.read_parquet(file_path)

# Spécifier les dates et stocks
dates = ["2024-08-12"]
stocks = ["CSX"]

for date in dates:
    for stock in tqdm(stocks, desc="Processing stocks"):
        # Charger et préparer les données
        df_gpu = load_parquet(stock, date)
        df_gpu = df_gpu.sample(frac=0.1, random_state=42)  # Downsampling initial
        df_gpu = df_gpu.sort_values('ts_event')
        
        # Calculate mid price
        df_gpu['mid_price'] = (df_gpu['bid_px_00'] + df_gpu['ask_px_00']) / 2
        
        # Downsampling supplémentaire si nécessaire
        sample_rate = max(1, len(df_gpu) // 20000)
        
        # Création de la figure
        fig = go.Figure()
        
        # Convertir les données GPU en numpy pour Plotly
        ts_event = df_gpu['ts_event'][::sample_rate].to_numpy()
        mid_price = df_gpu['mid_price'][::sample_rate].to_numpy()
        bid_px_00 = df_gpu['bid_px_00'][::sample_rate].to_numpy()
        ask_px_00 = df_gpu['ask_px_00'][::sample_rate].to_numpy()
        bid_sz_00 = df_gpu['bid_sz_00'][::sample_rate].to_numpy()
        ask_sz_00 = df_gpu['ask_sz_00'][::sample_rate].to_numpy()
        bid_px_01 = df_gpu['bid_px_01'][::sample_rate].to_numpy()
        ask_px_01 = df_gpu['ask_px_01'][::sample_rate].to_numpy()

        # Add mid price line
        fig.add_trace(go.Scatter(
            x=ts_event,
            y=mid_price,
            mode='lines',
            name='Mid Price',
            line=dict(color='black', width=1)
        ))

        # Add best bid
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

        # Add best ask
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

        # Add second best bid/ask
        fig.add_trace(go.Scatter(
            x=ts_event,
            y=bid_px_01,
            mode='lines',
            name='Second Best Bid',
            line=dict(color='rgba(0,255,0,0.3)', width=1)
        ))

        fig.add_trace(go.Scatter(
            x=ts_event,
            y=ask_px_01,
            mode='lines',
            name='Second Best Ask',
            line=dict(color='rgba(255,0,0,0.3)', width=1)
        ))

        # Add trades
        trades_gpu = df_gpu.query('rtype == 2')
        fig.add_trace(go.Scatter(
            x=trades_gpu['ts_event'].to_numpy(),
            y=trades_gpu['mid_price'].to_numpy(),
            mode='markers',
            name='Trades',
            marker=dict(color='black', size=8)
        ))

        # Update layout
        fig.update_layout(
            title=f"Order Book Visualization for {stock} on {date}",
            xaxis_title="Time",
            yaxis_title="Price",
            showlegend=True,
            width=1200,
            height=800,
            xaxis=dict(range=['13:00', '20:00']),
            yaxis=dict(range=[32, 36]),
            template="plotly_white",  # Template plus léger
            hovermode=False  # Désactive le hover pour performance
        )

        # Sauvegarde et affichage
        save_dir = f"/home/janis/3A/EA/HFT_QR_RL/data/smash4/plotly/{stock}"
        os.makedirs(save_dir, exist_ok=True)
        
        fig.write_html(
            f"{save_dir}/orderbook_visualization_{stock}_{date}.html",
            include_plotlyjs='cdn',
            full_html=False
        )
        
        fig.show()
        
        # Libérer la mémoire GPU
        del df_gpu, trades_gpu
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
# - 