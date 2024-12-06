import glob
import os
import polars as pl
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_imbalance(bid_size: float, ask_size: float) -> float:
    """Calcule l'imbalance selon la formule (bid_size - ask_size)/(bid_size + ask_size)"""
    if bid_size + ask_size == 0:
        return 0.0
    return (bid_size - ask_size) / (bid_size + ask_size)

def analyze_parquet_files(parquet_dir, stock_symbol, output_dir):
    """Analyse les fichiers parquet pour un stock donné"""
    parquet_files = glob.glob(os.path.join(parquet_dir, "*.parquet"))
    
    if not parquet_files:
        logger.warning(f"Aucun fichier parquet trouvé dans {parquet_dir}")
        return
    
    all_trades = []
    
    # Traitement de chaque fichier
    for file_path in tqdm(parquet_files, desc=f"Analyse de {stock_symbol}"):
        logger.info(f"Traitement du fichier: {os.path.basename(file_path)}")
        
        try:
            # Lecture du fichier parquet avec le schéma correct
            df = pl.read_parquet(
                file_path,
                columns=["ts_event", "action", "side", "size", "bid_sz_00", "ask_sz_00"]
            )
            
            # Conversion des timestamps
            df = df.with_columns([
                pl.col("ts_event").str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S%.f").alias("ts_event")
            ])
            
            # Filtrage des heures (15-18)
            df = df.filter(
                (pl.col("ts_event").dt.hour() >= 15) & 
                (pl.col("ts_event").dt.hour() <= 18)
            )
            
            # Calcul de l'imbalance
            df = df.with_columns([
                ((pl.col("bid_sz_00") - pl.col("ask_sz_00")) / 
                 (pl.col("bid_sz_00") + pl.col("ask_sz_00")))
                .round(3)
                .alias("imbalance")
            ])
            
            # Calcul du déséquilibre signé
            df = df.with_columns([
                pl.when(pl.col("side") == "A")
                .then(pl.col("imbalance"))
                .otherwise(-pl.col("imbalance"))
                .alias("imbalance_signed")
            ])
            
            # Filtrage des trades
            trades_df = df.filter(pl.col("action") == "T")
            
            if not trades_df.is_empty():
                all_trades.append(trades_df)
        
        except Exception as e:
            logger.error(f"Erreur lors du traitement de {file_path}: {str(e)}")
            continue
    
    if not all_trades:
        logger.warning(f"Aucun trade trouvé pour {stock_symbol}")
        return
    
    # Concaténation de tous les trades
    all_trades_df = pl.concat(all_trades)
    
    # Analyse des événements extrêmes
    extreme_ratio = 0.05
    window_size = 10
    
    # Création du graphique
    fig = go.Figure()
    
    # Conversion en timestamps pour le graphique
    timestamps = all_trades_df.get_column("ts_event").sort().to_numpy()
    
    if len(timestamps) > 0:
        # Calcul des périodes
        time_periods = np.linspace(
            timestamps[0].astype('datetime64[ns]'),
            timestamps[-1].astype('datetime64[ns]'),
            1000
        )
        
        # Calcul des probabilités
        probs = calculate_probabilities(timestamps, time_periods, window_size, extreme_ratio)
        
        # Ajout des traces
        fig.add_trace(go.Scatter(
            x=time_periods,
            y=probs,
            mode='lines',
            name='Événements extrêmes',
            line=dict(color='green', width=2)
        ))
        
        # Mise à jour du layout
        start_time = timestamps[0]
        fig.update_layout(
            title=f"Analyse des événements extrêmes - {stock_symbol}<br>Date: {pl.from_numpy(start_time).dt.strftime('%Y-%m-%d')}",
            xaxis_title="Heure",
            yaxis_title="Probabilité",
            showlegend=True
        )
        
        # Sauvegarde du graphique
        output_path = os.path.join(output_dir, f"{stock_symbol}_events_{pl.from_numpy(start_time).dt.strftime('%Y%m%d')}.png")
        fig.write_image(output_path)
        logger.info(f"Graphique sauvegardé : {output_path}")

def calculate_probabilities(timestamps, time_periods, window_size, extreme_ratio):
    """Calcule les probabilités d'événements extrêmes"""
    probs = []
    for i in range(window_size, len(time_periods)):
        end_time = time_periods[i]
        start_time = time_periods[i - window_size]
        events_in_window = np.sum((timestamps >= start_time) & (timestamps <= end_time))
        if events_in_window > 0:
            probs.append(events_in_window * extreme_ratio)
        else:
            probs.append(0)
    return probs

def main():
    """Fonction principale"""
    logger.info("Démarrage de l'analyse des événements")
    
    base_dir = "/home/janis/3A/EA/HFT_QR_RL/data"
    stocks = ['CSX', 'GOOGL', 'INTC', 'KHC', 'LCID', 'WBD']
    
    # Création du dossier de sortie
    output_dir = os.path.join(base_dir, "analysis4", "results")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Dossier de sortie créé: {output_dir}")
    
    # Analyse de chaque stock
    for stock in stocks:
        logger.info(f"\nDébut de l'analyse de {stock}...")
        parquet_dir = os.path.join(base_dir, "smash4", "parquet", stock)
        analyze_parquet_files(parquet_dir, stock, output_dir)

if __name__ == "__main__":
    main()