import polars as pl
import numpy as np
from tqdm import tqdm
import glob
import os
import matplotlib.pyplot as plt
import logging
from datetime import datetime

def analyze_sliding_windows(parquet_folder: str, txt_folder: str, window_size: int = 20):
    """
    Analyse les intensités par fenêtre glissante entre les actions seuillées.
    
    Args:
        parquet_folder: Dossier contenant les fichiers parquet
        txt_folder: Dossier contenant les fichiers txt des points seuillés
        window_size: Taille de la fenêtre glissante
    """
    # Configuration du logging
    log_dir = "/home/janis/3A/EA/HFT_QR_RL/data/likelihood/logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, f"sliding_windows_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    plot_output_dir = "/home/janis/3A/EA/HFT_QR_RL/data/likelihood/png"
    os.makedirs(plot_output_dir, exist_ok=True)
    
    # Parcourir les fichiers txt
    txt_files = glob.glob(os.path.join(txt_folder, "*normalized_intensities_thresholded.txt"))
    
    for txt_file in tqdm(txt_files):
        # Extraire les informations du nom du fichier
        filename = os.path.basename(txt_file)
        stock, date, spread_type = filename.split('_')[:3]
        spread_type = spread_type.replace("_normalized_intensities_thresholded.txt", "")
        
        logging.info(f"Traitement de {filename}")
        
        # Trouver le fichier parquet correspondant
        parquet_file = glob.glob(os.path.join(parquet_folder, f"{date}*PL.parquet"))[0]
        df = pl.read_parquet(parquet_file)
        
        # Convertir ts_event en timestamp nanoseconds
        df = df.with_columns([
            pl.col('ts_event').str.strptime(pl.Datetime).dt.timestamp().mul(1e9).alias('ts_event')
        ])
        
        # Filtrer par spread_type
        if spread_type == "zero_spread":
            df = df.filter(pl.col("diff_price") == 0)
        else:
            df = df.filter(pl.col("diff_price") != 0)
            
        # Lire les timestamps et valeurs seuillés du fichier txt
        thresholded_points = {}
        current_action = None
        
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('Action:'):
                    current_action = line.strip().split(':')[1].strip()
                    if current_action not in thresholded_points:
                        thresholded_points[current_action] = {'timestamps': [], 'values': []}
                elif line.strip() and current_action and ',' in line:
                    try:
                        timestamp, value = line.strip().split(',')
                        thresholded_points[current_action]['timestamps'].append(float(timestamp))
                        thresholded_points[current_action]['values'].append(float(value))
                    except ValueError:
                        continue
        
        # Pour chaque type d'action
        for action in thresholded_points:
            action_df = df.filter(pl.col('action') == action)
            timestamps = thresholded_points[action]['timestamps']
            values = thresholded_points[action]['values']
            
            if len(timestamps) < window_size:
                logging.warning(f"Pas assez de points pour {action}")
                continue
                
            # Trier les timestamps
            sorted_indices = np.argsort(timestamps)
            timestamps = [timestamps[i] for i in sorted_indices]
            values = [values[i] for i in sorted_indices]
            
            # Calculer les intensités entre les points seuillés
            intensities = []
            for i in range(len(timestamps)-1):
                count = len(action_df.filter(
                    (pl.col('ts_event') > timestamps[i]) &
                    (pl.col('ts_event') < timestamps[i+1])
                ))
                
                if count > 0:
                    intensities.append(1.0 / count)
                else:
                    intensities.append(0.0)
            
            # Calculer la moyenne glissante
            if len(intensities) >= window_size:
                sliding_means = []
                for i in range(len(intensities) - window_size + 1):
                    window_mean = np.mean(intensities[i:i+window_size])
                    sliding_means.append(window_mean)
                
                # Créer une figure avec deux sous-graphiques
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Premier graphique: intensités moyennes glissantes
                ax1.plot(range(len(sliding_means)), sliding_means, label=f'Fenêtre de {window_size}')
                ax1.set_title(f'Intensités moyennes glissantes - {action}\n{stock} {date} {spread_type}')
                ax1.set_xlabel('Index de la fenêtre')
                ax1.set_ylabel('Intensité moyenne')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Second graphique: exp(-x) des valeurs seuillées
                exp_values = np.exp(-np.array(values))
                ax2.plot(range(len(values)), exp_values, 'r-', label='exp(-x)')
                ax2.set_title(f'exp(-x) des valeurs seuillées - {action}')
                ax2.set_xlabel('Index')
                ax2.set_ylabel('exp(-x)')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                plt.tight_layout()
                
                # Sauvegarder le graphique
                plot_path = os.path.join(
                    plot_output_dir,
                    f"{stock}_{date}_{spread_type}_{action}_analysis.png"
                )
                plt.savefig(plot_path)
                plt.close()
                
                logging.info(f"Graphique sauvegardé: {plot_path}")

if __name__ == "__main__":
    parquet_folder = "/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ/LCID_filtered"
    txt_folder = "/home/janis/3A/EA/HFT_QR_RL/data/likelihood/txt"
    analyze_sliding_windows(parquet_folder, txt_folder)
