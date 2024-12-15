import polars as pl
import numpy as np
from tqdm import tqdm
import glob
import os
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from dataframe_transform import transform_dataframe
import seaborn as sns

def analyze_normalized_intensities(folder_path: str, alpha_add: float = 0.98, alpha_cancel: float = 0.98, alpha_trade: float = 0.985):
    """
    Analyse les intensités normalisées par bucket d'imbalance avec normalisation préalable puis seuillage.
    
    Args:
        folder_path: Chemin vers le dossier contenant les fichiers parquet
        alpha_add: Seuil pour les événements d'ajout
        alpha_cancel: Seuil pour les événements d'annulation
        alpha_trade: Seuil pour les événements de trade
    """
    # Obtenir le symbole de l'action depuis le chemin du dossier
    stock = os.path.basename(folder_path).split('_')[0].upper()
    
    # Configuration du logging
    log_dir = "/home/janis/3A/EA/HFT_QR_RL/data/likelihood/logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, f"normalized_intensities_{stock}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

    files = glob.glob(os.path.join(folder_path, "*PL.parquet"))
    plot_output_dir = "/home/janis/3A/EA/HFT_QR_RL/data/likelihood/png"
    txt_output_dir = "/home/janis/3A/EA/HFT_QR_RL/data/likelihood/txt"
    os.makedirs(plot_output_dir, exist_ok=True)
    os.makedirs(txt_output_dir, exist_ok=True)
    
    # Dictionnaire pour stocker les proportions par jour
    daily_proportions = {
        'zero_spread': {'A': {}, 'C': {}, 'T': {}},
        'nonzero_spread': {'A': {}, 'C': {}, 'T': {}}
    }
    
    for file in tqdm(files):
        logging.info(f"Traitement du fichier: {file}")
        file_date = os.path.basename(file).split('_')[0]
        
        # Lecture et préparation des données
        df = pl.read_parquet(file)
        df = transform_dataframe(df)
        
        # Séparer les spreads zéro et non-zéro
        df_zero = df.filter(pl.col("diff_price") == 0)
        df_nonzero = df.filter(pl.col("diff_price") != 0)
        
        for spread_type, df_spread in [("zero_spread", df_zero), ("nonzero_spread", df_nonzero)]:
            # Création des buckets d'imbalance
            imbalance_bins = np.linspace(-1, 1, 21)  # 20 buckets
            
            # Types d'événements et leurs seuils
            event_configs = [
                ('A', 'add_deltas', alpha_add),
                ('C', 'cancel_deltas', alpha_cancel),
                ('T', 'trade_deltas', alpha_trade)
            ]
            
            # Fichier txt pour sauvegarder les points
            txt_file = os.path.join(txt_output_dir, 
                                  f"{stock}_{file_date}_{spread_type}_normalized_intensities_thresholded.txt")
            
            with open(txt_file, 'w') as f:
                for event_type, delta_col, alpha in event_configs:
                    event_df = df_spread.filter(pl.col('action') == event_type)
                    
                    # Initialiser le stockage pour ce jour
                    if file_date not in daily_proportions[spread_type][event_type]:
                        daily_proportions[spread_type][event_type][file_date] = {}
                    
                    f.write(f"Action: {event_type}\n")
                    
                    # Liste pour stocker tous les deltas normalisés
                    all_normalized_deltas = []
                    all_timestamps = []
                    all_bucket_labels = []
                    
                    # Première passe pour normaliser par bucket
                    for i in range(len(imbalance_bins)-1):
                        bucket_label = f"[{imbalance_bins[i]:.2f}, {imbalance_bins[i+1]:.2f})"
                        
                        bucket_df = event_df.filter(
                            (pl.col('imbalance') >= imbalance_bins[i]) & 
                            (pl.col('imbalance') < imbalance_bins[i+1])
                        )
                        
                        if len(bucket_df) > 0:
                            deltas = bucket_df[delta_col].to_numpy()
                            mean_delta = np.mean(deltas)
                            normalized_deltas = deltas / mean_delta
                            
                            all_normalized_deltas.extend(normalized_deltas)
                            all_timestamps.extend(bucket_df['ts_event'].to_list())
                            all_bucket_labels.extend([bucket_label] * len(normalized_deltas))
                    
                    # Trier tous les deltas normalisés et appliquer le seuil
                    if all_normalized_deltas:
                        sorted_indices = np.argsort(all_normalized_deltas)
                        threshold_idx = int(len(all_normalized_deltas) * alpha)
                        
                        # Sélectionner les points au-dessus du seuil
                        thresholded_indices = sorted_indices[:threshold_idx]
                        
                        # Organiser les points par  
                        bucket_points = {}
                        for idx in thresholded_indices:
                            bucket = all_bucket_labels[idx]
                            if bucket not in bucket_points:
                                bucket_points[bucket] = []
                            bucket_points[bucket].append((all_timestamps[idx], all_normalized_deltas[idx]))
                        
                        # Écrire les points dans le fichier txt
                        for bucket in sorted(bucket_points.keys()):
                            f.write(f"Bucket: {bucket}\n")
                            points = bucket_points[bucket]
                            proportion = len(points) / threshold_idx
                            daily_proportions[spread_type][event_type][file_date][bucket] = proportion
                            
                            for ts, delta in points:
                                f.write(f"{ts},{delta}\n")
                            f.write("\n")
    
    # Créer les graphiques de proportions par jour
    for spread_type in ['zero_spread', 'nonzero_spread']:
        for event_type in ['A', 'C', 'T']:
            # Create three figures: time series, KDE, and after mix
            fig_ts, axes_ts = plt.subplots(4, 5, figsize=(25, 20))
            fig_kde, axes_kde = plt.subplots(4, 5, figsize=(25, 20))
            fig_mix, axes_mix = plt.subplots(4, 5, figsize=(25, 20))
            
            fig_ts.suptitle(f'Proportions journalières par bucket - {event_type} ({spread_type})', fontsize=16)
            fig_kde.suptitle(f'Distribution KDE des proportions - {event_type} ({spread_type})', fontsize=16)
            fig_mix.suptitle(f'Distribution après mélange - {event_type} ({spread_type})', fontsize=16)
            
            # Récupérer toutes les dates et buckets uniques
            all_dates = sorted(daily_proportions[spread_type][event_type].keys())
            all_buckets = set()
            for date_data in daily_proportions[spread_type][event_type].values():
                all_buckets.update(date_data.keys())
            all_buckets = sorted(all_buckets, key=lambda x: float(x.strip('[]').split(',')[0]))
            
            # Plot each bucket
            for idx, bucket in enumerate(all_buckets):
                if float(bucket.strip('[]').split(',')[0]) <= 0.9:
                    row = idx // 5
                    col = idx % 5
                    
                    proportions = [daily_proportions[spread_type][event_type][date].get(bucket, 0) 
                                 for date in all_dates]
                    
                    # Time series plot
                    ax_ts = axes_ts[row, col]
                    ax_ts.plot(range(len(all_dates)), proportions, marker='o')
                    ax_ts.set_title(f'Bucket {bucket}')
                    ax_ts.set_xlabel('Jours')
                    ax_ts.set_ylabel('Proportion')
                    ax_ts.set_xticks(range(len(all_dates)))
                    ax_ts.set_xticklabels(all_dates, rotation=45)
                    ax_ts.grid(True, alpha=0.3)
                    
                    # KDE plot
                    ax_kde = axes_kde[row, col]
                    sns.kdeplot(data=proportions, ax=ax_kde)
                    ax_kde.set_title(f'Bucket {bucket}')
                    ax_kde.set_xlabel('Proportion')
                    ax_kde.set_ylabel('Densité')
                    ax_kde.grid(True, alpha=0.3)
                    
                    # After mix plot
                    ax_mix = axes_mix[row, col]
                    sns.histplot(data=proportions, ax=ax_mix, bins=20, kde=True)
                    ax_mix.set_title(f'Bucket {bucket}')
                    ax_mix.set_xlabel('Proportion')
                    ax_mix.set_ylabel('Fréquence')
                    ax_mix.grid(True, alpha=0.3)
            
            # Remove empty subplots
            for idx in range(len(all_buckets), 20):
                row = idx // 5
                col = idx % 5
                fig_ts.delaxes(axes_ts[row, col])
                fig_kde.delaxes(axes_kde[row, col])
                fig_mix.delaxes(axes_mix[row, col])
            
            plt.tight_layout()
            
            # Save plots
            plot_path_ts = os.path.join(plot_output_dir, 
                                      f"{stock}_{spread_type}_{event_type}_daily_proportions_grid_thresholded.png")
            plot_path_kde = os.path.join(plot_output_dir, 
                                       f"{stock}_{spread_type}_{event_type}_kde_distributions_grid_thresholded.png")
            plot_path_mix = os.path.join(plot_output_dir, 
                                       f"{stock}_{spread_type}_{event_type}_after_mix_grid_thresholded.png")
            
            fig_ts.savefig(plot_path_ts, bbox_inches='tight')
            fig_kde.savefig(plot_path_kde, bbox_inches='tight')
            fig_mix.savefig(plot_path_mix, bbox_inches='tight')
            
            plt.close('all')
            
            logging.info(f"Graphiques sauvegardés: {plot_path_ts}, {plot_path_kde}, {plot_path_mix}")

if __name__ == "__main__":
    data_folder = "/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ/GOOGL_filtered"
    analyze_normalized_intensities(data_folder)
