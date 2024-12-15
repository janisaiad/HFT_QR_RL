# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# %%
import json
import polars as pl
import matplotlib.pyplot as plt


# %%
# Fonction pour charger les données JSON
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Charger les fichiers JSON
condition = load_json('/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash3/data/dbn/condition.json')
manifest = load_json('/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash3/data/dbn/manifest.json')
metadata = load_json('/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash3/data/csv/metadata.json')

# Fonction pour charger les données CSV
def load_csv(stock, date):
    file_path = f'/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash3/data/csv/{stock}/{date}.csv'
    return pl.read_csv(file_path)

# Spécifier les dates et les stocks
dates = ["20240722", "20240723", "20240724", "20240725", "20240726", "20240729", "20240730", "20240731", "20240801", "20240802", "20240805", "20240806", "20240807", "20240808", "20240809", "20240812", "20240813", "20240814", "20240815", "20240816", "20240819", "20240820", "20240821", "20240822", "20240823", "20240826", "20240827", "20240828", "20240829", "20240830", "20240903", "20240904", "20240905", "20240906", "20240909", "20240910", "20240911", "20240912", "20240913", "20240916", "20240917", "20240918", "20240919", "20240920", "20240923", "20240924", "20240925", "20240926", "20240927", "20240930", "20241001", "20241002", "20241003", "20241004", "20241007", "20241008", "20241009", "20241010", "20241011", "20241014", "20241015", "20241016", "20241017", "20241018", "20241021"]
stocks = ["LCID"]



# Charger les données pour chaque stock et chaque date dans des datasets différents
data_dict = {}
for stock in stocks:
    data_dict[stock] = {}
    for date in dates:
        # Use sample_n instead of sample(frac) for polars DataFrame
        df = load_csv(stock, date)
        sample_size = int(0.1 * len(df))
        data_dict[stock][date] = df.sample(n=sample_size, seed=1)

# Concaténer toutes les données
data_list = [data_dict[stock][date] for stock in stocks for date in dates]
data = pl.concat(data_list)

publisher_id = 39

# Filtrer par publisher_id = 39 
data = data.filter(pl.col('publisher_id') == publisher_id)

# Convertir ts_event en datetime
data = data.with_columns(pl.col('ts_event').str.to_datetime(time_unit='ns'))
data = data.sort('ts_event')


# %%
data.head(20)

# %%
from scipy.stats import pearsonr

# Fonction pour calculer la corrélation avec la fonction identité
def calculate_correlation(data, bucket_number):
    data_with_buckets = data.with_columns(
        pl.col('imbalance').map_elements(lambda x: round(bucket_number * x) if x is not None else None, return_dtype=pl.Int64).alias('imbalance_bucket')
    )
    data_with_buckets = data_with_buckets.drop_nulls(subset=['imbalance_bucket'])
    
    mean_delta_mid_price = data_with_buckets.group_by('imbalance_bucket').agg(
        pl.col('delta_mid_price').mean()
    ).sort('imbalance_bucket')
    
    correlation, _ = pearsonr(
        mean_delta_mid_price['imbalance_bucket'].to_numpy(),
        mean_delta_mid_price['delta_mid_price'].to_numpy()
    )
    return correlation

# Fonction pour tracer et sauvegarder les graphiques pour chaque stock
def plot_and_save_by_stock(data, stock, bucket_number):
    data_with_buckets = data.with_columns(
        pl.col('imbalance').map_elements(lambda x: round(bucket_number * x) if x is not None else None, return_dtype=pl.Int64).alias('imbalance_bucket')
    )
    data_with_buckets = data_with_buckets.drop_nulls(subset=['imbalance_bucket'])
    
    mean_delta_mid_price = data_with_buckets.group_by('imbalance_bucket').agg(
        pl.col('delta_mid_price').mean()
    ).sort('imbalance_bucket')
    
    plt.figure(figsize=(14, 7))
    plt.plot(mean_delta_mid_price['imbalance_bucket'], mean_delta_mid_price['delta_mid_price'], marker='o', label='Mean Delta Mid Price')
    plt.title(f"Mean Delta Mid Price in Horizon 10 Trades vs Imbalance Buckets for {stock}")
    plt.xlabel(f"Imbalance Buckets (round({bucket_number}*imbalance))")
    plt.ylabel("Mean Delta Mid Price in Horizon 10 Trades")
    plt.legend()
    plt.savefig(f"imbalance_plot_{stock}.png")
    plt.close()

# Calculer l'imbalance des meilleures offres et demandes
for stock in stocks:
    for date in dates:
        data_dict[stock][date] = data_dict[stock][date].with_columns([
            ((pl.col('bid_sz_00') - pl.col('ask_sz_00')) / (pl.col('bid_sz_00') + pl.col('ask_sz_00'))).alias('imbalance'),
            ((pl.col('bid_px_00') + pl.col('ask_px_00')) / 2).alias('mid_price')
        ])
        data_dict[stock][date] = data_dict[stock][date].with_columns([
            pl.col('mid_price').shift(100).alias('delta_mid_price')
        ])

# Liste pour stocker les meilleurs bucket_numbers
best_buckets = []

# Appliquer la fonction pour chaque stock
for stock in stocks:
    stock_data = pl.concat([data_dict[stock][date] for date in dates])
    stock_data = stock_data.drop_nulls(subset=['delta_mid_price'])
    
    if len(stock_data) > 0:
        best_correlation = -1
        best_bucket_number = 5
        for bucket_number in range(5, 11):
            correlation = calculate_correlation(stock_data, bucket_number)
            if correlation > best_correlation:
                best_correlation = correlation
                best_bucket_number = bucket_number
        best_buckets.append((stock, best_bucket_number))
        plot_and_save_by_stock(stock_data, stock, best_bucket_number)

# Afficher la liste des meilleurs bucket_numbers
print(best_buckets)

