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
import pandas as pd
import matplotlib.pyplot as plt


# %%
# Fonction pour charger les données JSON
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Charger les fichiers JSON
condition = load_json('/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash2/data/dbn/condition.json')
manifest = load_json('/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash2/data/dbn/manifest.json')
metadata = load_json('/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash2/data/csv/metadata.json')

# Fonction pour charger les données CSV
def load_csv(stock, date):
    file_path = f'/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash2/data/csv/{stock}/{date}.csv'
    return pd.read_csv(file_path)

# Spécifier les dates et les stocks
dates = ["20240624", "20240625", "20240626", "20240627", "20240628", "20240701", "20240702", "20240703", "20240705", "20240708", "20240709", "20240710", "20240711", "20240712", "20240715", "20240716", "20240717", "20240718", "20240719", "20240722", "20240723", "20240724", "20240725", "20240726", "20240729", "20240730", "20240731", "20240801", "20240802", "20240805", "20240806", "20240807", "20240808"]
stocks = ["HL","CGAU","RIOT","ASAI"]
# Charger les données pour chaque stock et chaque date dans des datasets différents
data_dict = {}
for stock in stocks:
    data_dict[stock] = {}
    for date in dates:
        data_dict[stock][date] = load_csv(stock, date).sample(frac=0.1, random_state=1)
# Concaténer toutes les données
data_list = [data_dict[stock][date] for stock in stocks for date in dates]
data = pd.concat(data_list, ignore_index=True)

# Filtrer par publisher_id = 39
data = data[data['publisher_id'] == 39]

# Convertir ts_event en datetime
data['ts_event'] = pd.to_datetime(data['ts_event'], utc=True)
data = data.sort_values(by='ts_event')


# %%
data.head()

# %%

# %%
from scipy.stats import pearsonr

# Fonction pour calculer la corrélation avec la fonction identité
def calculate_correlation(data, bucket_number):
    data['imbalance_bucket'] = data['imbalance'].apply(lambda x: round(bucket_number * x) if pd.notna(x) else None)
    data = data.dropna(subset=['imbalance_bucket'])
    mean_delta_mid_price = data.groupby('imbalance_bucket')['delta_mid_price'].mean().reset_index()
    correlation, _ = pearsonr(mean_delta_mid_price['imbalance_bucket'], mean_delta_mid_price['delta_mid_price'])
    return correlation

# Fonction pour tracer et sauvegarder les graphiques pour chaque stock
def plot_and_save_by_stock(data, stock, bucket_number):
    data['imbalance_bucket'] = data['imbalance'].apply(lambda x: round(bucket_number * x) if pd.notna(x) else None)
    data = data.dropna(subset=['imbalance_bucket'])
    mean_delta_mid_price = data.groupby('imbalance_bucket')['delta_mid_price'].mean().reset_index()
    
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
        df = data_dict[stock][date]
        df['imbalance'] = (df['bid_sz_00'] - df['ask_sz_00']) / (df['bid_sz_00'] + df['ask_sz_00'])
        df['mid_price'] = (df['bid_px_00'] + df['ask_px_00']) / 2
        df['delta_mid_price'] = df['mid_price'].diff(periods=100)

# Liste pour stocker les meilleurs bucket_numbers
best_buckets = []

# Appliquer la fonction pour chaque stock
for stock in stocks:
    stock_data = pd.concat([data_dict[stock][date] for date in dates])
    stock_data = stock_data.dropna(subset=['delta_mid_price'])
    
    if not stock_data.empty:
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


# %%
print(data)

# %%

# %%

# %%

# %%
