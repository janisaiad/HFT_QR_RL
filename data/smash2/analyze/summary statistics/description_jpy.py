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
# %matplotlib notebook

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
stocks = ["HL",'ASAI','RIOT','CGAU']
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
import pandas as pd

# Fonction pour calculer les statistiques sommaires
def calculate_summary_statistics(df):
    event_counts = df['action'].value_counts()
    trade_count = event_counts.get('T', 0) + event_counts.get('F', 0)
    cancel_count = event_counts.get('C', 0)
    add_count = event_counts.get('A', 0)
    modify_count = event_counts.get('M', 0)
    clear_book_count = event_counts.get('R', 0)
    
    return {
        'Total Events': len(df),
        'Trades': trade_count,
        'Cancels': cancel_count,
        'Adds': add_count,
        'Modifies': modify_count,
        'Clear Books': clear_book_count
    }

# Calculer les statistiques sommaires pour chaque stock
summary_stats = {}
for stock in stocks:
    df = data_dict[stock]
    stock_data = pd.concat(df.values(), ignore_index=True)
    summary_stats[stock] = calculate_summary_statistics(stock_data)

# Afficher les statistiques sommaires
for stock, stats in summary_stats.items():
    print(f"\nSummary Statistics for {stock}:")
    for stat, value in stats.items():
        print(f"{stat}: {value}")

# Visualiser les statistiques sommaires
fig, ax = plt.subplots(figsize=(12, 6))

x = list(summary_stats.keys())
total_events = [stats['Total Events'] for stats in summary_stats.values()]
trades = [stats['Trades'] for stats in summary_stats.values()]
cancels = [stats['Cancels'] for stats in summary_stats.values()]
adds = [stats['Adds'] for stats in summary_stats.values()]

ax.bar(x, total_events, label='Total Events')
ax.bar(x, trades, label='Trades')
ax.bar(x, cancels, label='Cancels')
ax.bar(x, adds, label='Adds')

ax.set_xlabel('Stocks')
ax.set_ylabel('Number of Events')
ax.set_title('Summary Statistics by Stock')
ax.legend()

plt.tight_layout()
plt.show()

# Calculer et afficher le prix moyen pour chaque stock
for stock in stocks:
    df = pd.concat(data_dict[stock].values(), ignore_index=True)
    avg_price = df['price'].mean() * 1e-9  # Convertir en unités réelles
    print(f"\nAverage price for {stock}: ${avg_price:.2f}")

# Visualiser la distribution des prix pour chaque stock
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
axs = axs.ravel()

for i, stock in enumerate(stocks):
    df = pd.concat(data_dict[stock].values(), ignore_index=True)
    prices = df['price'] * 1e-9  # Convertir en unités réelles
    
    axs[i].hist(prices, bins=50, edgecolor='black')
    axs[i].set_title(f'Price Distribution for {stock}')
    axs[i].set_xlabel('Price')
    axs[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


# %%
