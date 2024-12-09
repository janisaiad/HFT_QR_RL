import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import os
import polars as pl

files_parquet = glob.glob(os.path.join("/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/CHICAGO/LCID", "*.parquet"))

imb = np.array([])
price = np.array([])
limite = 0
for f in tqdm(files_parquet):
    df = pl.read_parquet(f)
    df = df[:-100]
    imb = np.concatenate([imb, df['imbalance'].to_numpy()])
    price = np.concatenate([price, df['Mean_price_diff'].to_numpy()])

print(f"A quel point c'est long: {len(imb)}")
# ATTENTION L'imbalance est moins l'imbalance et les mid price sont - les midprices, dcp ca chage rien au graphe mais a modif au cas ou


indices_trie = np.argsort(imb)

bounds = 0.85
imb_trie = imb[indices_trie]
price_trie = price[indices_trie]
mask = (imb_trie >= -bounds)&(imb_trie <= bounds)
imb_trie = imb_trie[mask]
price_trie = price_trie[mask]
group_size = 350000

imb_trie_groups = [imb_trie[i:i + group_size] for i in range(0, len(imb_trie), group_size)]
price_trie_groups = [price_trie[i:i + group_size] for i in range(0, len(price_trie), group_size)]

imb_trie_means = np.array([np.mean(group) for group in imb_trie_groups])
price_trie_means = np.array([np.mean(group) for group in price_trie_groups])

imb_trie_std = np.array([np.std(group)/np.sqrt(len(group)) for group in price_trie_groups])


imb_trade = np.array([])
price_trade = np.array([])
imb_add = np.array([])
price_add = np.array([])
imb_cancel = np.array([])
price_cancel = np.array([])
limite = 0
for f in tqdm(files_parquet):
    df = pl.read_parquet(f)
    df = df[:-100]
    df_trade = df[df['action'] == 'T']
    df_cancel = df[df['action'] == 'C']
    df_add = df[df['action'] == 'A']
    imb_trade = np.concatenate([imb_trade, df_trade['imbalance'].to_numpy()])
    price_trade = np.concatenate([price_trade, df_trade['Mean_price_diff'].to_numpy()])
    imb_add = np.concatenate([imb_add, df_add['imbalance'].to_numpy()])
    price_add = np.concatenate([price_add, df_add['Mean_price_diff'].to_numpy()])
    imb_cancel = np.concatenate([imb_cancel, df_cancel['imbalance'].to_numpy()])
    price_cancel = np.concatenate([price_cancel, df_cancel['Mean_price_diff'].to_numpy()])
    
imb_tot = [imb_trade,imb_add,imb_cancel]
price_tot = [price_trade,price_add,price_cancel]

print(f"A quel point c'est long: {len(imb_trade)}")


imb = np.array([])
intensity = np.array([])
limite = 0
for f in tqdm(files_parquet):
    df = pl.read_parquet(f)
    df = df[:-100]
    imb = np.concatenate([imb, df['imbalance'].to_numpy()])
    intensity = np.concatenate([intensity, df['time_diff'].to_numpy()])

intensity = 1/np.array(intensity)



indices_trie = np.argsort(imb)

bounds = 0.9
imb_trie = imb[indices_trie]
intensity_trie = intensity[indices_trie]
mask = (imb_trie >= -bounds)&(imb_trie <= bounds)
imb_trie = imb_trie[mask]
intensity_trie = intensity_trie[mask]
group_size = 300000

imb_trie_groups = [imb_trie[i:i + group_size] for i in range(0, len(imb_trie), group_size)]
intensity_trie_groups = [intensity_trie[i:i + group_size] for i in range(0, len(intensity_trie), group_size)]

imb_trie_means = np.array([np.mean(group) for group in imb_trie_groups])
intensity_trie_means = np.array([np.mean(group) for group in intensity_trie_groups])

intensity_trie_std = np.array([np.std(group)/np.sqrt(len(group)) for group in intensity_trie_groups])


imb_trade = np.array([])
intensity_trade = np.array([])
imb_add = np.array([])
intensity_add = np.array([])
imb_cancel = np.array([])
intensity_cancel = np.array([])
limite = 0
for f in tqdm(files_parquet):
    df = pl.read_parquet(f)
    df = df[:-100]
    df_trade = df[df['action'] == 'T']
    df_cancel = df[df['action'] == 'C']
    df_add = df[df['action'] == 'A']
    imb_trade = np.concatenate([imb_trade, df_trade['imbalance'].to_numpy()])
    intensity_trade = np.concatenate([intensity_trade, df_trade['time_diff'].to_numpy()])
    imb_add = np.concatenate([imb_add, df_add['imbalance'].to_numpy()])
    intensity_add = np.concatenate([intensity_add, df_add['time_diff'].to_numpy()])
    imb_cancel = np.concatenate([imb_cancel, df_cancel['imbalance'].to_numpy()])
    intensity_cancel = np.concatenate([intensity_cancel, df_cancel['time_diff'].to_numpy()])
    
imb_tot = [imb_trade,imb_add,imb_cancel]
intensity_tot = [1/np.array(intensity_trade),1/np.array(intensity_add),1/np.array(intensity_cancel)]

print(f"A quel point c'est long: {len(imb_trade)}")