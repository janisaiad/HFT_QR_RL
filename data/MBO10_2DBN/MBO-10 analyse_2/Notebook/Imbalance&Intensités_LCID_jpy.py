# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: IA_m1 (Python 3.9)
#     language: python
#     name: ia_m1
# ---

# %% [markdown]
# # LCID
#

# %%
import glob
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
import matplotlib.pyplot as plt

# %% [markdown]
# # Étude de l'imbalance

# %%
files_csv = glob.glob(os.path.join("/Volumes/T9/CSV_LCID_NASDAQ_PL", "*.csv"))
imb = np.array([])
price = np.array([])
limite = 0
for f in tqdm(files_csv):
    df = pd.read_csv(f)
    df = df[:-100]
    imb = np.concatenate([imb, df['imbalance'].to_numpy()])
    price = np.concatenate([price, df['Mean_price_diff'].to_numpy()])

print(f"A quel point c'est long: {len(imb)}")
# ATTENTION L'imbalance est moins l'imbalance et les mid price sont - les midprices, dcp ca chage rien au graphe mais a modif au cas ou

# %%
indices_trie = np.argsort(imb)

bounds = 0.85
imb_trie = imb[indices_trie]
price_trie = price[indices_trie]
mask = (imb_trie >= -bounds)&(imb_trie <= bounds)
imb_trie = imb_trie[mask]
price_trie = price_trie[mask]
group_size = 70000

imb_trie_groups = [imb_trie[i:i + group_size] for i in range(0, len(imb_trie), group_size)]
price_trie_groups = [price_trie[i:i + group_size] for i in range(0, len(price_trie), group_size)]

imb_trie_means = np.array([np.mean(group) for group in imb_trie_groups])
price_trie_means = np.array([np.mean(group) for group in price_trie_groups])

imb_trie_std = np.array([np.std(group)/np.sqrt(len(group)) for group in price_trie_groups])

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=imb_trie_means,y=price_trie_means + 1.96 * imb_trie_std,mode='lines',line=dict(width=0),name='Upper Bound',showlegend=False))
fig.add_trace(go.Scatter(x=imb_trie_means,y=price_trie_means - 1.96 * imb_trie_std,mode='lines',line=dict(width=0),fill='tonexty',fillcolor='blue',name='Intervalle de confiance à 95%',showlegend=True))
fig.add_trace(go.Scatter(x=imb_trie_means,y=price_trie_means,mode='lines',name='Prix',line=dict(color='red'),showlegend=True))
fig.update_layout(title='Imbalance non recentré',xaxis_title='imbalance',yaxis_title='delta_price',showlegend=True)
fig.show()

# %%
nb_bins = 30
counts, bin_edges = np.histogram(imb_trie, bins=nb_bins)
bin_centers = (bin_edges[:-1]+bin_edges[1:])/2

fig = go.Figure()
fig.add_trace(go.Scatter(x=bin_centers,y=counts,mode='lines',name='Density Curve'))
fig.update_layout(title=f"Courbe de distribution de l'imbalance (tous event confondus)",xaxis_title='imbalance',yaxis_title='number of events',showlegend=False)
fig.show()

# %%
imb_trade = np.array([])
price_trade = np.array([])
imb_add = np.array([])
price_add = np.array([])
imb_cancel = np.array([])
price_cancel = np.array([])
limite = 0
for f in tqdm(files_csv):
    df = pd.read_csv(f)
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


# %%
def visu_imbalance_respec(imb_tot, price_tot, i , string, group_size, bound = 0.95):
    indices_trie = np.argsort(imb_tot[i])
    imb_trie = imb_tot[i][indices_trie]
    price_trie = price_tot[i][indices_trie]
    mask = (imb_trie >= -bound)&(imb_trie <= bound)
    imb_trie = imb_trie[mask]
    price_trie = price_trie[mask]
    imb_trie_groups = [imb_trie[i:i+group_size] for i in range(0, len(imb_trie), group_size)]
    price_trie_groups = [price_trie[i:i+group_size] for i in range(0, len(price_trie), group_size)]
    imb_trie_means = np.array([np.mean(group) for group in imb_trie_groups])
    price_trie_means = np.array([np.mean(group) for group in price_trie_groups])
    imb_trie_std = np.array([np.std(group)/np.sqrt(len(group)) for group in price_trie_groups])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=imb_trie_means,y=price_trie_means + 1.96 * imb_trie_std,mode='lines',line=dict(width=0),name='Upper Bound',showlegend=False))
    fig.add_trace(go.Scatter(x=imb_trie_means,y=price_trie_means - 1.96 * imb_trie_std,mode='lines',line=dict(width=0),fill='tonexty',fillcolor='blue',name='Intervalle de confiance à 95%',showlegend=True))
    fig.add_trace(go.Scatter(x=imb_trie_means,y=price_trie_means,mode='lines',name='Prix',line=dict(color='red'),showlegend=True))
    fig.update_layout(title=f'Imbalance des {string}',xaxis_title='imbalance',yaxis_title='delta_price',showlegend=True)
    fig.show()

    nb_bins = 20
    counts, bin_edges = np.histogram(imb_trie, bins=nb_bins)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bin_centers,y=counts,mode='lines',name='Density Curve'))
    fig.update_layout(title=f"Courbe de distribution de l'imbalance des {string}",xaxis_title='imbalance',yaxis_title='number of events',showlegend=False)
    fig.show()

def visu_event_vs_imbalance(imb_tot, price_tot, string, bound = 0.9):
    fig = go.Figure()
    nb_bins = 25
    counts_tot = []
    bins_centers_tot = []
    for i in range (len(imb_tot)):
        indices_trie = np.argsort(imb_tot[i])
        imb_trie = imb_tot[i][indices_trie]
        price_trie = price_tot[i][indices_trie]
        mask = (imb_trie >= -bound)&(imb_trie <= bound)
        imb_trie = imb_trie[mask]
        price_trie = price_trie[mask]
        counts, bin_edges = np.histogram(imb_trie, bins=nb_bins)
        counts_tot.append(counts)
        bins_centers_tot.append((bin_edges[:-1]+bin_edges[1:])/2)
    counts_total = np.array(counts_tot[0])+np.array(counts_tot[1])+np.array(counts_tot[2])
    for i in range (len(imb_tot)):
        fig.add_trace(go.Scatter(x=bins_centers_tot[i],y=np.array(counts_tot[i])/counts_total,mode='lines',name=f'{string[i]}'))
    fig.update_layout(title=f"Courbes de distribution de l'imbalance (tous les event))",xaxis_title='imbalance',yaxis_title="Probabilité de l'event",showlegend=True)
    fig.show()


# %%
visu_imbalance_respec(imb_tot, price_tot, 0 , 'trades', 5000)
visu_imbalance_respec(imb_tot, price_tot, 1 , 'add', 50000)
visu_imbalance_respec(imb_tot, price_tot, 2 , 'cancel', 40000)
visu_event_vs_imbalance(imb_tot, price_tot, ['Trades', 'Cancel',' Add'], bound = .95)

# %%
imb = np.array([])
intensity = np.array([])
limite = 0
for f in tqdm(files_csv):
    df = pd.read_csv(f)
    df = df[:-100]
    imb = np.concatenate([imb, df['imbalance'].to_numpy()])
    intensity = np.concatenate([intensity, df['time_diff'].to_numpy()])

intensity = 1/np.array(intensity)

# %%
indices_trie = np.argsort(imb)

bounds = 0.95
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

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=imb_trie_means,y=intensity_trie_means + 1.96 * intensity_trie_std,mode='lines',line=dict(width=0),name='Upper Bound',showlegend=False))
fig.add_trace(go.Scatter(x=imb_trie_means,y=intensity_trie_means - 1.96 * intensity_trie_std,mode='lines',line=dict(width=0),fill='tonexty',fillcolor='blue',name='Intervalle de confiance à 95%',showlegend=True))
fig.add_trace(go.Scatter(x=imb_trie_means,y=intensity_trie_means,mode='lines',name='intensity',line=dict(color='red'),showlegend=True))
fig.update_layout(title='Intensity (tous events confondus)',xaxis_title='imbalance',yaxis_title='Intensity',showlegend=True)
fig.show()

# %%
imb_trade = np.array([])
intensity_trade = np.array([])
imb_add = np.array([])
intensity_add = np.array([])
imb_cancel = np.array([])
intensity_cancel = np.array([])
limite = 0
for f in tqdm(files_csv):
    df = pd.read_csv(f)
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


# %%
def visu_intensity_respec(imb_tot, intensity_tot, i , string, group_size, bound = 0.9):
    indices_trie = np.argsort(imb_tot[i])
    imb_trie = imb_tot[i][indices_trie]
    intensity_trie = intensity_tot[i][indices_trie]
    mask = (imb_trie >= -bound) & (imb_trie <= bound)
    imb_trie = imb_trie[mask]
    intensity_trie = intensity_trie[mask]
    imb_trie_groups = [imb_trie[i:i + group_size] for i in range(0, len(imb_trie), group_size)]
    intensity_trie_groups = [intensity_trie[i:i + group_size] for i in range(0, len(intensity_trie), group_size)]
    imb_trie_means = np.array([np.mean(group) for group in imb_trie_groups])
    intensity_trie_means = np.array([np.mean(group) for group in intensity_trie_groups])
    intensity_trie_std = np.array([np.std(group)/np.sqrt(len(group)) for group in intensity_trie_groups])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=imb_trie_means,y=intensity_trie_means + 1.96 * intensity_trie_std,mode='lines',line=dict(width=0),name='Upper Bound',showlegend=False))
    fig.add_trace(go.Scatter(x=imb_trie_means,y=intensity_trie_means - 1.96 * intensity_trie_std,mode='lines',line=dict(width=0),fill='tonexty',fillcolor='blue',name='Intervalle de confiance à 95%',showlegend=True))
    fig.add_trace(go.Scatter(x=imb_trie_means,y=intensity_trie_means,mode='lines',name='intensity',line=dict(color='red'),showlegend=True))
    fig.update_layout(title=f"Intensity des {string} en fonction de l'imbalance",xaxis_title='imbalance',yaxis_title='Intensity',showlegend=True)
    fig.show()


# %%
visu_intensity_respec(imb_tot, intensity_tot, 0 , 'trades', 5000)
visu_intensity_respec(imb_tot, intensity_tot, 1 , 'add', 100000)
visu_intensity_respec(imb_tot, intensity_tot, 2 , 'cancel', 90000)


# %% [markdown]
# # Calcul Intensités par average event size

# %%
def dico_queue_size(sizes, dic):
    for i in range (len(sizes)):
        if sizes[i] not in dic:
            dic[sizes[i]] = [[], [], []]
    return dic

def compute_means(dico):
    sums = 0
    means = 0
    keys = np.array(list(dico.keys()))
    for i in range (len(keys)):
        means = means+keys[i]*len(dico[keys[i]][0])+keys[i]*len(dico[keys[i]][1])+keys[i]*len(dico[keys[i]][2])
        sums = sums+len(dico[keys[i]][0])+len(dico[keys[i]][1])+len(dico[keys[i]][2])
    return means/sums

def filtrage(dico, nombre_bins, threshold=100):
    dico_p = dict(reversed(list(dico.items())))
    keys = list(dico_p.keys())
    i = 0
    while len(dico_p[keys[i]][0])<threshold:
        i += 1
    values = np.linspace(0, keys[i], nombre_bins, endpoint=True)
    keys = np.array(list(dico.keys()))
    
    real_dic = {}
    for i in range (len(keys)):
        real_k_index = np.argmin(np.abs(values-keys[i]))
        real_k = values[real_k_index]
        
        if real_k not in real_dic:
            real_dic[real_k] = [
                np.array(dico[keys[i]][0]),
                np.array(dico[keys[i]][1]),
                np.array(dico[keys[i]][2])
            ]
        else:
            real_dic[real_k] = [
                np.concatenate([real_dic[real_k][0], dico[keys[i]][0]]),
                np.concatenate([real_dic[real_k][1], dico[keys[i]][1]]),
                np.concatenate([real_dic[real_k][2], dico[keys[i]][2]])
            ]
    return real_dic

def remove_nan_from_dico(dico):
    cleaned_dico = {}
    for key, value_lists in dico.items():
        cleaned_value_lists = []
        for value_list in value_lists:
            value_array = np.array(value_list)
            cleaned_array = value_array[~np.isnan(value_array)]
            cleaned_value_lists.append(cleaned_array.tolist())
        cleaned_dico[key] = cleaned_value_lists
    return cleaned_dico


# %%
dic = {}

average_event_size = []
for f in tqdm(files_csv):
    df = pd.read_csv(f)
    df['time_diff'] = df['time_diff']*1.0 # pour le modif en float c'est un timedelta là
    sizes = np.unique(np.array((np.unique(df['bid_sz_00'].to_numpy())).tolist() + (np.unique(df['ask_sz_00'].to_numpy())).tolist()))
    sizes.sort()
    dic = dico_queue_size(sizes, dic)  # Add, Cancel, Trade
    
    for row in df.itertuples():
        average_event_size.append(row.size)
        if row.side == 'A':
            taille = row.ask_sz_00
        elif row.side == 'B':
            taille = row.bid_sz_00
        if row.action == 'A':
            dic[taille][0].append(row.time_diff)
        elif row.action == 'C':
            dic[taille][1].append(row.time_diff)
        elif row.action == 'T':
            dic[taille][2].append(row.time_diff)

# %%
Add = []
Cancel = []
Trade = []
sizes_add = []
sizes_cancel = []
sizes_trade = []
threshold = 100

#dic = remove_nan_from_dico(dic)
average_sizes = compute_means(dic)
intensities = dict(sorted(dic.items()))
intensities = filtrage(intensities, 30, threshold=100)

threshold_trade = 1000
threshold = 100

quarter_add = []
quarter_cancel = []
quarter_trade = []

for i in intensities:
    tab = np.concatenate((intensities[i][0], intensities[i][1], intensities[i][2]))
    if (len(intensities[i][0])>threshold):
            Add.append(1/np.mean(tab)*len(intensities[i][0])/len(tab))
            quarter_add.append(np.var(tab)*1/np.mean(tab)*len(intensities[i][0])/len(tab)*1/np.sqrt(len(tab)))
            sizes_add.append(i)
    if len(intensities[i][1])!=0:
        if (len(intensities[i][1])>threshold):
            Cancel.append(1/np.mean(tab)*len(intensities[i][1])/len(tab))
            quarter_cancel.append(np.var(tab)*1/np.mean(tab)*len(intensities[i][1])/len(tab)*1/np.sqrt(len(tab)))
            sizes_cancel.append(i)
    if len(intensities[i][2])!=0:
        if (len(intensities[i][2])>threshold_trade):
            Trade.append(1/np.mean(tab)*len(intensities[i][2])/len(tab))
            quarter_trade.append(np.var(tab)*1/np.mean(tab)*len(intensities[i][2])/len(tab)*1/np.sqrt(len(tab)))
            sizes_trade.append(i)

fig = go.Figure()

fig.add_trace(go.Scatter(x=sizes_add/np.mean(average_event_size),y=Add,mode='lines',name='Add',showlegend=True,line=dict(color='blue')))
fig.add_trace(go.Scatter(x=sizes_add/np.mean(average_event_size),y=np.array(Add)+1.96*np.array(quarter_add),mode='lines',name='Add Upper CI',line=dict(width=0),fill=None,showlegend=False))
fig.add_trace(go.Scatter(x=sizes_add/np.mean(average_event_size),y=np.array(Add)-1.96*np.array(quarter_add),mode='lines',name='Add Lower CI',fill='tonexty',line=dict(width=0),fillcolor='rgba(0, 0, 255, 0.2)',showlegend=False))

fig.add_trace(go.Scatter(x=sizes_cancel/np.mean(average_event_size),y=Cancel,mode='lines',name='Cancel',showlegend=True,line=dict(color='green')))
fig.add_trace(go.Scatter(x=sizes_cancel/np.mean(average_event_size),y=np.array(Cancel)+1.96*np.array(quarter_cancel),mode='lines',name='Cancel Upper CI',line=dict(width=0),fill=None,showlegend=False))
fig.add_trace(go.Scatter(x=sizes_cancel/np.mean(average_event_size),y=np.array(Cancel)-1.96*np.array(quarter_cancel),mode='lines',name='Cancel Lower CI',fill='tonexty',line=dict(width=0),fillcolor='rgba(0, 255, 0, 0.2)',showlegend=False))

fig.add_trace(go.Scatter(x=sizes_trade/np.mean(average_event_size),y=Trade,mode='lines',name='Trade',showlegend=True,line=dict(color='red')))
fig.add_trace(go.Scatter(x=sizes_trade/np.mean(average_event_size),y=np.array(Trade)+1.96*np.array(quarter_trade),mode='lines',name='Trade Upper CI',line=dict(width=0),fill=None,showlegend=False))
fig.add_trace(go.Scatter(x=sizes_trade/np.mean(average_event_size),y=np.array(Trade)-1.96*np.array(quarter_trade),mode='lines',name='Trade Lower CI',fill='tonexty',line=dict(width=0),fillcolor='rgba(255, 0, 0, 0.2)',showlegend=False))

fig.update_layout(title="Intensité par Queue size avec intervalles d'incertitude",xaxis_title='Size (par Mean Event Size)',yaxis_title='Intensity (num par sec)',showlegend=True)
fig.show()

# %%
dic = {}

average_event_size = []
for f in tqdm(files_csv):
    df = pd.read_csv(f)
    #df['time_diff'] = df['time_diff']*1.0 # pour le modif en float c'est un timedelta là
    sizes = np.unique(np.array((np.unique(df['bid_sz_00'].to_numpy())).tolist() + (np.unique(df['ask_sz_00'].to_numpy())).tolist()))
    sizes.sort()
    dic = dico_queue_size(sizes, dic)  # Add, Cancel, Trade
    
    for row in df.itertuples():
        average_event_size.append(row.size)
        if row.side == 'A':
            taille = row.ask_sz_00
        elif row.side == 'B':
            taille = row.bid_sz_00
        if row.action == 'A':
            dic[taille][0].append(row.imbalance)
        elif row.action == 'C':
            dic[taille][1].append(row.imbalance)
        elif row.action == 'T':
            dic[taille][2].append(row.imbalance)

# %%
Add = []
Cancel = []
Trade = []
sizes_add = []
sizes_cancel = []
sizes_trade = []

#dic = remove_nan_from_dico(dic)
average_sizes = compute_means(dic)
intensities = dict(sorted(dic.items()))
intensities = filtrage(intensities, 30, threshold=100)

threshold_trade = 1000
threshold = 10

quarter_add = []
quarter_cancel = []
quarter_trade = []

for i in intensities:
    tab = np.concatenate((intensities[i][0], intensities[i][1], intensities[i][2]))
    if (len(intensities[i][0])>threshold):
            Add.append(np.mean(tab)*len(intensities[i][0])/len(tab))
            quarter_add.append(np.var(tab)*1/np.mean(tab)*len(intensities[i][0])/len(tab)*1/np.sqrt(len(tab)))
            sizes_add.append(i)
    if len(intensities[i][1])!=0:
        if (len(intensities[i][1])>threshold):
            Cancel.append(np.mean(tab)*len(intensities[i][1])/len(tab))
            quarter_cancel.append(np.var(tab)*1/np.mean(tab)*len(intensities[i][1])/len(tab)*1/np.sqrt(len(tab)))
            sizes_cancel.append(i)
    if len(intensities[i][2])!=0:
        if (len(intensities[i][2])>threshold_trade):
            Trade.append(np.mean(tab)*len(intensities[i][2])/len(tab))
            quarter_trade.append(np.var(tab)*1/np.mean(tab)*len(intensities[i][2])/len(tab)*1/np.sqrt(len(tab)))
            sizes_trade.append(i)

fig = go.Figure()

fig.add_trace(go.Scatter(x=sizes_add/np.mean(average_event_size),y=Add,mode='lines',name='Add',showlegend=True,line=dict(color='blue')))
#fig.add_trace(go.Scatter(y=sizes_add/np.mean(average_event_size),x=np.array(Add)+1.96*np.array(quarter_add),mode='lines',name='Add Upper CI',line=dict(width=0),fill=None,showlegend=False))
#fig.add_trace(go.Scatter(y=sizes_add/np.mean(average_event_size),x=np.array(Add)-1.96*np.array(quarter_add),mode='lines',name='Add Lower CI',fill='tonexty',line=dict(width=0),fillcolor='rgba(0, 0, 255, 0.2)',showlegend=False))

fig.add_trace(go.Scatter(x=sizes_cancel/np.mean(average_event_size),y=Cancel,mode='lines',name='Cancel',showlegend=True,line=dict(color='green')))
#fig.add_trace(go.Scatter(y=sizes_cancel/np.mean(average_event_size),x=np.array(Cancel)+1.96*np.array(quarter_cancel),mode='lines',name='Cancel Upper CI',line=dict(width=0),fill=None,showlegend=False))
#fig.add_trace(go.Scatter(y=sizes_cancel/np.mean(average_event_size),x=np.array(Cancel)-1.96*np.array(quarter_cancel),mode='lines',name='Cancel Lower CI',fill='tonexty',line=dict(width=0),fillcolor='rgba(0, 255, 0, 0.2)',showlegend=False))

fig.add_trace(go.Scatter(x=sizes_trade/np.mean(average_event_size),y=Trade,mode='lines',name='Trade',showlegend=True,line=dict(color='red')))
#fig.add_trace(go.Scatter(y=sizes_trade/np.mean(average_event_size),x=np.array(Trade)+1.96*np.array(quarter_trade),mode='lines',name='Trade Upper CI',line=dict(width=0),fill=None,showlegend=False))
#fig.add_trace(go.Scatter(y=sizes_trade/np.mean(average_event_size),x=np.array(Trade)-1.96*np.array(quarter_trade),mode='lines',name='Trade Lower CI',fill='tonexty',line=dict(width=0),fillcolor='rgba(255, 0, 0, 0.2)',showlegend=False))

fig.update_layout(title="Queue size avec intervalles d'incertitude en fonction de l'imbalance",xaxis_title='imbalance',yaxis_title='Size (par Mean Event Size)',showlegend=True)
fig.show()

# %%
queue_size_same_trade = np.array([])
queue_size_opposite_trade = np.array([])
time_trade = np.array([])

queue_size_same_cancel = np.array([])
queue_size_opposite_cancel = np.array([])
time_cancel = np.array([])

queue_size_same_add = np.array([])
queue_size_opposite_add = np.array([])
time_add = np.array([])

limite = 0
for f in tqdm(files_csv):
    df = pd.read_csv(f)
    df = df[:-100]
    df_cancel = df[df['action'] == 'C']
    df_cancel['ts_event'] = pd.to_datetime(df_cancel['ts_event'], errors='coerce')
    df_cancel['seconds_since_start_of_day'] = (
        df_cancel['ts_event'].dt.hour*3600+
        df_cancel['ts_event'].dt.minute*60+
        df_cancel['ts_event'].dt.second+
        df_cancel['ts_event'].dt.microsecond/1e6
    )
    df_add = df[df['action'] == 'A']
    df_add['ts_event'] = pd.to_datetime(df_add['ts_event'], errors='coerce')
    df_add['seconds_since_start_of_day'] = (
        df_add['ts_event'].dt.hour*3600+
        df_add['ts_event'].dt.minute*60+
        df_add['ts_event'].dt.second+
        df_add['ts_event'].dt.microsecond/1e6
    )
    df_trade = df[df['action'] == 'T']
    df_trade['ts_event'] = pd.to_datetime(df_trade['ts_event'], errors='coerce')
    df_trade['seconds_since_start_of_day'] = (
        df_trade['ts_event'].dt.hour*3600+
        df_trade['ts_event'].dt.minute*60+
        df_trade['ts_event'].dt.second+
        df_trade['ts_event'].dt.microsecond/1e6
    )

    queue_size_same_trade = np.concatenate([queue_size_same_trade, df_trade['size_same'].to_numpy()])
    queue_size_opposite_trade = np.concatenate([queue_size_opposite_trade, df_trade['size_opposite'].to_numpy()])
    time_trade = np.concatenate([time_trade, df_trade['seconds_since_start_of_day'].dropna().to_numpy()])
    
    queue_size_same_add = np.concatenate([queue_size_same_add, df_add['size_same'].to_numpy()])
    queue_size_opposite_add = np.concatenate([queue_size_opposite_add, df_add['size_opposite'].to_numpy()])
    time_add = np.concatenate([time_add, df_add['seconds_since_start_of_day'].dropna().to_numpy()])
    
    queue_size_same_cancel = np.concatenate([queue_size_same_cancel, df_cancel['size_same'].to_numpy()])
    queue_size_opposite_cancel = np.concatenate([queue_size_opposite_cancel, df_cancel['size_opposite'].to_numpy()])
    time_cancel = np.concatenate([time_cancel, df_cancel['seconds_since_start_of_day'].dropna().to_numpy()])

time_add = np.array(time_add, dtype=float)
time_trade = np.array(time_trade, dtype=float)
time_cancel = np.array(time_cancel, dtype=float)
print(f"A quel point c'est long: {len(time_add)}")

# %%
indices_trie_add = np.argsort(time_add)


time_add = time_add[indices_trie_add]
queue_size_same_add = queue_size_same_add[indices_trie_add]
queue_size_opposite_add = queue_size_opposite_add[indices_trie_add]
group_size_add = 10000

time_trie_groups_add = [time_add[i:i + group_size_add] for i in range(0, len(time_add), group_size_add)]
queue_size_same_trie_groups_add = [queue_size_same_add[i:i + group_size_add] for i in range(0, len(queue_size_same_add), group_size_add)]
queue_size_opposite_trie_groups_add = [queue_size_opposite_add[i:i + group_size_add] for i in range(0, len(queue_size_opposite_add), group_size_add)]

time_trie_means_add = np.array([np.mean(group) for group in time_trie_groups_add])
queue_size_same_trie_means_add = np.array([np.mean(group) for group in queue_size_same_trie_groups_add])
queue_size_opposite_trie_mean_add = np.array([np.mean(group) for group in queue_size_opposite_trie_groups_add])

#imb_trie_std = np.array([np.std(group)/np.sqrt(len(group)) for group in price_trie_groups])
time_trie_means_add = pd.to_datetime(time_trie_means_add, unit='s')



indices_trie_trade = np.argsort(time_trade)


time_trade = time_trade[indices_trie_trade]
queue_size_same_trade = queue_size_same_trade[indices_trie_trade]
queue_size_opposite_trade = queue_size_opposite_trade[indices_trie_trade]
group_size_trade = 700

time_trie_groups_trade = [time_trade[i:i + group_size_trade] for i in range(0, len(time_trade), group_size_trade)]
queue_size_same_trie_groups_trade = [queue_size_same_trade[i:i + group_size_trade] for i in range(0, len(queue_size_same_trade), group_size_trade)]
queue_size_opposite_trie_groups_trade = [queue_size_opposite_trade[i:i + group_size_trade] for i in range(0, len(queue_size_opposite_trade), group_size_trade)]

time_trie_means_trade = np.array([np.mean(group) for group in time_trie_groups_trade])
queue_size_same_trie_means_trade = np.array([np.mean(group) for group in queue_size_same_trie_groups_trade])
queue_size_opposite_trie_means_trade = np.array([np.mean(group) for group in queue_size_opposite_trie_groups_trade])

#imb_trie_std = np.array([np.std(group)/np.sqrt(len(group)) for group in price_trie_groups])
time_trie_means_trade = pd.to_datetime(time_trie_means_trade, unit='s')


indices_trie_cancel = np.argsort(time_cancel)


time_cancel = time_cancel[indices_trie_cancel]
queue_size_same_cancel = queue_size_same_cancel[indices_trie_cancel]
queue_size_opposite_cancel = queue_size_opposite_cancel[indices_trie_cancel]
group_size_cancel = 1000

time_trie_groups_cancel = [time_cancel[i:i + group_size] for i in range(0, len(time_cancel), group_size_cancel)]
queue_size_same_trie_groups_cancel = [queue_size_same_cancel[i:i + group_size] for i in range(0, len(queue_size_same_cancel), group_size_cancel)]
queue_size_opposite_trie_groups_cancel = [queue_size_opposite_cancel[i:i + group_size] for i in range(0, len(queue_size_opposite_cancel), group_size_cancel)]

time_trie_means_cancel = np.array([np.mean(group) for group in time_trie_groups_cancel])
queue_size_same_trie_means_cancel = np.array([np.mean(group) for group in queue_size_same_trie_groups_cancel])
queue_size_opposite_trie_means_cancel = np.array([np.mean(group) for group in queue_size_opposite_trie_groups_cancel])

#imb_trie_std = np.array([np.std(group)/np.sqrt(len(group)) for group in price_trie_groups])
time_trie_means_cancel = pd.to_datetime(time_trie_means_cancel, unit='s')

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=time_trie_means_trade,y=queue_size_same_trie_means_trade,mode='lines',name='same',line=dict(color='blue'),showlegend=True))
fig.add_trace(go.Scatter(x=time_trie_means_trade,y=queue_size_opposite_trie_means_trade,mode='lines',name='Opposite',line=dict(color='red'),showlegend=True))
fig.update_layout(title='Queue size aux trades',xaxis_title='time',yaxis_title='Queue size',showlegend=True)
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=queue_size_opposite_trie_means_trade,y=queue_size_same_trie_means_trade,mode='markers',name='same',line=dict(color='blue'),showlegend=True))
fig.update_layout(title='Heatmap Queue size aux trades',xaxis_title='Queue size opposite',yaxis_title='Queue size same',showlegend=True)
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=time_trie_means_add,y=queue_size_same_trie_means_add,mode='lines',name='same',line=dict(color='blue'),showlegend=True))
fig.add_trace(go.Scatter(x=time_trie_means_add,y=queue_size_opposite_trie_mean_add,mode='lines',name='Opposite',line=dict(color='red'),showlegend=True))
fig.update_layout(title='Queue size aux add',xaxis_title='time',yaxis_title='Queue size',showlegend=True)
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=queue_size_opposite_trie_mean_add,y=queue_size_same_trie_means_add,mode='markers',name='same',line=dict(color='blue'),showlegend=True))
fig.update_layout(title='Heatmap Queue size aux add',xaxis_title='Queue size opposite',yaxis_title='Queue size same',showlegend=True)
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=time_trie_means_cancel,y=queue_size_same_trie_means_cancel,mode='lines',name='same',line=dict(color='blue'),showlegend=True))
fig.add_trace(go.Scatter(x=time_trie_means_cancel,y=queue_size_opposite_trie_means_cancel,mode='lines',name='Opposite',line=dict(color='red'),showlegend=True))
fig.update_layout(title='Queue size aux cancel',xaxis_title='time',yaxis_title='Queue size',showlegend=True)
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=queue_size_opposite_trie_means_cancel,y=queue_size_same_trie_means_cancel,mode='markers',name='same',line=dict(color='blue'),showlegend=True))
fig.update_layout(title='Heatmap Queue size aux cancel',xaxis_title='Queue size opposite',yaxis_title='Queue size same',showlegend=True)
fig.show()


fig = go.Figure()
fig.add_trace(go.Scatter(x=queue_size_opposite_trie_mean_add,y=queue_size_same_trie_means_add,mode='markers',name='add',line=dict(color='red'),showlegend=True))
fig.add_trace(go.Scatter(x=queue_size_opposite_trie_means_cancel,y=queue_size_same_trie_means_cancel,mode='markers',name='cancel',line=dict(color='green'),showlegend=True))
fig.add_trace(go.Scatter(x=queue_size_opposite_trie_means_trade,y=queue_size_same_trie_means_trade,mode='markers',name='trade',line=dict(color='blue'),showlegend=True))
fig.update_layout(title='Heatmap Queue size',xaxis_title='Queue size opposite',yaxis_title='Queue size same',showlegend=True)
fig.show()

# %%
dic = {}

average_event_size = []
for f in tqdm(files_csv):
    df = pd.read_csv(f)
    #df['time_diff'] = df['time_diff']*1.0 # pour le modif en float c'est un timedelta là
    sizes = np.unique(np.array((np.unique(df['bid_sz_00'].to_numpy())).tolist() + (np.unique(df['ask_sz_00'].to_numpy())).tolist()))
    sizes.sort()
    dic = dico_queue_size(sizes, dic)  # Add, Cancel, Trade
    
    for row in df.itertuples():
        average_event_size.append(row.size)
        if row.side == 'A':
            taille = row.ask_sz_00
        elif row.side == 'B':
            taille = row.bid_sz_00
        if row.action == 'A':
            dic[taille][0].append(row.diff_price)
        elif row.action == 'C':
            dic[taille][1].append(row.diff_price)
        elif row.action == 'T':
            dic[taille][2].append(row.diff_price)

# %%
Add = []
Cancel = []
Trade = []
sizes_add = []
sizes_cancel = []
sizes_trade = []

#dic = remove_nan_from_dico(dic)
average_sizes = compute_means(dic)
intensities = dict(sorted(dic.items()))
intensities = filtrage(intensities, 30, threshold=100)

threshold_trade = 100
threshold = 10

quarter_add = []
quarter_cancel = []
quarter_trade = []

for i in intensities:
    tab = np.concatenate((intensities[i][0], intensities[i][1], intensities[i][2]))
    if (len(intensities[i][0])>threshold):
            Add.append(np.mean(tab)*len(intensities[i][0])/len(tab))
            quarter_add.append(np.var(tab)*1/np.mean(tab)*len(intensities[i][0])/len(tab)*1/np.sqrt(len(tab)))
            sizes_add.append(i)
    if len(intensities[i][1])!=0:
        if (len(intensities[i][1])>threshold):
            Cancel.append(np.mean(tab)*len(intensities[i][1])/len(tab))
            quarter_cancel.append(np.var(tab)*1/np.mean(tab)*len(intensities[i][1])/len(tab)*1/np.sqrt(len(tab)))
            sizes_cancel.append(i)
    if len(intensities[i][2])!=0:
        if (len(intensities[i][2])>threshold_trade):
            Trade.append(np.mean(tab)*len(intensities[i][2])/len(tab))
            quarter_trade.append(np.var(tab)*1/np.mean(tab)*len(intensities[i][2])/len(tab)*1/np.sqrt(len(tab)))
            sizes_trade.append(i)

fig = go.Figure()

fig.add_trace(go.Scatter(x=sizes_add/np.mean(average_event_size),y=Add,mode='lines',name='Add',showlegend=True,line=dict(color='blue')))
#fig.add_trace(go.Scatter(y=sizes_add/np.mean(average_event_size),x=np.array(Add)+1.96*np.array(quarter_add),mode='lines',name='Add Upper CI',line=dict(width=0),fill=None,showlegend=False))
#fig.add_trace(go.Scatter(y=sizes_add/np.mean(average_event_size),x=np.array(Add)-1.96*np.array(quarter_add),mode='lines',name='Add Lower CI',fill='tonexty',line=dict(width=0),fillcolor='rgba(0, 0, 255, 0.2)',showlegend=False))

fig.add_trace(go.Scatter(x=sizes_cancel/np.mean(average_event_size),y=Cancel,mode='lines',name='Cancel',showlegend=True,line=dict(color='green')))
#fig.add_trace(go.Scatter(y=sizes_cancel/np.mean(average_event_size),x=np.array(Cancel)+1.96*np.array(quarter_cancel),mode='lines',name='Cancel Upper CI',line=dict(width=0),fill=None,showlegend=False))
#fig.add_trace(go.Scatter(y=sizes_cancel/np.mean(average_event_size),x=np.array(Cancel)-1.96*np.array(quarter_cancel),mode='lines',name='Cancel Lower CI',fill='tonexty',line=dict(width=0),fillcolor='rgba(0, 255, 0, 0.2)',showlegend=False))

fig.add_trace(go.Scatter(x=sizes_trade/np.mean(average_event_size),y=Trade,mode='lines',name='Trade',showlegend=True,line=dict(color='red')))
#fig.add_trace(go.Scatter(y=sizes_trade/np.mean(average_event_size),x=np.array(Trade)+1.96*np.array(quarter_trade),mode='lines',name='Trade Upper CI',line=dict(width=0),fill=None,showlegend=False))
#fig.add_trace(go.Scatter(y=sizes_trade/np.mean(average_event_size),x=np.array(Trade)-1.96*np.array(quarter_trade),mode='lines',name='Trade Lower CI',fill='tonexty',line=dict(width=0),fillcolor='rgba(255, 0, 0, 0.2)',showlegend=False))

fig.update_layout(title="Price différence en fonction des queues size",xaxis_title='Size (par Mean Event Size)',yaxis_title='price difference',showlegend=True)
fig.show()

# %%
