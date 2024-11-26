import glob
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from scipy.stats import gaussian_kde

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

def visu(files, symbole):
    df_final = pd.DataFrame(columns=['date', 'nombre de trades', 'variance', 'max_proba', 'min_proba', 'variance_proba', 'intensité max', 'seuil'])
    files_csv = glob.glob(os.path.join(files, "*.csv"))
    dic = {}
    means_trades = []
    all_time = []
    for f in tqdm(files_csv):
        df = pd.read_csv(f)
        df['ts_event'] = pd.to_datetime(df['ts_event'], errors='coerce')
        df = df[(df['ts_event'].dt.hour >= 15) & (df['ts_event'].dt.hour <= 18)]
        df['ts_event'] = df['ts_event'].dt.strftime('%H:%M:%S.%f').str[:-3]
        df['time_diff'] = df['time_diff']*1.0 # pour le modif en float c'est un timedelta là
        df['imbalance'] = df['imbalance'].round(1)
        df['imbalance'] = np.where(df['side']=='A', df['imbalance'], -df['imbalance'])
        df['imbalance'] = df['imbalance'].shift()
        df = df.dropna()
        df = df[df['action'] == 'T']
        all_time.extend(df['ts_event'].tolist())
        sizes = (np.unique(df['imbalance'].to_numpy())).tolist()
        sizes.sort()
        dic = dico_queue_size(sizes, dic)  # Add, Cancel, Trade
        for row in df.itertuples():
            taille = row.imbalance
            dic[taille][2].append([row.time_diff, row.size])
    all_time = pd.Series(pd.to_datetime(all_time, errors='coerce'))
    all_time = all_time.dropna()
    all_time = all_time.sort_values().reset_index(drop=True)
    #print(all_time)
    dic = dict(sorted(dic.items()))
    means_size = []
    for i in tqdm(dic):
        means_trades.append(np.mean(dic[i][2][0]))
        means_size.append(np.mean(dic[i][2][1]))
    ou = 0
    files_csv = glob.glob(os.path.join(files, "*.csv"))
    for f in tqdm(files_csv):
        file = f
        df = pd.read_csv(file)
        df['ts_event'] = pd.to_datetime(df['ts_event'], errors='coerce')
        df = df[(df['ts_event'].dt.hour >= 15) & (df['ts_event'].dt.hour <= 18)]
        dic_bid = {}
        df['time_diff'] = df['time_diff']*1.0 # pour le modif en float c'est un timedelta là
        df['imbalance'] = df['imbalance'].round(1)
        df['imbalance'] = np.where(df['side']=='A', df['imbalance'], -df['imbalance'])
        df['imbalance'] = df['imbalance'].shift()
        df = df.dropna()
        sizes_ask = np.array((np.unique(df['imbalance'].to_numpy())).tolist())
        sizes_ask.sort()
        sizes_bid = np.array((np.unique(df['imbalance'].to_numpy())).tolist())
        sizes_bid.sort()
        dic_bid = dico_queue_size(sizes_bid, dic_bid)  # Add, Cancel, Trade
        for row in df.itertuples():
            if row.side == 'A':
                taille = row.imbalance
                if row.action == 'A':
                    dic_bid[taille][0].append([row.time_diff,row.ts_event])
                elif row.action == 'C':
                    dic_bid[taille][1].append([row.time_diff,row.ts_event])
                elif row.action == 'T':
                    dic_bid[taille][2].append([row.time_diff,row.ts_event,row.size])
            elif row.side == 'B':
                taille = row.imbalance
                if row.action == 'A':
                    dic_bid[taille][0].append([row.time_diff,row.ts_event])
                elif row.action == 'C':
                    dic_bid[taille][1].append([row.time_diff,row.ts_event])
                elif row.action == 'T':
                    dic_bid[taille][2].append([row.time_diff,row.ts_event, row.size])

        intensities_bid = dict(sorted(dic_bid.items()))
        Add = []
        intens = []
        l = 0
        size = []
        
        for i in intensities_bid :
            tab_1 = []
            tab_time = []
            tab_size = []
            for j in range (len(intensities_bid[i][2])):
                tab_1.append(intensities_bid[i][2][j][0])
                tab_time.append(intensities_bid[i][2][j][1])
                tab_size.append(intensities_bid[i][2][j][2])
            tab_1 = np.array(tab_1)
            tab_1 = tab_1/np.array(means_trades[l])
            size.append(np.array(tab_size)/np.array(means_size[l]))
            l+=1
            intens.append(tab_1)
            Add.append(np.array(pd.to_datetime(tab_time)))

        extreme = 0.05
        intens = np.concatenate(intens)
        indices = np.argsort(intens)
        intens = intens[indices]
        add = np.concatenate(Add)
        add = add[indices]
        add = add[:int(extreme*len(add))]
        for i in range (len(add)):
            add[i] = add[i].strftime('%H:%M:%S.%f')[:-3]
        add = pd.Series(pd.to_datetime(add, errors='coerce'))

        Size_tot = np.concatenate(size)
        indices = np.argsort(Size_tot)
        add_tot = np.concatenate(Add)
        
        Size_tot = Size_tot[indices][::-1][:int(extreme*len(intens))]
        add_tot = add_tot[indices][::-1][:int(extreme*len(intens))]

        for i in range (len(add_tot)):
            add_tot[i] = add_tot[i].strftime('%H:%M:%S.%f')[:-3]
        add_tot = pd.Series(pd.to_datetime(add_tot, errors='coerce'))

        df = pd.read_csv(file)

        df = df[df['action'] == 'T']
        df['ts_event'] = pd.to_datetime(df['ts_event'], errors='coerce')
        df = df[(df['ts_event'].dt.hour >= 15) & (df['ts_event'].dt.hour <= 18)]

        start_time_ = pd.to_datetime(df['ts_event'].iloc[0])
        end_time = pd.to_datetime(df['ts_event'].iloc[-1])
        timy = pd.date_range(start=all_time[0], end=all_time[len(all_time)-1], periods=1000)
        counts = []
        counts_size = []
        window_size = 10
        for i in range(window_size, len(timy)):
            end_time = timy[i]
            start_time = timy[i - window_size]
            count_all_time = ((all_time >= start_time) & (all_time <= end_time)).sum()
            count = ((add >= start_time) & (add <= end_time)).sum()
            count_size = ((add_tot >= start_time) & (add_tot <= end_time)).sum()
            #print(count_all_time, count)
            counts.append(count/(count_all_time/len(files_csv)))
            counts_size.append(count_size/(count_all_time/len(files_csv)))
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timy, y=counts, mode='lines', line=dict(color='green', width=2), name='Intensités extrêmes'))
        fig.add_trace(go.Scatter(x=timy, y=counts_size, mode='lines', line=dict(color='red', width=2), name='Sizes extrêmes'))
        fig.update_layout(
            title=(
                f"Étude du jour {start_time_.day}/{start_time_.month}/{start_time_.year}:<br>"
                f"Proba d'événements extrêmes sur une sliding window de {window_size}<br>"
                f"pour {len(add)} événements ({extreme}% plus extrême)"
            ),
            title_x=0.5,
            title_y=0.15,
            yaxis_title='probabilité',
            margin=dict(t=50, b=110),
            showlegend=True
        )
        fig.write_image(f"/Volumes/T9/CSV_{symbole}_NASDAQ_PL_proba/{start_time_.day}-{start_time_.month}-{start_time_.year}.png", format="png")
        fig.show()
        
        # fig = go.Figure()

        # fig.add_trace(go.Scatter(x=timy, y=np.array(counts_size)/np.array(counts), mode='lines', line=dict(color='red', width=2), name='Intensité/size'))

        # fig.update_layout(
        #     title=(
        #         f"Étude du jour {start_time.day}-{start_time.month}-{start_time.year}:<br>"
        #         f"Proba Intensité/Proba size d'événements extrêmes sur une sliding window de {window_size}<br>"
        #         f"pour {len(add)} événements ({extreme}% plus extrême)"
        #     ),
        #     title_x=0.5,
        #     title_y=0.15,
        #     yaxis_title='probabilité',
        #     margin=dict(t=50, b=110),
        #     showlegend=True
        # )
        # fig.write_image(f"/Volumes/T9/CSV_LCID_NASDAQ_PL_rapport/{start_time.day}-{start_time.month}-{start_time.year}.png", format="png")
        df_final.loc[ou] = [f'{start_time_.day}/{start_time_.month}/{start_time_.year}',len(df[df['action'] == 'T']),np.var(df[df['action'] == 'T']['price'].to_numpy()),np.max(np.array(counts)),np.min(np.array(counts)),np.var(counts), np.min(np.array(intens)),np.mean(intens[:int(extreme*len(intens))]) ]
        ou+=1
    df_final.to_csv(f'/Volumes/T9/CSV_{symbole}_PL_Analyse.csv', index = False)
    return df_final
    