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

def visu(files):
    df_final = pd.DataFrame(columns=['date', 'nombre de trades', 'variance', 'max_proba', 'min_proba', 'mean_rapport', 'intensité max', 'seuil'])
    files_csv = glob.glob(os.path.join(files, "*.csv"))
    dic = {}
    means_trades = []
    for f in tqdm(files_csv):
        df = pd.read_csv(f)
        df['ts_event'] = pd.to_datetime(df['ts_event'], errors='coerce')
        df = df[(df['ts_event'].dt.hour > 14) | ((df['ts_event'].dt.hour == 14) & (df['ts_event'].dt.minute >= 30)) & (df['ts_event'].dt.hour < 19)]
        df['time_diff'] = df['time_diff']*1.0 # pour le modif en float c'est un timedelta là
        df['imbalance'] = df['imbalance'].round(1)
        df['imbalance'] = np.where(df['side']=='A', df['imbalance'], -df['imbalance'])
        df['imbalance'] = df['imbalance'].shift()
        df = df.dropna()
        df = df[df['action'] == 'T']
        
        sizes = (np.unique(df['imbalance'].to_numpy())).tolist()
        sizes.sort()
        dic = dico_queue_size(sizes, dic)  # Add, Cancel, Trade
        for row in df.itertuples():
            taille = row.imbalance
            dic[taille][2].append([row.time_diff, row.size])

    dic = dict(sorted(dic.items()))
    means_size = []
    for i in tqdm(dic):
        means_trades.append(np.mean(dic[i][2][0]))
        means_size.append(np.mean(dic[i][2][1]))
    print(f"A quel point c'est long: {len(dic)}")
    ou = 0
    files_csv = glob.glob(os.path.join(files, "*.csv"))
    for f in tqdm(files_csv):
        file = f
        df = pd.read_csv(file)
        df['ts_event'] = pd.to_datetime(df['ts_event'], errors='coerce')
        #file = '/Users/edouard/Desktop/EA p1  HFT/HFT_QR_RL_save/Sans titre/HFT_QR_RL/data/MBO-10 analyse_2/test_csv/xnas-itch-20240726.mbp-10.csv'
        df = df[(df['ts_event'].dt.hour > 14) | ((df['ts_event'].dt.hour == 14) & (df['ts_event'].dt.minute >= 30)) & (df['ts_event'].dt.hour < 19)]
        dic_bid = {}
        dic_ask = {}
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
        dic_timestamp = dico_queue_size(sizes_bid, dic_bid)
        dic_ask = dico_queue_size(sizes_ask, dic_ask)
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

        Add = []
        Cancel = []
        Trade = []
        sizes_add = []
        sizes_cancel = []
        sizes_trade = []


        intensities_bid = dict(sorted(dic_bid.items()))

        threshold_trade = 1000
        threshold = 100

        quarter_add = []
        quarter_cancel = []
        quarter_trade = []
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

        #print(intensities_bid)
        extreme = 0.05
        intens = np.concatenate(intens)
        indices = np.argsort(intens)
        intens = intens[indices]

        add = np.concatenate(Add)
        add = add[indices]
        add = add[:int(extreme*len(add))]

        Size = np.concatenate(size)
        Size = Size[indices]
        Size = Size[:int(extreme*len(intens))]

        Size_tot = np.concatenate(size)
        indices = np.argsort(Size_tot)
        add_tot = np.concatenate(Add)

        Size_tot = Size_tot[indices][::-1][:int(extreme*len(intens))]
        add_tot = add_tot[indices][::-1][:int(extreme*len(intens))]

        prices = df[df['action'] == 'T']['price']
        time = pd.to_datetime(df[df['action'] == 'T']['ts_event'])


        prices = df[df['action'] == 'T']['price'].reset_index(drop=True)
        time = pd.to_datetime(df[df['action'] == 'T']['ts_event']).reset_index(drop=True)

        timestamps_numeric = np.array([(ts - add[0]).total_seconds() for ts in add])

        # Estimer la densité des points `add`
        density_estimator = gaussian_kde(timestamps_numeric)
        densities = density_estimator(timestamps_numeric)

        densities_normalized = (densities - np.min(densities)) / (np.max(densities) - np.min(densities))

        y_min, y_max = np.min(prices) - 0.1, np.max(prices) + 0.1
        x_values = np.repeat(add, 50)
        y_values = np.random.uniform(y_min, y_max, size=len(x_values))

        data = pd.DataFrame({'Timestamp': x_values, 'Price': y_values})
        indices = np.argsort(add_tot)
        add_tot = add_tot[indices]
        Size_tot = Size_tot[indices]
        indices = np.argsort(add)
        add = add[indices]
        Size = Size[indices]

        df = pd.read_csv(file)

        df = df[df['action'] == 'T']
        df['ts_event'] = pd.to_datetime(df['ts_event'], errors='coerce')
        df = df[(df['ts_event'].dt.hour > 14) | ((df['ts_event'].dt.hour == 14) & (df['ts_event'].dt.minute >= 30)) & (df['ts_event'].dt.hour < 19)]

        start_time = pd.to_datetime(df['ts_event'].iloc[0])
        end_time = pd.to_datetime(df['ts_event'].iloc[-1])

        timy = pd.date_range(start=start_time, end=end_time, periods=len(df['ts_event']))

        counts = []
        counts_size = []
        window_size = int((len(df['ts_event']))/10)
        for i in range(window_size, len(timy)):
            end_time = timy[i]
            start_time = timy[i - window_size]
            count = ((add >= start_time) & (add <= end_time)).sum()
            count_size = ((add_tot >= start_time) & (add_tot <= end_time)).sum()
            counts.append(count/window_size)
            counts_size.append(count_size/window_size)
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timy, y=counts, mode='lines', line=dict(color='green', width=2), name='Intensités extrêmes'))
        fig.add_trace(go.Scatter(x=timy, y=counts_size, mode='lines', line=dict(color='red', width=2), name='Sizes extrêmes'))
        fig.update_layout(
            title=(
                f"Étude du jour {timy[0].day}/{timy[0].month}/{timy[0].year}:<br>"
                f"Proba d'événements extrêmes sur une sliding window de {window_size}<br>"
                f"pour {len(add)} événements ({extreme}% plus extrême)"
            ),
            title_x=0.5,  # Center-align the title
            title_y=0.15,    # Place the title below the plot
            yaxis_title='probabilité',
            # legend=dict(
            #     x=1,         # Position the legend in the top-right
            #     y=1,         # Top position
            #     xanchor='right',
            #     yanchor='top'
            # ),
            margin=dict(t=50, b=110),
            showlegend=True
        )
        fig.write_image(f"/Volumes/T9/CSV_LCID_NASDAQ_PL_proba/{timy[0].day}-{timy[0].month}-{timy[0].year}.png", format="png")
        fig.show()
        
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=timy, y=np.array(counts_size)/np.array(counts), mode='lines', line=dict(color='red', width=2), name='Intensité/size'))

        fig.update_layout(
            title=(
                f"Étude du jour {timy[0].day}-{timy[0].month}-{timy[0].year}:<br>"
                f"Proba Intensité/Proba size d'événements extrêmes sur une sliding window de {window_size}<br>"
                f"pour {len(add)} événements ({extreme}% plus extrême)"
            ),
            title_x=0.5,  # Center-align the title
            title_y=0.15,    # Place the title below the plot
            yaxis_title='probabilité',
            # legend=dict(
            #     x=1,         # Position the legend in the top-right
            #     y=1,         # Top position
            #     xanchor='right',
            #     yanchor='top'
            # ),
            margin=dict(t=50, b=110),
            showlegend=True
        )
        fig.write_image(f"/Volumes/T9/CSV_LCID_NASDAQ_PL_rapport/{timy[0].day}-{timy[0].month}-{timy[0].year}.png", format="png")
        #fig.show()
        #print([f'{timy[0].day}/{timy[0].month}/{timy[0].year}',len(df[df['action'] == 'T']),np.var(df[df['action'] == 'T']['price'].to_numpy()),np.max(np.array(counts)),np.min(np.array(counts)),np.mean(np.array(counts_size)/np.array(counts)), np.min(np.array(intens))])
        df_final.loc[ou] = [f'{timy[0].day}/{timy[0].month}/{timy[0].year}',len(df[df['action'] == 'T']),np.var(df[df['action'] == 'T']['price'].to_numpy()),np.max(np.array(counts)),np.min(np.array(counts)),np.mean(np.array(counts_size)/np.array(counts)), np.min(np.array(intens)),intens[:int(extreme*len(intens))][-1] ]
        
        ou+=1
    df_final.to_csv('/Volumes/T9/CSV_LCID_PL_Analyse.csv', index = False)
    return df_final
    