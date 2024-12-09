import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import glob
import os
matplotlib.use('Agg')

def dico_queue_size(sizes, dic):
    for i in range (len(sizes)):
        if sizes[i] not in dic:
            dic[sizes[i]] = [[], [], []]
    return dic


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

def process_data(files_parquet):
    dic = {}
    average_event_size = []
    
    for f in tqdm(files_parquet):
        df = pd.read_parquet(f)
        
        df['time_diff'] = df['time_diff']*1.0
        sizes = np.unique(np.array((np.unique(df['bid_sz_00'].to_numpy())).tolist() + (np.unique(df['ask_sz_00'].to_numpy())).tolist()))
        sizes.sort()
        dic = dico_queue_size(sizes, dic)
        
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
                
    return dic, average_event_size

def calculate_intensities(dic, average_event_size, threshold_trade=1000, threshold=100):
    
    intensities = dict(sorted(dic.items()))
    intensities = filtrage(intensities, 30, threshold=100)
    
    Add, Cancel, Trade = [], [], []
    sizes_add, sizes_cancel, sizes_trade = [], [], []
    quarter_add, quarter_cancel, quarter_trade = [], [], []
    
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
                
    return sizes_add, Add, quarter_add, sizes_cancel, Cancel, quarter_cancel, sizes_trade, Trade, quarter_trade, np.mean(average_event_size)

def plot_intensities(sizes_add, Add, quarter_add, sizes_cancel, Cancel, quarter_cancel, 
                    sizes_trade, Trade, quarter_trade, mean_event_size):
    plt.figure(figsize=(10, 6))
    
    plt.plot(sizes_add/mean_event_size, Add, color='blue', label='Add')
    plt.fill_between(sizes_add/mean_event_size,
                    np.array(Add) - 1.96*np.array(quarter_add),
                    np.array(Add) + 1.96*np.array(quarter_add),
                    color='blue', alpha=0.2)
                    
    plt.plot(sizes_cancel/mean_event_size, Cancel, color='green', label='Cancel')
    plt.fill_between(sizes_cancel/mean_event_size,
                    np.array(Cancel) - 1.96*np.array(quarter_cancel),
                    np.array(Cancel) + 1.96*np.array(quarter_cancel),
                    color='green', alpha=0.2)
                    
    plt.plot(sizes_trade/mean_event_size, Trade, color='red', label='Trade')
    plt.fill_between(sizes_trade/mean_event_size,
                    np.array(Trade) - 1.96*np.array(quarter_trade),
                    np.array(Trade) + 1.96*np.array(quarter_trade),
                    color='red', alpha=0.2)
                    
    plt.title("Intensité par Queue size avec intervalles d'incertitude")
    plt.xlabel('Size (par Mean Event Size)')
    plt.ylabel('Intensity (num par sec)')
    plt.legend()
    
    plt.savefig('intensity_plot.png')
    plt.close()

if __name__ == "__main__":
    files_parquet = glob.glob(os.path.join("/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ/LCID_filtered", "*PL.parquet"))
    
    dic, average_event_size = process_data(files_parquet)
    
    
    results = calculate_intensities(dic, average_event_size)
    plot_intensities(*results)