import pandas as pd # ca on va devoir bcp l'uiliser
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)
from tqdm import tqdm
import os
import glob



#### Tout ce code pour nasdaq

def processing(file, output):
    
    
    
    import pandas as pd
    actif = 'LCID'
    limite = 0
    df = pd.read_parquet(file)
    
    
    
    df = df[df['symbol'] == actif] # on ne garde que l'actif
    df = df[df['depth'] == limite] # on ne garde que la limite
    df['ts_event'] = pd.to_datetime(df['ts_event'], errors='coerce') # on convertit la colonne ts_event en datetime pour les calculs, coerce f
    df = df[(df['ts_event'].dt.hour >= 14) & (df['ts_event'].dt.hour < 19)] # on ne garde que les trades entre 14h et 19h
    df = df[df['side'].isin(['A','B'])] # on ne garde que les Ask et Bid, car il y a des actions qui n'ont pas d'impact (exemple un mec qui s'est placé en trade mais y'a pas eu d'impact car le trade n'a pas eu lieu)

    df_ = df[['ts_event','action', 'side', 'size', 'price',f'bid_px_0{limite}', f'ask_px_0{limite}', f'bid_sz_0{limite}', f'ask_sz_0{limite}',f'bid_ct_0{limite}', f'ask_ct_0{limite}',f'bid_px_0{limite+1}', f'ask_px_0{limite+1}', f'bid_sz_0{limite+1}', f'ask_sz_0{limite+1}',f'bid_ct_0{limite+1}', f'ask_ct_0{limite+1}']] # pour limite et mid price
    df_['ts_event'] = pd.to_datetime(df_['ts_event'], errors='coerce') # repasser en datetime
    df_['time_diff'] = df_['ts_event'].diff().dt.total_seconds() # timediff en secondes, on doit transformer ça en float et faire *1.0

    df_['price_same'] = np.where(df['side'] == 'A', df[f'ask_px_0{limite}'],df[f'bid_px_0{limite}']) # price de la même cote
    df_['price_opposite'] = np.where(df['side'] == 'A', df[f'bid_px_0{limite}'], df[f'ask_px_0{limite}']) # price de l'opposée
    df_['size_same'] = np.where(df['side'] == 'A', df[f'ask_sz_0{limite}'],df[f'bid_sz_0{limite}']) # size de la même cote
    df_['size_opposite'] = np.where(df['side'] == 'A', df[f'bid_sz_0{limite}'], df[f'ask_sz_0{limite}']) # size de l'opposée
    df_['nb_ppl_same'] = np.where(df['side'] == 'A', df[f'ask_ct_0{limite}'],df[f'bid_ct_0{limite}']) # bid ct est le nombre de personnes à la même
    df_['nb_ppl_opposite'] = np.where(df['side'] == 'A', df[f'bid_ct_0{limite}'], df[f'ask_ct_0{limite}']) # nombre de personnes qui ont l'opposée
    #df_.drop(columns=[f'bid_px_0{limite}', f'ask_px_0{limite}', f'bid_sz_0{limite}', f'ask_sz_0{limite}', f'bid_ct_0{limite}', f'ask_ct_0{limite}'], axis=1, inplace=True)
    df_['diff_price'] = df_['price'].diff() # diff price
    df_['Mean_price_diff'] = df_['diff_price'].rolling(window=10).mean().shift(1) # moyenne mobile de la diff price
    df_['imbalance'] = (df_[f'ask_sz_0{limite}']-df_[f'bid_sz_0{limite}'])/(df_[f'ask_sz_0{limite}']+df_[f'bid_sz_0{limite}']) # imbalance
    df_['ts_event'] = pd.to_datetime(df_['ts_event'], errors='coerce') # repasser en datetime
    df_['time_diff'] = df_['ts_event'].diff().dt.total_seconds() # timediff en secondes
    df_['indice'] = range(len(df_)) # indice
    df_[f'bid_sz_0{limite}_diff'] = df_[f'bid_sz_0{limite}'].diff() # diff size bid
    df_[f'ask_sz_0{limite}_diff'] = df_[f'ask_sz_0{limite}'].diff() # diff size ask
    
    
    condition_T = (
        (df_['action'] == 'T') &
        (
            ((df_['side'] == 'B') & (df_[f'bid_sz_0{limite}_diff'] == -df_['size'])) |
            ((df_['side'] == 'A') & (df_[f'ask_sz_0{limite}_diff'] == -df_['size']))
        )
    )

    # Condition pour 'A'
    condition_A = (
        (df_['action'] == 'A') &
        (
            ((df_['side'] == 'B') & (df_[f'bid_sz_0{limite}_diff'] == df_['size'])) |
            ((df_['side'] == 'A') & (df_[f'ask_sz_0{limite}_diff'] == df_['size']))
        )
    )

    # Condition pour 'C'
    condition_C = (
        (df_['action'] == 'C') &
        (
            ((df_['side'] == 'B') & (df_[f'bid_sz_0{limite}_diff'] == -df_['size'])) |
            ((df_['side'] == 'A') & (df_[f'ask_sz_0{limite}_diff'] == -df_['size']))
        )
    )
    # tout ça permet de faire correspondre afin de savoir si on a une nouvelle limite ou pas

    # Appliquer 'OK' ou 'NOK' en fonction des conditions respectées
    df_['status'] = np.where(condition_T | condition_A | condition_C, 'OK', 'NOK')
    # environ 4/5 des données sont NOK, mais les résultats sont plus logiques
    #df_ = df_['status' == 'OK']

    df_['new_limite'] = np.where((df_[f'bid_px_0{limite}'].diff() > 0) | (df_[f'ask_px_0{limite}'].diff() > 0), 'new_limite', 'n') # correspond à une nouvelle limite
    # pour détercter une nouvelle limite, on regarde les conditoins A B C, mais avec le fait stylisé bizarre des trades c'est un peu bizarre
    # on peut aussi regarder les différences de prix, car il y a eu un cancel et il a pas été pris en compte

    #df_ = df_[['action','side','size','bid_sz_00','ask_sz_00','status']]#,'status_N','status_diff']]
    df_.head(1)

    import pandas as pd

    df_ = df_.reset_index(drop=True) # reset index
    # on crée un new dataset csv à la main, quand on ajoute une ligne c'est un format différent de transformer une ligne
    new_rows = []
    lims = []
    i = 0
    total_rows = len(df_)
    while i < total_rows-1:

        # if i % (total_rows//10) == 0:
        #     print(f"Progression du code de merde: {int((i/total_rows)*100)}%")
        current_timestamp = df_.loc[i, 'ts_event'] # la ligne i, on regarde la data
        indices_group = [i]
        j = i+1

        while j < total_rows and df_.loc[j, 'ts_event'] == current_timestamp: # on regarde les lignes suivantes
            indices_group.append(j) # 
            j += 1
        # grand max linéaire

        if len(indices_group) > 1: # au moins 2 events au même moment
            events_group = df_.iloc[indices_group]
            all_trades_then_cancel = all(events_group['action'].iloc[:-1] == 'T') and events_group['action'].iloc[-1] == 'C' # que des trades puis cancel
            all_trades = all(events_group['action'] == 'T') # que des trades
            complex_trades_cancels = (
                events_group['action'].iloc[-1] == 'C' and
                all(
                    events_group['action'].iloc[start:k].eq('T').all()
                    for k, action in enumerate(events_group['action'])
                    if action == 'C' and (start := events_group['action'].iloc[:k].last_valid_index()) is not None
                )
            ) # une liste de trades qui sont suivis d'un cancel, le dernier est cancel
            no_new_limite = 'new_limite' not in events_group['new_limite'].values # s'il y a une nouvelle limite ou pas, il faut traiter différemment

            complex_trades_cancels_not_ended_by_cancel = (
                all(
                    events_group['action'].iloc[start:k].eq('T').all()
                    for k, action in enumerate(events_group['action'])
                    if action == 'C' and (start := events_group['action'].iloc[:k].last_valid_index()) is not None
                )
            ) # une liste de trades qui sont suivis d'un cancel, le dernier n'est pas cancel

            if all_trades_then_cancel and complex_trades_cancels  and no_new_limite: # erreur due pas à une new limite et qui est ue à un cancel '''le and ne sert à rien mais on n'est pas sûr'''
                total_size = events_group.loc[events_group['action'] == 'T', 'size'].sum()
                new_row = events_group.iloc[-1].copy()
                new_row['size'] = total_size
                new_row['action'] = 'T'
                new_rows.append(new_row)
                lims.extend((np.unique(events_group['new_limite'].to_numpy())))
                # on transforme tout en un seul trade, très souvent la taille du cancel c'est la taille de l'avant dernier trade, trop bizarre
            # normalement ne sert à rien
            elif complex_trades_cancels and no_new_limite: # censé être le même qu'au dessus
                total_size = events_group.loc[events_group['action'] == 'T', 'size'].sum()
                new_row = events_group.iloc[-1].copy()
                new_row['size'] = total_size
                new_row['action'] = 'T'
                new_rows.append(new_row)
                lims.extend((np.unique(events_group['new_limite'].to_numpy())))
            # on observe aussi qu'on a un cancel par limite passée, si 3 limites passées, 3 cancels
            # cas ultra chiant
            # on a une nouvelle limite et un cancel
            elif all_trades_then_cancel and not no_new_limite:
                
                limite_ = events_group['new_limite'].values
                bonne_limite = [0]+[l for l in range(len(limite_)) if limite_[l] == 'new_limite'] # on regarde le smoments new limite
                # on veut calculer la taille des trades entre les new limites
                #
                for k in range(len(bonne_limite) - 1): # les moments où il y a une nouvelle limite, on a plein de trades cancel
                    start_index = bonne_limite[k] #ce qu'il se passe entre l'ancienne nouvelle limite et la nouvelle limite
                    end_index = bonne_limite[k + 1]
                    total_size = events_group[start_index:end_index].query("action == 'T'")['size'].sum() #taille totale des trades
                    new_row = events_group.iloc[-1].copy() # y'a que le cancel du bon côté
                    new_row['size'] = total_size
                    new_row['action'] = 'T' # on remplace par un trade
                    new_row['new_limite'] = 'limite_épuisée' # on remplace par limite épuisée
                    if new_row['side'] == 'B':
                        new_row[f'bid_sz_0{limite}'] = 0 # on remplace car on atteint
                    elif new_row['side'] == 'A':
                        new_row[f'ask_sz_0{limite}'] = 0 # on remplace car on atteint la limite
                    new_rows.append(new_row)
                total_size = events_group.loc[bonne_limite[-1]:].query("action == 'T'")['size'].sum() # on prend en compte les trades sur la nouvelle limite (qui a changé)
                new_row = events_group.iloc[-1].copy()
                new_row['size'] = total_size
                new_row['action'] = 'T'
                new_row['new_limite'] = 'new_limite' # on marque ça comme une nouvelle limite pour traquer les cancels
                new_rows.append(new_row)
                
                
            #exactement pareil avec le cas d'avoir plusieurs limites passées
            elif complex_trades_cancels and not no_new_limite: # cas où on a une nouvelle limite et que des trades et un cancel
                limite_ = events_group['new_limite'].values
                bonne_limite = [0]+[l for l in range(len(limite_)) if limite_[l] == 'new_limite']
                for k in range(len(bonne_limite) - 1):
                    start_index = bonne_limite[k]
                    end_index = bonne_limite[k + 1]
                    total_size = events_group[start_index:end_index].query("action == 'T'")['size'].sum()
                    new_row = events_group.iloc[-1].copy()
                    new_row['size'] = total_size
                    new_row['action'] = 'T'
                    new_row['new_limite'] = 'limite_épuisée'
                    if new_row['side'] == 'B':
                        new_row[f'bid_sz_0{limite}'] = 0
                    elif new_row['side'] == 'A':
                        new_row[f'ask_sz_0{limite}'] = 0
                    new_rows.append(new_row)
                total_size = events_group.loc[bonne_limite[-1]:].query("action == 'T'")['size'].sum()
                new_row = events_group.iloc[-1].copy()
                new_row['size'] = total_size
                new_row['action'] = 'T'
                new_row['new_limite'] = 'new_limite'
                new_rows.append(new_row)

            elif all_trades: # cas où on a que des trades
                if df_.iloc[j]['new_limite'] == 'new_limite': # si la dernière est une nouvelle limite, normalement on est censé avoir un cancel
                    total_size = events_group.loc[events_group['action'] == 'T', 'size'].sum()
                    new_row = events_group.iloc[-1].copy()
                    new_row['size'] = 0
                    new_row['action'] = 'T'
                    new_rows.append(new_row)
                else:
                    total_size = events_group.loc[events_group['action'] == 'T', 'size'].sum()
                    new_row = events_group.iloc[-1].copy()
                    new_row['size'] = total_size
                    new_row['action'] = 'T'
                    new_rows.append(new_row)

            elif complex_trades_cancels_not_ended_by_cancel: # cas où on a une liste de trades qui sont suivis d'un cancel, le dernier n'est pas cancel, cancel entre les deux dû à des erreurs
                if df_.iloc[j]['new_limite'] == 'new_limite': # si la dernière est une nouvelle limite, normalement on est censé avoir un cancel
                    total_size = events_group.loc[events_group['action'] == 'T', 'size'].sum()
                    new_row = events_group.iloc[-1].copy()
                    new_row['size'] = 0
                    new_row['action'] = 'T'
                    new_rows.append(new_row)
                else: # aucun event en même temps
                    total_size = events_group.loc[events_group['action'] == 'T', 'size'].sum()
                    new_row = events_group.iloc[-1].copy()
                    new_row['size'] = total_size
                    new_row['action'] = 'T'
                    new_rows.append(new_row)

            else:
                new_rows.extend(events_group.to_dict(orient='records'))

            i = j # complexité linéaire
        else: # au cas ou
            new_rows.append(df_.iloc[i].to_dict())
            i += 1
    # les event.iloc sont renvoyés en dictionnaire
    standardized_series_list = [] # on transforme en série pour standardiser
    for item in new_rows:
        if isinstance(item, dict):
            standardized_series_list.append(pd.Series(item)) # on transforme en série pour standardiser
        elif isinstance(item, pd.Series):
            standardized_series_list.append(item)
        else:
            raise ValueError("DAAAAAAIIIMMMM")
    df__ = pd.concat(standardized_series_list, axis=1).T.reset_index(drop=True) # on concatene les séries

    
    # exactement les mêmes calculs qu'avant
    #df__['ts_diff'] = pd.to_datetime(df__['ts_event']).diff().dt.total_seconds()
    df__['price_middle'] = (df__['ask_px_00']+df__['bid_px_00'])/2
    df__['price_same'] = np.where(df__['side'] == 'A', df__[f'ask_px_0{limite}'],df__[f'bid_px_0{limite}'])
    df__['price_opposite'] = np.where(df__['side'] == 'A', df__[f'bid_px_0{limite}'], df__[f'ask_px_0{limite}'])
    df__['size_same'] = np.where(df__['side'] == 'A', df__[f'ask_sz_0{limite}'],df__[f'bid_sz_0{limite}'])
    df__['size_opposite'] = np.where(df__['side'] == 'A', df__[f'bid_sz_0{limite}'], df__[f'ask_sz_0{limite}'])
    df__['nb_ppl_same'] = np.where(df__['side'] == 'A', df__[f'ask_ct_0{limite}'],df__[f'bid_ct_0{limite}'])
    df__['nb_ppl_opposite'] = np.where(df__['side'] == 'A', df__[f'bid_ct_0{limite}'], df__[f'ask_ct_0{limite}'])
    #df__.drop(columns=[f'bid_px_0{limite}', f'ask_px_0{limite}', f'bid_sz_0{limite}', f'ask_sz_0{limite}', f'bid_ct_0{limite}', f'ask_ct_0{limite}'], axis=1, inplace=True)
    df__['diff_price'] = df__['price_middle'].diff()
    df__['time_diff'] = df__['ts_event'].diff().dt.total_seconds()
    df__['indice'] = range(len(df__))
    df__[f'bid_sz_0{limite}_diff'] = df__[f'bid_sz_0{limite}'].diff()
    df__[f'ask_sz_0{limite}_diff'] = df__[f'ask_sz_0{limite}'].diff()

    condition_T = (
        (df__['action'] == 'T') &
        (
            ((df__['side'] == 'B') & (df__[f'bid_sz_0{limite}_diff'] == -df__['size'])) |
            ((df__['side'] == 'A') & (df__[f'ask_sz_0{limite}_diff'] == -df__['size']))
        )
    )

    condition_A = (
        (df__['action'] == 'A') &
        (
            ((df__['side'] == 'B') & (df__[f'bid_sz_0{limite}_diff'] == df__['size'])) |
            ((df__['side'] == 'A') & (df__[f'ask_sz_0{limite}_diff'] == df__['size']))
        )
    )

    condition_C = (
        (df__['action'] == 'C') &
        (
            ((df__['side'] == 'B') & (df__[f'bid_sz_0{limite}_diff'] == -df__['size'])) |
            ((df__['side'] == 'A') & (df__[f'ask_sz_0{limite}_diff'] == -df__['size']))
        )
    )

    df__['status'] = np.where(condition_T | condition_A | condition_C, 'OK', 'NOK')
    df__.loc[df__['new_limite'] == 'new_limite', 'time_diff'] = np.nan # si on a une nouvelle limite, le middle price a bougé
    # ne pas enlever plus tard pour le faire sur un grille de mid price
    df__ = df__[df__['time_diff']>0] # on vire les times négatifs c'est les cas potentiellement pas bons
    df__ = df__[df__['time_diff'] != np.nan] # on vire les times nan
    df_final = df__[df__['status'] != 'NOK']
    df__['Mean_price_diff'] = df__['diff_price'].shift(-50) - df__['diff_price']#df__['diff_price'].rolling(window=50).mean().shift(1)
    
    # df__ = df__[df__[f'ask_sz_0{limite}']+df__[f'bid_sz_0{limite}'] != 0]
    df__['imbalance'] = -(df__[f'ask_sz_0{limite}']-df__[f'bid_sz_0{limite}'])/(df__[f'ask_sz_0{limite}']+df__[f'bid_sz_0{limite}'])
    
    
    #df__['imbalance'] = (df__['size_same']-df__['size_opposite'])/(df__['size_same']+df__['size_opposite'])
    df__.to_parquet(output) # removed index == false


if __name__ == "__main__":
    files = glob.glob(os.path.join('/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ/KHC', '*.parquet'))
    error_count = 0    
    for file in tqdm(files):
        output_path = os.path.join('/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ/KHC_filtered', os.path.basename(file)[:-8] + '_filtered.parquet')
       
        try:
            processing(file, output_path)
        except Exception as e:
            error_count += 1
            print(f"Error processing file {file}: {e}")
    print(f"Total number of errors: {error_count}")