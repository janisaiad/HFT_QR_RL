import pandas as pd # ca on va devoir bcp l'uiliser
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)
from tqdm import tqdm
import os
import glob
from collections import Counter


def processing(file, output):
    import pandas as pd
    actif = 'GOOGL'
    limite = 0
    df = pd.read_csv(file)
    df = df[df['symbol'] == actif]
    df = df[df['depth'] == limite]
    print(len(df))
    df = df[df['size'] > 0]
    print(len(df))
    df['ts_event'] = pd.to_datetime(df['ts_event'], errors='coerce')
    df = df[(df['ts_event'].dt.hour > 14) | ((df['ts_event'].dt.hour == 14) & (df['ts_event'].dt.minute >= 30)) & (df['ts_event'].dt.hour < 19)]
    df = df[df['side'].isin(['A','B'])]
    print(len(df))
    df_ = df[['ts_event','action', 'side', 'size', 'price',f'bid_px_0{limite}', f'ask_px_0{limite}', f'bid_sz_0{limite}', f'ask_sz_0{limite}',f'bid_ct_0{limite}', f'ask_ct_0{limite}',f'bid_px_0{limite+1}', f'ask_px_0{limite+1}', f'bid_sz_0{limite+1}', f'ask_sz_0{limite+1}',f'bid_ct_0{limite+1}', f'ask_ct_0{limite+1}', f'bid_px_00', f'ask_px_00', f'bid_sz_00', f'ask_sz_00',f'bid_ct_00', f'ask_ct_00']]
    df_['ts_event'] = pd.to_datetime(df_['ts_event'], errors='coerce')
    df_['time_diff'] = df_['ts_event'].diff().dt.total_seconds()
    print(len(df))
    df_['price_same'] = np.where(df['side'] == 'A', df[f'ask_px_0{limite}'],df[f'bid_px_0{limite}'])
    df_['price_opposite'] = np.where(df['side'] == 'A', df[f'bid_px_0{limite}'], df[f'ask_px_0{limite}'])
    df_['size_same'] = np.where(df['side'] == 'A', df[f'ask_sz_0{limite}'],df[f'bid_sz_0{limite}'])
    df_['size_opposite'] = np.where(df['side'] == 'A', df[f'bid_sz_0{limite}'], df[f'ask_sz_0{limite}'])
    df_['nb_ppl_same'] = np.where(df['side'] == 'A', df[f'ask_ct_0{limite}'],df[f'bid_ct_0{limite}'])
    df_['nb_ppl_opposite'] = np.where(df['side'] == 'A', df[f'bid_ct_0{limite}'], df[f'ask_ct_0{limite}'])
    #df_.drop(columns=[f'bid_px_0{limite}', f'ask_px_0{limite}', f'bid_sz_0{limite}', f'ask_sz_0{limite}', f'bid_ct_0{limite}', f'ask_ct_0{limite}'], axis=1, inplace=True)
    df_['diff_price'] = df_['price'].diff()
    df_['Mean_price_diff'] = df_['diff_price'].rolling(window=10).mean().shift(1)
    df_['imbalance'] = (df_[f'ask_sz_0{limite}']-df_[f'bid_sz_0{limite}'])/(df_[f'ask_sz_0{limite}']+df_[f'bid_sz_0{limite}'])
    df_['ts_event'] = pd.to_datetime(df_['ts_event'], errors='coerce')
    df_['time_diff'] = df_['ts_event'].diff().dt.total_seconds()
    df_['indice'] = range(len(df_))
    df_[f'bid_sz_0{limite}_diff'] = df_[f'bid_sz_0{limite}'].diff()
    df_[f'ask_sz_0{limite}_diff'] = df_[f'ask_sz_0{limite}'].diff()
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


    # Appliquer 'OK' ou 'NOK' en fonction des conditions respectées
    df_['status'] = np.where(condition_T | condition_A | condition_C, 'OK', 'NOK')
    #df_ = df_['status' == 'OK']

    df_['new_limite'] = np.where((df_[f'bid_px_0{limite}'].diff() > 0) | (df_[f'ask_px_0{limite}'].diff() > 0), 'new_limite', 'n')

    #df_ = df_[['action','side','size','bid_sz_00','ask_sz_00','status']]#,'status_N','status_diff']]
    df_.head(1)

    import pandas as pd

    df_ = df_.reset_index(drop=True)

    new_rows = []
    lims = []
    i = 0
    total_rows = len(df_)
    while i < total_rows-1:

        # if i % (total_rows//10) == 0:
        #     print(f"Progression du code de merde: {int((i/total_rows)*100)}%")
        current_timestamp = df_.loc[i, 'ts_event']
        indices_group = [i]
        j = i+1

        while j < total_rows and df_.loc[j, 'ts_event'] == current_timestamp:
            indices_group.append(j)
            j += 1

        if len(indices_group) > 1:
            events_group = df_.iloc[indices_group]
            all_trades_then_cancel = all(events_group['action'].iloc[:-1] == 'T') and events_group['action'].iloc[-1] == 'C'
            all_trades = all(events_group['action'] == 'T')
            complex_trades_cancels = (
                events_group['action'].iloc[-1] == 'C' and
                all(
                    events_group['action'].iloc[start:k].eq('T').all()
                    for k, action in enumerate(events_group['action'])
                    if action == 'C' and (start := events_group['action'].iloc[:k].last_valid_index()) is not None
                )
            )
            no_new_limite = 'new_limite' not in events_group['new_limite'].values

            complex_trades_cancels_not_ended_by_cancel = (
                all(
                    events_group['action'].iloc[start:k].eq('T').all()
                    for k, action in enumerate(events_group['action'])
                    if action == 'C' and (start := events_group['action'].iloc[:k].last_valid_index()) is not None
                )
            )

            if all_trades_then_cancel and complex_trades_cancels and no_new_limite:
                total_size = events_group.loc[events_group['action'] == 'T', 'size'].sum()
                new_row = events_group.iloc[-1].copy()
                new_row['size'] = total_size
                new_row['action'] = 'T'
                new_rows.append(new_row)
                lims.extend((np.unique(events_group['new_limite'].to_numpy())))

            elif complex_trades_cancels and no_new_limite:
                total_size = events_group.loc[events_group['action'] == 'T', 'size'].sum()
                new_row = events_group.iloc[-1].copy()
                new_row['size'] = total_size
                new_row['action'] = 'T'
                new_rows.append(new_row)
                lims.extend((np.unique(events_group['new_limite'].to_numpy())))

            elif all_trades_then_cancel and not no_new_limite:
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

            elif complex_trades_cancels and not no_new_limite:
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

            elif all_trades:
                if df_.iloc[j]['new_limite'] == 'new_limite':
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

            elif complex_trades_cancels_not_ended_by_cancel:
                if df_.iloc[j]['new_limite'] == 'new_limite':
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

            else:
                new_rows.extend(events_group.to_dict(orient='records'))

            i = j
        else:
            new_rows.append(df_.iloc[i].to_dict())
            i += 1

    standardized_series_list = []
    for item in new_rows:
        if isinstance(item, dict):
            standardized_series_list.append(pd.Series(item))
        elif isinstance(item, pd.Series):
            standardized_series_list.append(item)
        else:
            raise ValueError("DAAAAAAIIIMMMM")

    df__ = pd.concat(standardized_series_list, axis=1).T.reset_index(drop=True)

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
    df__.loc[df__['new_limite'] == 'new_limite', 'time_diff'] = np.nan
    df__ = df__[df__['time_diff']>0]
    df__ = df__[df__['time_diff'] != np.nan]
    df_final = df__[df__['status'] != 'NOK']
    df__['Mean_price_diff'] = df__['price'].shift(-50) - df__['price']#df__['diff_price'].rolling(window=50).mean().shift(1)
    print(df.head)
    df__['imbalance'] = -(df__[f'ask_sz_0{limite}']-df__[f'bid_sz_0{limite}'])/(df__[f'ask_sz_0{limite}']+df__[f'bid_sz_0{limite}'])
    df['imbalance'] = df['imbalance'].shift()
    df['Mean_price_diff'] = df['price'].shift(-50) - df['price']
    df = df.dropna()
    df = df[:-100]
    #df__['imbalance'] = (df__['size_same']-df__['size_opposite'])/(df__['size_same']+df__['size_opposite'])
    df__.to_csv(output+file[-29:], index = False)
