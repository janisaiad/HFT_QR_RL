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

# %%
from visu_imbalance import processing
import glob
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)
from tqdm import tqdm
from events_rares import visu

# %%
df = visu("/Volumes/T9/CSV_GOOGL_NASDAQ_PL", 'GOOGL')

# %%
df.head()

# %%
files_csv = glob.glob(os.path.join("/Volumes/T9/CSV_dezippe_nasdaq", "*.csv"))
output = '/Volumes/T9/CSV_GOOGL_LCID_PL2/'
for f in tqdm(range(0, len(files_csv))):
    processing(files_csv[f], output)

# %%
