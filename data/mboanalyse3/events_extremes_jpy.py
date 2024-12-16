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

# %%
df = pd.read_csv('/Volumes/T9/CSV_GOOGL_PL_Analyse.csv')
df = df.sort_values(by = 'max_proba', axis = 0)


# %%
df.head(100)

# %%
fig = go.Figure()

fig.add_trace(go.Scatter(x = df['variance_proba'], y = df['seuil'], mode = 'markers'))
fig.update_layout(
    title=('Graphe'),
    #title_x=0.5,
    #title_y=0.15,
    yaxis_title='variance',
    xaxis_title='seuil',
    #margin=dict(t=50, b=110),
    showlegend=True
)
fig.show()

# %%

# %%
