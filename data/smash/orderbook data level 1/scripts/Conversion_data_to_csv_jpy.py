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

# %% [markdown]
# # CONVERSION DE LA DATA EN CSV

# %%
import os
import zstandard as zstd
import json
import databento as dbn  
import pandas as pd

def decompress_zst(zst_path, extract_path):
    """Décompresse un fichier zst comme un boss."""
    with open(zst_path, 'rb') as zst_file:
        dctx = zstd.ZstdDecompressor()
        with open(extract_path, 'wb') as out_file:
            dctx.copy_stream(zst_file, out_file)
    print(f"Bam ! Fichier décompressé dans : {extract_path}")

def load_metadata(metadata_path):
    """Charge les métadonnées du fichier JSON, fastoche."""
    with open(metadata_path, 'r') as f:
        return json.load(f)

def load_symbology(symbology_path):
    """Charge la symbologie du fichier JSON, rien de ouf."""
    with open(symbology_path, 'r') as f:
        return json.load(f)
    
    
    
def visualize_orderbook(dbn_file, symbol):
    """Visualise le carnet d'ordres pour un symbole donné en utilisant @databento."""
    try:
        # On crée un DBNStore à partir du chemin du fichier
        store = dbn.DBNStore.from_file(dbn_file)
        
        # On récupère les données pour le symbole spécifié
        df = store.to_df( # Assurez-vous que c'est le bon schéma pour vos données
        )
    except FileNotFoundError:
        print(f"Erreur : Le fichier {dbn_file} n'a pas été trouvé.")
        return
    except Exception as e:
        print(f"Une erreur s'est produite lors de la lecture du fichier : {str(e)}")
        return
    
    # On vérifie si des données ont été récupérées
    if df.empty:
        print(f"Aucune donnée n'a été trouvée pour le symbole {symbol}")
        return
    # On filtre les données pour ne garder que celles correspondant au symbole spécifié
    df = df[df['symbol'] == symbol]
    
    # On vérifie à nouveau si des données ont été trouvées après le filtrage
    if df.empty:
        print(f"Aucune donnée n'a été trouvée pour le symbole {symbol}")
        return
    # On extrait les données pertinentes
    
    df['timestamp'] = pd.to_datetime(df['ts_event'], unit='ns')
    
def main():
    
    # TBBO
    # Le chemin vers le dossier avec tous les trucs dedans
    folder_path = "../DBEQ-20240924-4468SLPNAT"
    ext_path = "../Trades_unzip"
    # On décompresse le fichier zst s'il y en a un
    zst_files = [f for f in os.listdir(folder_path) if f.endswith('.zst')]
    for i in range (len(zst_files)):
        zst_path = os.path.join(folder_path, zst_files[i])
        extract_path = os.path.join(ext_path, zst_files[i][:-4])  # On enlève l'extension .zst
        decompress_zst(zst_path, extract_path)
    
    # On charge les métadonnées et la symbologie
    metadata = load_metadata(os.path.join(folder_path, "metadata.json"))
    symbology = load_symbology(os.path.join(folder_path, "symbology.json"))
    
    # On cherche le fichier DBN, ça devrait pas trop galère
    dbn_files = [f for f in os.listdir(folder_path) if f.endswith('.dbn')]
    if not dbn_files:
        print("Merde, pas de fichier DBN dans le dossier.")
        return
    
    dbn_file = os.path.join(folder_path, dbn_files[0])
    
    # On visualise le carnet d'ordres pour chaque symbole
    for symbol in symbology['symbols']:
        visualize_orderbook(dbn_file, symbol)
    
    
    # Pour les trades
    
    # Le chemin vers le dossier avec tous les trucs dedans
    folder_path = "../DBEQ-20240924-QA9WT8Y3V9"
    ext_path = "../TBBO_unzip"
    # On décompresse le fichier zst s'il y en a un
    zst_files = [f for f in os.listdir(folder_path) if f.endswith('.zst')]
    for i in range (len(zst_files)):
        zst_path = os.path.join(folder_path, zst_files[i])
        extract_path = os.path.join(ext_path, zst_files[i][:-4])  # On enlève l'extension .zst
        decompress_zst(zst_path, extract_path)
    
    # On charge les métadonnées et la symbologie
    metadata = load_metadata(os.path.join(folder_path, "metadata.json"))
    symbology = load_symbology(os.path.join(folder_path, "symbology.json"))
    
    # On cherche le fichier DBN, ça devrait pas trop galère
    dbn_files = [f for f in os.listdir(folder_path) if f.endswith('.dbn')]
    if not dbn_files:
        print("Merde, pas de fichier DBN dans le dossier.")
        return
    
    dbn_file = os.path.join(folder_path, dbn_files[0])
    
    # On visualise le carnet d'ordres pour chaque symbole
    for symbol in symbology['symbols']:
        visualize_orderbook(dbn_file, symbol)
        
    # Status
    
    # Le chemin vers le dossier avec tous les trucs dedans
    folder_path = "../DBEQ-20240924-WMXF48CXBP"
    ext_path = "../Status_unzip"
    # On décompresse le fichier zst s'il y en a un
    zst_files = [f for f in os.listdir(folder_path) if f.endswith('.zst')]
    for i in range (len(zst_files)):
        zst_path = os.path.join(folder_path, zst_files[i])
        extract_path = os.path.join(ext_path, zst_files[i][:-4])  # On enlève l'extension .zst
        decompress_zst(zst_path, extract_path)
    
    # On charge les métadonnées et la symbologie
    metadata = load_metadata(os.path.join(folder_path, "metadata.json"))
    symbology = load_symbology(os.path.join(folder_path, "symbology.json"))
    
    # On cherche le fichier DBN, ça devrait pas trop galère
    dbn_files = [f for f in os.listdir(folder_path) if f.endswith('.dbn')]
    if not dbn_files:
        print("Merde, pas de fichier DBN dans le dossier.")
        return
    
    dbn_file = os.path.join(folder_path, dbn_files[0])
    
    # On visualise le carnet d'ordres pour chaque symbole
    for symbol in symbology['symbols']:
        visualize_orderbook(dbn_file, symbol)

main()

# %%
ex_path = "../Trades_unzip"
dbn_files = [f for f in os.listdir(ex_path) if f.endswith('.dbn')]
df_list = []

for f in dbn_files:
    file_path = os.path.join(ex_path, f)
    store_TBBO_temp = dbn.DBNStore.from_file(file_path)
    df_TBBO_temp = store_TBBO_temp.to_df().reset_index()
    df_list.append(df_TBBO_temp)

df_Trades = pd.concat(df_list, ignore_index=True)

output_file_path = "../../Trade.csv"  # Chemin du fichier de sortie
df_Trades.to_csv(output_file_path, index=False)

print(f"DataFrame exporté avec succès en fichier CSV : {output_file_path}")


# %%
ex_path = "../TBBO_unzip"
dbn_files = [f for f in os.listdir(ex_path) if f.endswith('.dbn')]
df_list = []

for f in dbn_files:
    file_path = os.path.join(ex_path, f)
    store_TBBO_temp = dbn.DBNStore.from_file(file_path)
    df_TBBO_temp = store_TBBO_temp.to_df().reset_index()
    df_list.append(df_TBBO_temp)

df_TBBO = pd.concat(df_list, ignore_index=True)

output_file_path = "../../TBBO.csv"  # Chemin du fichier de sortie
df_TBBO.to_csv(output_file_path, index=False)

# %%
ex_path = "../Status_unzip"
dbn_files = [f for f in os.listdir(ex_path) if f.endswith('.dbn')]
df_list = []

for f in dbn_files:
    file_path = os.path.join(ex_path, f)
    store_TBBO_temp = dbn.DBNStore.from_file(file_path)
    df_TBBO_temp = store_TBBO_temp.to_df().reset_index()
    df_list.append(df_TBBO_temp)

df_status = pd.concat(df_list, ignore_index=True)

output_file_path = "../../Status.csv"  # Chemin du fichier de sortie
df_status.to_csv(output_file_path, index=False)

# %%

# %%
