import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    """
    Loads data from the CSV file generated by script.py.
    
    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    file_path = "databento/databento_equities_basic.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Run script.py first.")
    return pd.read_csv(file_path)

def plot_trading_volumes(df):
    """
    Creates a bar plot of trading volumes for each symbol.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data.
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(x=df.index, y='volume', data=df)
    plt.title("Trading Volume by Symbol")
    plt.xlabel("Symbols")
    plt.ylabel("Volume")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("databento/trading_volumes.png")
    plt.close()

def plot_closing_prices(df):
    """
    Creates a line plot of closing prices for each symbol.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data.
    """
    plt.figure(figsize=(12, 6))
    for symbol in df.columns:
        plt.plot(df.index, df[symbol], label=symbol)
    plt.title("Closing Prices by Symbol")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("databento/closing_prices.png")
    plt.close()

def main():
    """
    Main function that orchestrates the data visualization.
    """
    try:
        df = load_data()
        
        if df.empty:
            print("The DataFrame is empty. No visualization is possible.")
            return
        
        os.makedirs("databento", exist_ok=True)
        
        plot_trading_volumes(df)
        plot_closing_prices(df)
        
        print("Visualizations successfully created in the 'databento' folder.")
    except Exception as e:
        print(f"An error occurred while creating the visualizations: {str(e)}")
        
        

def plot_trading_volumes_terminal(df):
    """
    Crée un graphique en barres des volumes de trading pour chaque symbole, adapté pour le terminal.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données.
    """
    volumes = df['volume'].tolist()
    symbols = df.index.tolist()
    
    max_volume = max(volumes)
    bar_width = 20
    
    print("Volumes de trading par symbole:")
    for symbol, volume in zip(symbols, volumes):
        bar_length = int((volume / max_volume) * bar_width)
        bar = '█' * bar_length
        print(f"{symbol:<5} | {bar:<{bar_width}} {volume}")

def plot_closing_prices_terminal(df):
    """
    Crée un graphique en ligne des prix de clôture pour chaque symbole, adapté pour le terminal.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données.
    """
    print("\nPrix de clôture par symbole:")
    for symbol in df.columns:
        prices = df[symbol].tolist()
        min_price = min(prices)
        max_price = max(prices)
        price_range = max_price - min_price
        
        if price_range == 0:
            normalized_prices = [0] * len(prices)
        else:
            normalized_prices = [(p - min_price) / price_range for p in prices]
        
        chart_width = 40
        chart = ''
        for price in normalized_prices:
            position = int(price * (chart_width - 1))
            line = [' '] * chart_width
            line[position] = '█'
            chart += ''.join(line) + '\n'
        
        print(f"{symbol}:")
        print(chart)
        print(f"Min: {min_price:.2f}, Max: {max_price:.2f}\n")

def main():
    """
    Fonction principale qui orchestre la visualisation des données.
    """
    try:
        df = load_data()
        
        if df.empty:
            print("Le DataFrame est vide. Aucune visualisation n'est possible.")
            return
        
        os.makedirs("databento", exist_ok=True)
        
        plot_trading_volumes(df)
        plot_closing_prices(df)
        
        print("Visualisations créées avec succès dans le dossier 'databento'.")
        
        # Ajout des visualisations pour le terminal
        plot_trading_volumes_terminal(df)
        plot_closing_prices_terminal(df)
        
    except Exception as e:
        print(f"Une erreur s'est produite lors de la création des visualisations : {str(e)}")



if __name__ == "__main__":
    main()

