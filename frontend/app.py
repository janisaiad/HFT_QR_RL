from flask import Flask, jsonify
from flask_cors import CORS
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

class MarketDataSimulator:
    def __init__(self):
        self.current_price = 100.0
        self.historical_prices = []
        self.historical_heatmap = []
    
    def update_price(self):
        """Mise à jour du prix avec un mouvement brownien"""
        self.current_price += np.random.normal(0, 0.5)
        timestamp = datetime.now().isoformat()
        self.historical_prices.append({
            'timestamp': timestamp,
            'price': self.current_price
        })
        # Garder seulement les 50 derniers points
        self.historical_prices = self.historical_prices[-50:]
        return self.historical_prices

    def generate_order_book(self):
        """Génère un carnet d'ordres simulé"""
        center_price = self.current_price
        prices = np.linspace(center_price - 5, center_price + 5, 10)
        
        return [{
            'price': float(price),
            'bids': float(np.random.exponential(5) * np.exp(-0.1 * abs(price - center_price))),
            'asks': float(np.random.exponential(5) * np.exp(-0.1 * abs(price - center_price)))
        } for price in prices]

    def update_heatmap(self):
        """Met à jour la carte thermique"""
        new_row = np.random.random(10)
        self.historical_heatmap.append(new_row.tolist())
        # Garder seulement les 20 dernières lignes
        self.historical_heatmap = self.historical_heatmap[-20:]
        return self.historical_heatmap

simulator = MarketDataSimulator()

@app.route('/api/market-data')
def get_market_data():
    price_data = simulator.update_price()
    order_book = simulator.generate_order_book()
    heatmap = simulator.update_heatmap()
    
    return jsonify({
        'priceData': price_data,
        'orderBook': order_book,
        'heatmap': heatmap
    })

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)