import { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer
} from 'recharts';

interface PriceData {
  timestamp: string;
  price: number;
}

interface OrderBookLevel {
  price: number;
  bids: number;
  asks: number;
}

export function FinancialDashboard() {
  const [priceData, setPriceData] = useState<PriceData[]>([]);
  const [orderBookData, setOrderBookData] = useState<OrderBookLevel[]>([]);
  const [historicalHeatmap, setHistoricalHeatmap] = useState<number[][]>([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        console.log("Fetching data from backend...");
        const response = await fetch('http://localhost:5000/api/market-data', {
          method: 'GET',
          headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
          },
        });
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log("Received data:", data);
        
        setPriceData(data.priceData);
        setOrderBookData(data.orderBook);
        setHistoricalHeatmap(data.heatmap);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData(); // Premier appel immédiat
    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, []);

  const getHeatmapColor = (value: number): string => {
    const hue = ((1 - value) * 120).toString(10);
    return `hsl(${hue}, 70%, 50%)`;
  };
  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">Dashboard Financier</h1>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Graphique des prix */}
          <div className="bg-white p-6 rounded-xl shadow-lg">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Prix en temps réel</h2>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={priceData}>
                  <XAxis dataKey="timestamp" />
                  <YAxis />
                  <Tooltip />
                  <Line 
                    type="monotone" 
                    dataKey="price" 
                    stroke="#4f46e5" 
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Carnet d'ordres */}
          <div className="bg-white p-6 rounded-xl shadow-lg">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Carnet d'ordres</h2>
            <table className="w-full">
              <thead>
                <tr className="text-left text-gray-600">
                  <th className="p-2">Prix</th>
                  <th className="p-2">Achats</th>
                  <th className="p-2">Ventes</th>
                </tr>
              </thead>
              <tbody>
                {orderBookData.map((level, i) => (
                  <tr key={i} className="border-t">
                    <td className="p-2 text-gray-900">{level.price.toFixed(2)}</td>
                    <td className="p-2">
                      <div className="bg-green-100 rounded-full h-4">
                        <div
                          className="bg-green-500 rounded-full h-full transition-all"
                          style={{ width: `${level.bids * 10}%` }}
                        />
                      </div>
                    </td>
                    <td className="p-2">
                      <div className="bg-red-100 rounded-full h-4">
                        <div
                          className="bg-red-500 rounded-full h-full transition-all"
                          style={{ width: `${level.asks * 10}%` }}
                        />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Carte thermique */}
          <div className="bg-white p-6 rounded-xl shadow-lg col-span-2">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Carte thermique historique</h2>
            <div className="grid grid-cols-10 gap-1">
              {historicalHeatmap.map((row, i) => (
                <div key={i} className="flex gap-1">
                  {row.map((value, j) => (
                    <div
                      key={`${i}-${j}`}
                      className="w-8 h-8 rounded transition-colors"
                      style={{ backgroundColor: getHeatmapColor(value) }}
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
