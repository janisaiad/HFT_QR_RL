import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm  # type: ignore

"""
Définition de la classe Queue pour représenter une file d'attente avec un prix et une taille.
"""
@dataclass
class Queue:
    price: float  # Prix de la file d'attente
    size: int  # Taille de la file d'attente

"""
Définition de l'énumération QType pour représenter les types de file d'attente : ASK et BID.
"""
class QType(Enum):
    ASK = 1  # Type ASK
    BID = 2  # Type BID

"""
Définition de la classe OrderBook pour représenter un carnet d'ordres avec un état, un nombre de niveaux de prix et un prix de référence.
"""
@dataclass
class OrderBook:
    state: List[Queue]  # État du carnet d'ordres
    k: int  # Nombre de niveaux de prix
    p_ref: float  # Prix de référence

    def get_best(self) -> Tuple[float, float]:
        """
        Obtient les meilleurs prix bid et ask.

        :return: Tuple contenant le meilleur prix bid et le meilleur prix ask
        """
        k = self.k
        bid = self.state[k - 1].price if self.state[k - 1].size > 0 else self.state[k - 2].price
        ask = self.state[k].price if self.state[k].size > 0 else self.state[k + 1].price
        return bid, ask

    def update_state(self, lambda_: List[List[float]], mju: List[List[float]], stf: int, Cbound: int, delta: float, H: float):
        """
        Met à jour l'état du carnet d'ordres avec des intensités calibrées.

        :param lambda_: Taux d'arrivée des ordres calibrés
        :param mju: Taux d'annulation des ordres calibrés
        :param stf: Facteur de mise à l'échelle du temps
        :param Cbound: Limite supérieure pour la taille de la file d'attente
        :param delta: Valeur delta pour la diminution de la taille de la file d'attente
        :param H: Limite supérieure pour le flux entrant
        """
        for i in range(self.k * 2):
            size = self.state[i].size
            # Intensités calibrées selon la profondeur du carnet
            depth_factor = np.exp(-0.1 * abs(i - self.k))  # Décroissance exponentielle avec la profondeur
            
            # Taux d'arrivée ajusté selon la profondeur et la taille actuelle
            arrival_intensity = lambda_[i % self.k][0] * depth_factor * (1 - size/Cbound)
            s_lambda = np.random.poisson(arrival_intensity * stf)
            
            # Taux d'annulation proportionnel à la taille actuelle
            cancel_intensity = mju[i % self.k][0] * (size/Cbound) * depth_factor
            s_mju = np.random.poisson(cancel_intensity * stf)
            
            proposed_change = s_lambda - s_mju

            # Assumption 1: Negative individual drift pour les grandes queues
            if size > Cbound:
                proposed_change -= delta * (size/Cbound)

            # Assumption 2: Bound on the incoming flow avec scaling dynamique
            total_incoming_flow = sum(
                np.random.poisson(lambda_[j % self.k][0] * np.exp(-0.1 * abs(j)) * stf) 
                for j in range(-self.k, self.k) if j != 0
            )
            
            if total_incoming_flow > H:
                scale_factor = H / total_incoming_flow
                proposed_change = int(proposed_change * scale_factor)

            # Régénération si nécessaire
            if size + proposed_change <= 0:
                self.state[i].size = self.regenerate_queue(i)
            else:
                self.state[i].size = min(Cbound, max(0, size + proposed_change))

    def regenerate_queue(self, i: int) -> int:
        """
        Régénère une queue vide avec une taille initiale calibrée.
        
        :param i: Index de la queue
        :return: Nouvelle taille de la queue
        """
        depth_factor = np.exp(-0.1 * abs(i - self.k))
        base_size = 5  # Taille de base pour la régénération
        return int(base_size * depth_factor)

def init_order_book(p_ref: float, k: int, tick_size: float) -> OrderBook:
    """
    Initialise un carnet d'ordres avec des tailles calibrées.

    :param p_ref: Prix de référence
    :param k: Nombre de niveaux de prix
    :param tick_size: Taille du tick
    :return: Carnet d'ordres initialisé
    """
    p_lowest = p_ref - tick_size/2 - tick_size * k
    state = []
    for i in range(k * 2):
        depth_factor = np.exp(-0.1 * abs(i - k))
        initial_size = int(50 * depth_factor)  # Taille initiale calibrée selon la profondeur
        state.append(Queue(p_lowest + i * tick_size, initial_size))
    return OrderBook(state, k, p_ref)

def get_calibrated_intensities(k: int) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Génère des intensités calibrées pour le modèle.

    :param k: Nombre de niveaux de prix
    :return: Tuple des intensités lambda et mju calibrées
    """
    base_lambda = 100  # Intensité de base pour les arrivées
    base_mju = 80     # Intensité de base pour les annulations
    
    lambda_ = [[base_lambda * np.exp(-0.1 * i)] for i in range(k)]
    mju = [[base_mju * np.exp(-0.05 * i)] for i in range(k)]
    
    return lambda_, mju

# Paramètres optimisés
params = {
    "k": 10,
    "p_ref": 100.0,
    "tick_size": 0.01,
    "stf": 1,
    "simulation_time": 60000,
    "Cbound": 1000,
    "delta": 0.5,
    "H": 500
}

# Initialisation avec intensités calibrées
lambda_, mju = get_calibrated_intensities(params["k"])
order_books = []
times = []

# Simulation
lob = init_order_book(params["p_ref"], params["k"], params["tick_size"])
for t in tqdm(range(0, params["simulation_time"], params["stf"]), desc="Simulation Progress"):
    lob.update_state(lambda_, mju, params["stf"], params["Cbound"], params["delta"], params["H"])
    if t % 1000 == 0:
        order_books.append(OrderBook([Queue(q.price, q.size) for q in lob.state], lob.k, lob.p_ref))
        times.append(t)

# Visualisation
def plot_order_book_heatmap(order_books: List[OrderBook], times: List[int]):
    prices = sorted(set(q.price for lob in order_books for q in lob.state))
    heatmap_data = np.zeros((len(prices), len(times)))
    price_to_index = {price: idx for idx, price in enumerate(prices)}

    for t_idx, lob in enumerate(order_books):
        for q in lob.state:
            price_idx = price_to_index[q.price]
            heatmap_data[price_idx, t_idx] = q.size

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=times,
        y=prices,
        colorscale='Viridis'
    ))

    fig.update_layout(
        title="Order Book Depth Heatmap",
        xaxis_title="Time",
        yaxis_title="Price",
        yaxis=dict(type='category')
    )

    fig.show()

plot_order_book_heatmap(order_books, times)