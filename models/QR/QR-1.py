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
    theta: float = 0.7  # Probabilité de mouvement de prix quand une limite est consommée

    def get_best(self) -> Tuple[float, float]:
        """
        Obtient les meilleurs prix bid et ask.

        :return: Tuple contenant le meilleur prix bid et le meilleur prix ask
        """
        k = self.k
        bid = self.state[k - 1].price if self.state[k - 1].size > 0 else self.state[k - 2].price
        ask = self.state[k].price if self.state[k].size > 0 else self.state[k + 1].price
        return bid, ask

    def handle_empty_limit(self, index: int, is_bid: bool):
        """
        Gère le cas où une limite devient vide.
        
        :param index: Index de la limite vide
        :param is_bid: True si c'est une limite bid, False si c'est une limite ask
        """
        if np.random.random() < self.theta:
            # Mouvement de prix
            tick_size = self.state[1].price - self.state[0].price
            if is_bid:
                # Le prix de référence diminue d'un tick
                self.p_ref -= tick_size
                # Décalage des états vers la gauche
                for i in range(len(self.state)-1):
                    self.state[i].size = self.state[i+1].size
                    self.state[i].price = self.state[i+1].price - tick_size
                # Nouvelle limite ask à la fin
                self.state[-1].size = int(50 * np.exp(-0.1))
                self.state[-1].price = self.state[-2].price + tick_size
            else:
                # Le prix de référence augmente d'un tick
                self.p_ref += tick_size
                # Décalage des états vers la droite
                for i in range(len(self.state)-1, 0, -1):
                    self.state[i].size = self.state[i-1].size
                    self.state[i].price = self.state[i-1].price + tick_size
                # Nouvelle limite bid au début
                self.state[0].size = int(50 * np.exp(-0.1))
                self.state[0].price = self.state[1].price - tick_size
        else:
            # Régénération de la limite vide
            self.state[index].size = self.regenerate_queue(index)

    def update_state(self, lambda_: List[List[float]], mju: List[List[float]], stf: int, Cbound: int, delta: float, H: float):
        """
        Met à jour l'état du carnet d'ordres avec des intensités calibrées.
        """
        k = self.k
        for i in range(self.k * 2):
            size = self.state[i].size
            # Intensités calibrées selon la profondeur du carnet
            depth_factor = np.exp(-0.1 * abs(i - k))
            
            # Taux d'arrivée ajusté selon la profondeur et la taille actuelle
            # Assurer que l'intensité est positive
            arrival_intensity = max(0.0, lambda_[i % k][0] * depth_factor * (1 - size/Cbound))
            s_lambda = np.random.poisson(arrival_intensity * stf) if arrival_intensity > 0 else 0
            
            # Taux d'annulation proportionnel à la taille actuelle
            # Assurer que l'intensité est positive
            cancel_intensity = max(0.0, mju[i % k][0] * (size/Cbound) * depth_factor)
            s_mju = np.random.poisson(cancel_intensity * stf) if cancel_intensity > 0 else 0
            
            proposed_change = s_lambda - s_mju

            # Assumption 1: Negative individual drift pour les grandes queues
            if size > Cbound:
                proposed_change -= int(delta * (size/Cbound))

            # Assumption 2: Bound on the incoming flow avec scaling dynamique
            total_incoming_flow = sum(
                np.random.poisson(max(0.0, lambda_[j % k][0] * np.exp(-0.1 * abs(j)) * stf))
                for j in range(-k, k) if j != 0
            )
            
            if total_incoming_flow > H:
                scale_factor = H / max(total_incoming_flow, 1e-10)  # Éviter division par zéro
                proposed_change = int(proposed_change * scale_factor)

            # Mise à jour de la taille
            new_size = min(Cbound, max(0, size + proposed_change))
            
            # Si la taille devient nulle, gérer le cas spécial
            if new_size == 0 and size > 0:
                self.handle_empty_limit(i, i < k)
            else:
                self.state[i].size = new_size

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
    base_lambda = 500.0  # Intensité réduite pour les arrivées
    base_mju = 400.0     # Intensité réduite pour les annulations
    
    # Assurer des valeurs positives pour les intensités
    lambda_ = [[max(0.1, base_lambda * np.exp(-0.05 * i))] for i in range(k)]  # Minimum de 0.1
    mju = [[max(0.1, base_mju * np.exp(-0.03 * i))] for i in range(k)]        # Minimum de 0.1
    
    return lambda_, mju
# Paramètres optimisés
params = {
    "k": 10,             # Nombre de niveaux de prix de chaque côté du carnet d'ordres
    "p_ref": 10.0,      # Prix de référence initial
    "tick_size": 1,    # Taille minimale de variation de prix entre deux niveaux
    "stf": 1,            # Facteur d'échelle temporelle pour la simulation
    "simulation_time": 1000,  # Durée totale de la simulation en unités de temps
    "Cbound": 20,        # Limite maximale de la taille des ordres
    "delta": 0.8,        # Paramètre de dérive pour le processus de prix
    "H": 20              # Limite maximale du flux d'ordres par unité de temps
}

def plot_price_evolution(order_books: List[OrderBook], times: List[int]):
    # Préparation des données
    mid_prices = []
    best_bids = []
    best_asks = []
    all_asks = [[] for _ in range(3)]  # 3 niveaux d'asks
    all_bids = [[] for _ in range(3)]  # 3 niveaux de bids
    
    for lob in order_books:
        # Prix médian et meilleurs prix
        bid, ask = lob.get_best()
        mid_price = (bid + ask) / 2
        mid_prices.append(mid_price)
        best_bids.append(bid)
        best_asks.append(ask)
        
        # Niveaux de prix supplémentaires
        k = lob.k
        state = lob.state
        
        # 3 niveaux d'asks (du meilleur au pire)
        for i in range(3):
            idx = k + i
            if idx < len(state) and state[idx].size > 0:
                all_asks[i].append(state[idx].price)
            else:
                all_asks[i].append(None)
        
        # 3 niveaux de bids (du meilleur au pire)
        for i in range(3):
            idx = k - 1 - i
            if idx >= 0 and state[idx].size > 0:
                all_bids[i].append(state[idx].price)
            else:
                all_bids[i].append(None)

    # Création du graphique
    fig = go.Figure()

    # Mid price en noir
    fig.add_trace(go.Scatter(
        x=times,
        y=mid_prices,
        mode='lines',
        name='Mid Price',
        line=dict(color='black', width=2)
    ))

    # Couleurs pour les différents niveaux
    ask_colors = ['red', 'salmon', 'lightcoral']
    bid_colors = ['green', 'lightgreen', 'palegreen']

    # Ajout des niveaux d'asks
    for i in range(3):
        fig.add_trace(go.Scatter(
            x=times,
            y=all_asks[i],
            mode='lines',
            name=f'Ask Level {i+1}',
            line=dict(color=ask_colors[i], width=1, dash='dot' if i > 0 else 'solid')
        ))

    # Ajout des niveaux de bids
    for i in range(3):
        fig.add_trace(go.Scatter(
            x=times,
            y=all_bids[i],
            mode='lines',
            name=f'Bid Level {i+1}',
            line=dict(color=bid_colors[i], width=1, dash='dot' if i > 0 else 'solid')
        ))

    fig.update_layout(
        title="Évolution des Prix - Multiple Niveaux",
        xaxis_title="Temps",
        yaxis_title="Prix",
        showlegend=True
    )

    fig.show()

# Simulation avec intensités calibrées
lambda_, mju = get_calibrated_intensities(params["k"])
order_books = []
times = []

# Simulation
lob = init_order_book(params["p_ref"], params["k"], params["tick_size"])
for t in tqdm(range(0, params["simulation_time"], params["stf"]), desc="Simulation Progress"):
    lob.update_state(lambda_, mju, params["stf"], params["Cbound"], params["delta"], params["H"])
    if t % 10 == 0:  # Enregistrement plus fréquent
        order_books.append(OrderBook([Queue(q.price, q.size) for q in lob.state], lob.k, lob.p_ref))
        times.append(t)

# Visualisation
plot_price_evolution(order_books, times)