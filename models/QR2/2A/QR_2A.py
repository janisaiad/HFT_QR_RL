import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import random
from tqdm import tqdm  # type: ignore # Importation de tqdm pour afficher la progression

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
    p_ref: float  # Prix de référence, fixé

    def get_best(self) -> Tuple[float, float]:
        """
        Obtient les meilleurs prix bid et ask.

        :return: Tuple contenant le meilleur prix bid et le meilleur prix ask
        """
        k = self.k  # Nombre de niveaux de prix
        bid = self.state[k - 1].price if self.state[k - 1].size > 0 else self.state[k - 2].price  # Meilleur prix bid
        ask = self.state[k].price if self.state[k].size > 0 else self.state[k + 1].price  # Meilleur prix ask
        return bid, ask  # Retourne les meilleurs prix bid et ask



    def update_state(self, lambda_funcs: List[Callable], mju_funcs: List[Callable], stf: int, Cbound: int, delta: float, H: float):
        """
        Met à jour l'état du carnet d'ordres en fonction des fonctions d'intensité lambda et mju.

        :param lambda_funcs: Fonctions d'intensité d'arrivée des ordres
        :param mju_funcs: Fonctions d'intensité d'annulation des ordres
        :param stf: Facteur de mise à l'échelle du temps
        :param Cbound: Limite supérieure pour la taille de la file d'attente
        :param delta: Valeur delta pour la diminution de la taille de la file d'attente
        :param H: Limite supérieure pour le flux entrant
        """
        for i in range(self.k * 2):
            size = self.state[i].size
            q1_size = self.state[self.k - 1].size if i < self.k else self.state[self.k].size
            q1_not_empty = q1_size > 0
            
            # Calcul des intensités en fonction de la position dans le carnet d'ordres
            if i in [self.k - 2, self.k + 1]:  # Q±2
                lambda_intensity = lambda_funcs[i](size, q1_not_empty)
            else:
                lambda_intensity = lambda_funcs[i](size, q1_not_empty)
            
            mju_intensity = mju_funcs[i](size)
            
            s_lambda = np.random.poisson(lambda_intensity * stf)
            s_mju = np.random.poisson(mju_intensity * stf)
            proposed_change = s_lambda - s_mju

            if size > Cbound:
                proposed_change -= delta

            total_incoming_flow = sum(np.random.poisson(lambda_funcs[j](self.state[j].size, q1_not_empty) * stf) for j in range(self.k * 2) if j != i)
            if total_incoming_flow > H:
                proposed_change = min(proposed_change, H - size)

            if size + proposed_change <= 0:
                Regen_func_basic(i, self.state)

            self.state[i].size = max(0, size + proposed_change)

    def process_market_orders(self, market_order_func: Callable):
        """
        Traite les ordres de marché en fonction de l'état des files d'attente Q±1 et Q±2.

        :param market_order_func: Fonction d'intensité des ordres de marché
        """
        for side in [QType.BID, QType.ASK]:
            idx = self.k - 1 if side == QType.BID else self.k
            if self.state[idx].size > 0:
                intensity = market_order_func(self.state[idx].size)
            else:
                intensity = market_order_func(self.state[idx + (1 if side == QType.ASK else -1)].size)
            
            market_orders = np.random.poisson(intensity)
            self.execute_market_orders(side, market_orders)

    def execute_market_orders(self, side: QType, quantity: int):
        """
        Exécute les ordres de marché sur le côté spécifié du carnet d'ordres.

        :param side: Côté du carnet d'ordres (BID ou ASK)
        :param quantity: Quantité d'ordres de marché à exécuter
        """
        idx = self.k - 1 if side == QType.BID else self.k
        while quantity > 0 and idx >= 0 and idx < len(self.state):
            if self.state[idx].size > 0:
                executed = min(quantity, self.state[idx].size)
                self.state[idx].size -= executed
                quantity -= executed
            idx += (1 if side == QType.ASK else -1)

def Regen_func_basic(i: int, state: List[Queue]):
    """
    Fonction de régénération pour ajuster la taille de la file d'attente.

    :param i: Indice de la file d'attente
    :param state: État actuel du carnet d'ordres
    """
    if state[i].size <= 0:
        state[i].size = 1  # Régénération de la taille de la file d'attente

def init_order_book(p_ref: float, initial_state: List[int], k: int, tick_size: float) -> OrderBook:
    """
    Initialise un nouveau carnet d'ordres.

    :param p_ref: Prix de référence
    :param initial_state: État initial des files d'attente
    :param k: Nombre de niveaux de prix de chaque côté
    :param tick_size: Taille du tick
    :return: Carnet d'ordres initialisé
    """
    p_lowest = p_ref - tick_size/2 - tick_size * k
    state = [Queue(p_lowest + i * tick_size, initial_state[i]) for i in range(k * 2)]
    return OrderBook(state, k, p_ref)

def plot_heatmap(order_books: List[OrderBook], times: List[int]):
    """
    Trace une heatmap de l'évolution des files d'attente au fil du temps.

    :param order_books: Liste des carnets d'ordres à différents moments
    :param times: Liste des moments correspondants
    """
    z_data = []
    for lob in order_books:
        row = [q.size for q in lob.state]
        z_data.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=[f'{"Bid" if i < len(z_data[0])//2 else "Ask"} {abs(i - len(z_data[0])//2 + 1)}' for i in range(len(z_data[0]))],
        y=times,
        colorscale='Viridis'
    ))

    fig.update_layout(
        title="Évolution des files d'attente au fil du temps",
        xaxis_title="Files d'attente",
        yaxis_title="Temps",
        yaxis=dict(autorange="reversed")
    )

    fig.show()

# Paramètres du modèle
params = {
    "k": 3,
    "p_ref": 10.0,
    "tick_size": 0.1,
    "stf": 1,
    "simulation_time": 20000,
    "Cbound": 10,
    "delta": 0.1,
    "H": 5
}

# Définition des fonctions d'intensité
def lambda_func(i: int, scale: float) -> Callable:
    if i in [params["k"] - 2, params["k"] + 1]:  # Q±2
        return lambda q, q1_not_empty: scale * (1 + 0.5 * q1_not_empty) / (1 + q)
    else:
        return lambda q, q1_not_empty: scale / (1 + q)

def mju_func(scale: float) -> Callable:
    return lambda q: scale * q / (1 + q)

def market_order_func(scale: float) -> Callable:
    return lambda q: scale / (1 + q)

# Initialisation
lambda_funcs = [lambda_func(i, 100) for i in range(params["k"] * 2)]
mju_funcs = [mju_func(50) for _ in range(params["k"] * 2)]
market_order_func = market_order_func(75)

initial_state = [random.randint(1, 10) for _ in range(params["k"] * 2)]
lob = init_order_book(params["p_ref"], initial_state, params["k"], params["tick_size"])

order_books = []
times = []

# Simulation
for t in tqdm(range(0, params["simulation_time"], params["stf"]), desc="Simulation Progress"):
    lob.update_state(lambda_funcs, mju_funcs, params["stf"], params["Cbound"], params["delta"], params["H"])
    lob.process_market_orders(market_order_func)
    if t % 100 == 0:
        order_books.append(OrderBook([Queue(q.price, q.size) for q in lob.state], lob.k, lob.p_ref))
        times.append(t)

# Affichage des résultats
plot_heatmap(order_books, times)



def plot_intensities_at_Q2(lambda_funcs: List[Callable], mju_funcs: List[Callable], market_order_func: Callable, max_queue_size: int):
    """
    Trace les intensités des ordres limites, des annulations et des ordres de marché à Q±2 en fonction de q±1 et q±2.

    :param lambda_funcs: Fonctions d'intensité d'arrivée des ordres
    :param mju_funcs: Fonctions d'intensité d'annulation des ordres
    :param market_order_func: Fonction d'intensité des ordres de marché
    :param max_queue_size: Taille maximale de la file d'attente
    """
    q2_values = list(range(max_queue_size + 1))
    q1_values = [0, 1]  # q1 == 0 et q1 > 0

    fig = go.Figure()

    for q1 in q1_values:
        lambda_intensities = [lambda_funcs[params["k"] - 2](q2, q1 > 0) for q2 in q2_values]
        mju_intensities = [mju_funcs[params["k"] - 2](q2) for q2 in q2_values]
        market_order_intensities = [market_order_func(q2) for q2 in q2_values]

        fig.add_trace(go.Scatter(x=q2_values, y=lambda_intensities, mode='lines', name=f'Limit Order Insertion q1 {"== 0" if q1 == 0 else "> 0"}'))
        fig.add_trace(go.Scatter(x=q2_values, y=mju_intensities, mode='lines', name=f'Limit Order Cancellation q1 {"== 0" if q1 == 0 else "> 0"}'))
        fig.add_trace(go.Scatter(x=q2_values, y=market_order_intensities, mode='lines', name=f'Market Order Insertion q1 {"== 0" if q1 == 0 else "> 0"}'))

    fig.update_layout(
        title="Intensities at Q±2",
        xaxis_title="Queue Size (per average event size)",
        yaxis_title="Intensity (num per second)",
        legend_title="Intensity Types",
        height=800
    )

    fig.show()

# Plotting the intensities at Q2
plot_intensities_at_Q2(lambda_funcs, mju_funcs, market_order_func, max_queue_size=40)
