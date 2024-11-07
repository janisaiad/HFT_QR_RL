import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple
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
    k: int  # Nombre de niveaux de prix pour bid et pour ask
    p_ref: float  # Prix de référence

    def get_best(self) -> Tuple[float, float]:
        """
        Obtient les meilleurs prix bid et ask.

        :return: Tuple contenant le meilleur prix bid et le meilleur prix ask
        """
        k = self.k 
        bid = self.state[k - 1].price if self.state[k - 1].size > 0 else self.state[k - 2].price  # best price bid
        ask = self.state[k].price if self.state[k].size > 0 else self.state[k + 1].price  # best price ask
        return bid, ask 


    def update_state(self, lambda_: List[List[float]], mju: List[List[float]], stf: int, Cbound: int, delta: float, H: float):
        """
        Met à jour l'état du carnet d'ordres en fonction des distributions lambda et mju en tenant compte des Assumptions 1 et 2.

        :param lambda_: Taux d'arrivée des ordres
        :param mju: Taux d'annulation des ordres
        :param stf: Facteur de mise à l'échelle du temps
        :param Cbound: Limite supérieure pour la taille de la file d'attente
        :param delta: Valeur delta pour la diminution de la taille de la file d'attente
        :param H: Limite supérieure pour le flux entrant
        """
        
        
        for i in range(self.k * 2):  # Pour chaque niveau de prix
            size = self.state[i].size  # Taille actuelle de la file d'attente
            s_lambda = np.random.poisson(lambda_[i % self.k] * stf)  # Taux d'arrivée des ordres
            s_mju = np.random.poisson(mju[i % self.k] * stf)  # Taux d'annulation des ordres
            proposed_change = s_lambda - s_mju  # Changement proposé de la taille

            # Assumption 1: Negative individual drift
            if size > Cbound:
                proposed_change -= delta

            # Assumption 2: Bound on the incoming flow
            total_incoming_flow = sum(np.random.poisson(lambda_[j % self.k] * stf) for j in range(-self.k, self.k) if j != 0)
            if total_incoming_flow > H:
                proposed_change = min(proposed_change, H - size)

            # Utilisation de la fonction de régénération pour ajuster la taille
            if size + proposed_change <= 0:
                Regen_func_basic(i, self.state)  # Appel à la fonction de régénération

            self.state[i].size = max(0, size + proposed_change)  # Mise à jour de la taille de la file d'attente
            
            
def Regen_func_basic(i: int, state: List[Queue]):
    """
    Fonction de régénération pour ajuster la taille de la file d'attente.

    :param i: Indice de la file d'attente
    :param state: État actuel du carnet d'ordres
    """
    if state[i].size <= 0:
        state[i].size = 1  # Régénération de la taille de la file d'attente


def init_order_book(p_ref: float, invariant: List[List[float]], k: int, tick_size: float) -> OrderBook:
    """
    Initialise un nouveau carnet d'ordres.

    :param p_ref: Prix de référence
    :param invariant: Liste des états invariants
    :param k: Nombre de niveaux de prix de chaque côté
    :param tick_size: Taille du tick
    :return: Carnet d'ordres initialisé
    """
    p_lowest = p_ref - tick_size/2 - tick_size * k  # Calcul du prix le plus bas
    state = [Queue(p_lowest + i * tick_size, random.choice(invariant[i % k])) for i in range(k * 2)]  # Initialisation de l'état
    return OrderBook(state, k, p_ref)  # Retourne le carnet d'ordres initialisé



def plot_prices(order_books: List[OrderBook], times: List[int]):
    """
    Trace l'évolution des prix (mid, bid, ask) au fil du temps.

    :param order_books: Liste des carnets d'ordres à différents moments
    :param times: Liste des moments correspondants
    """
    # Préparation des données pour le graphique
    best_bid_prices = []
    best_ask_prices = []
    mid_prices = []
    
    for lob in order_books:
        bid, ask = lob.get_best()
        best_bid_prices.append(bid)
        best_ask_prices.append(ask)
        mid_prices.append((bid + ask) / 2)

    # Création du graphique
    fig = go.Figure()
    
    # Ajout des courbes de prix
    fig.add_trace(go.Scatter(
        x=times,
        y=best_bid_prices,
        mode='lines',
        name='Bid Price'
    ))
    
    fig.add_trace(go.Scatter(
        x=times,
        y=best_ask_prices,
        mode='lines',
        name='Ask Price'
    ))
    
    fig.add_trace(go.Scatter(
        x=times,
        y=mid_prices,
        mode='lines',
        name='Mid Price'
    ))

    fig.update_layout(
        title="Évolution des prix au fil du temps",
        xaxis_title="Temps",
        yaxis_title="Prix",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    fig.show()


def plot_sizes(order_books: List[OrderBook], times: List[int]):
    """
    Trace l'évolution des tailles des files d'attente au fil du temps.
    
    :param order_books: Liste des carnets d'ordres à différents moments
    :param times: Liste des moments correspondants
    """
    # Création d'un tableau numpy pour stocker les tailles
    n_times = len(times)
    n_queues = len(order_books[0].state)
    sizes = np.zeros((n_times, n_queues))
    
    # Remplissage efficace du tableau
    for t, lob in enumerate(order_books):
        sizes[t] = [q.size for q in lob.state]
    
    # Création du graphique
    fig = go.Figure()
    
    # Ajout des courbes pour chaque niveau de prix
    for i in range(n_queues):
        fig.add_trace(go.Scatter(
            x=times,
            y=sizes[:, i],
            mode='lines',
            name=f'Queue {i+1} (Price: {order_books[0].state[i].price:.2f})'
        ))

    fig.update_layout(
        title="Évolution des tailles des files d'attente",
        xaxis_title="Temps",
        yaxis_title="Taille",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    fig.show()


# Paramètres du modèle
params = {
    "k": 3,  # Nombre de niveaux de prix
    "p_ref": 10.0,  # Prix de référence
    "tick_size": 0.1,  # Taille du tick
    "stf": 1,  # Facteur de mise à l'échelle du temps
    "simulation_time": 60000,  # Temps de simulation
    "Cbound": 10,  # Limite supérieure pour la taille de la file d'attente
    "delta": 0.1,  # Valeur delta pour la diminution de la taille de la file d'attente
    "H": 5  # Limite supérieure pour le flux entrant
}

# Initialisation
def get_distributions(k: int, scale: int) -> Tuple[List[float], List[float]]:
    """
    Génère des distributions lambda et mju pour le modèle.

    :param k: Nombre de niveaux de prix de chaque côté
    :param scale: Échelle pour les distributions
    :return: Tuple des listes lambda et mju
    """
    lambda_ = [min(scale, max(0, scale - i)) for i in range(k)]  # Génération des valeurs lambda selon Assumption 1 et 2
    mju = [random.uniform(0, 1) * scale for _ in range(k)]  # Génération des valeurs mju
    print('lambda_', lambda_)
    print('mju', mju)
    return lambda_, mju  # Retourne les distributions lambda et mju



def assign_invariant(lambda_: List[float], mju: List[float], k: int) -> List[List[float]]:
    """
    Assigne des états invariants basés sur les distributions lambda et mju.

    :param lambda_: Liste des valeurs lambda
    :param mju: Liste des valeurs mju
    :param k: Nombre de niveaux de prix de chaque côté
    :return: Liste des états invariants
    """
    invariant = [[lambda_[i], mju[i]] for i in range(k)]  # Création des états invariants
    return invariant  # Retourne les états invariants



lambda_, mju = get_distributions(params["k"], 100)  # Récupération des distributions lambda et mju
invariant = assign_invariant(lambda_, mju, params["k"])  # Assignation des états invariants
order_books = []  # Liste des carnets d'ordres
times = []  # Liste des moments



# Simulation
lob = init_order_book(params["p_ref"], invariant, params["k"], params["tick_size"])  # Initialisation du carnet d'ordres
for t in tqdm(range(0, params["simulation_time"], params["stf"]), desc="Simulation Progress"):  # Pour chaque instant de simulation avec barre de progression
    lob.update_state(lambda_, mju, params["stf"], params["Cbound"], params["delta"], params["H"])  # Mise à jour de l'état du carnet d'ordres
    if t % 100 == 0:  # Enregistrer l'état toutes les 100 itérations
        order_books.append(OrderBook([Queue(q.price, q.size) for q in lob.state], lob.k, lob.p_ref))  # Ajout de l'état actuel à la liste
        times.append(t)  # Ajout du moment actuel à la liste

# Affichage des résultats
plot_prices(order_books, times)  # Affichage de l'évolution des prix
plot_sizes(order_books, times)  # Affichage de l'évolution des tailles