import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple, Dict
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
    p_ref: float  # Prix de référence

    def get_best(self) -> Tuple[float, float]:
        """
        Obtient les meilleurs prix bid et ask.

        :return: Tuple contenant le meilleur prix bid et le meilleur prix ask
        """
        k = self.k  # Nombre de niveaux de prix
        bid = self.state[k - 1].price if self.state[k - 1].size > 0 else self.state[k - 2].price  # Meilleur prix bid
        ask = self.state[k].price if self.state[k].size > 0 else self.state[k + 1].price  # Meilleur prix ask
        return bid, ask  # Retourne les meilleurs prix bid et ask

    def update_state(self, lambda_: List[List[float]], mju: List[List[float]], stf: int):
        """
        Met à jour l'état du carnet d'ordres en fonction des distributions lambda et mju.

        :param lambda_: Taux d'arrivée des ordres
        :param mju: Taux d'annulation des ordres
        :param stf: Facteur de mise à l'échelle du temps
        """
        for i in range(self.k * 2):  # Pour chaque niveau de prix
            size = self.state[i].size  # Taille actuelle de la file d'attente
            s_lambda = np.random.poisson(lambda_[i % self.k] * stf)  # Taux d'arrivée des ordres
            s_mju = np.random.poisson(mju[i % self.k] * stf)  # Taux d'annulation des ordres
            proposed_change = s_lambda - s_mju  # Changement proposé de la taille
            self.state[i].size = max(0, size + proposed_change)  # Mise à jour de la taille de la file d'attente

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

def plot_order_book_heatmap(order_books: List[OrderBook], times: List[int]):
    """
    Trace une carte thermique du carnet d'ordres avec l'axe x représentant le temps, l'axe y représentant le prix et la couleur représentant la taille de la file d'attente.

    :param order_books: Liste des carnets d'ordres à différents moments
    :param times: Liste des moments correspondants
    """
    prices = sorted(set(q.price for lob in order_books for q in lob.state))  # Récupération de tous les prix uniques
    heatmap_data = np.zeros((len(prices), len(times)))  # Initialisation de la matrice de données pour la carte thermique

    price_to_index = {price: idx for idx, price in enumerate(prices)}  # Mapping des prix aux indices de la matrice

    for t_idx, lob in enumerate(order_books):  # Pour chaque carnet d'ordres
        for q in lob.state:  # Pour chaque file d'attente dans le carnet d'ordres
            price_idx = price_to_index[q.price]  # Récupération de l'indice du prix
            heatmap_data[price_idx, t_idx] = q.size  # Mise à jour de la matrice de données avec la taille de la file d'attente

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=times,
        y=prices,
        colorscale='Viridis'
    ))

    fig.update_layout(
        title="Carte thermique du carnet d'ordres",
        xaxis_title="Temps",
        yaxis_title="Prix",
        yaxis=dict(type='category')
    )

    fig.show()  # Affichage de la figure

# Paramètres du modèle
params = {
    "k": 10,  # Nombre de niveaux de prix
    "p_ref": 10.0,  # Prix de référence
    "tick_size": 0.1,  # Taille du tick
    "stf": 1,  # Facteur de mise à l'échelle du temps
    "simulation_time": 60000,  # Temps de simulation
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
    lob.update_state(lambda_, mju, params["stf"])  # Mise à jour de l'état du carnet d'ordres
    if t % 1000 == 0:  # Enregistrer l'état toutes les 1000 itérations
        order_books.append(OrderBook([Queue(q.price, q.size) for q in lob.state], lob.k, lob.p_ref))  # Ajout de l'état actuel à la liste
        times.append(t)  # Ajout du moment actuel à la liste

# Affichage des résultats
plot_order_book_heatmap(order_books, times)  # Affichage de la carte thermique du carnet d'ordres
