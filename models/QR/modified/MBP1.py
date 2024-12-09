import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple
from dataclasses import dataclass
from enum import Enum
import random


@dataclass
class Queue:
    price: float
    size: int

class QType(Enum):
    ASK = 1
    BID = 2

class Age(Enum):
    CURRENT = 1
    LAST = 2

@dataclass
class Price:
    bid: float
    ask: float
    time: int

@dataclass
class Candle:
    high: float
    open: float
    close: float
    low: float
    time: int
    last_price_index: int

class OrderBook:
    def __init__(self, initial_state: List[Queue], k: int, p_ref: float):
        """
        Initialise un carnet d'ordres.

        :param initial_state: État initial des files d'attente
        :param k: Nombre de niveaux de prix de chaque côté
        :param p_ref: Prix de référence
        """
        self.state = initial_state
        self.last_state = [Queue(q.price, q.size) for q in initial_state]
        self.k = k
        self.p_ref = p_ref

    def get_best(self) -> Tuple[float, float]:
        """
        Obtient les meilleurs prix bid et ask.

        :return: Tuple contenant le meilleur prix bid et le meilleur prix ask
        """
        k = self.k
        if self.state[k - 1].size == 0:
            bid = self.state[k - 2].price
        else:
            bid = self.state[k - 1].price
        if self.state[k].size == 0:
            ask = self.state[k + 1].price
        else:
            ask = self.state[k].price
        return bid, ask

    def print(self):
        """
        Affiche l'état actuel du carnet d'ordres.
        """
        print("Limit Order-book state:")
        for q in self.state:
            print(f"Price: {q.price:.5f}, \t Size: {q.size}")
        print("________________________")

    def get_size(self, level: int, age: Age, q_type: QType) -> int:
        """
        Obtient la taille d'une file d'attente spécifique.

        :param level: Niveau de prix
        :param age: Âge de l'état (actuel ou dernier)
        :param q_type: Type de file d'attente (ask ou bid)
        :return: Taille de la file d'attente
        """
        if age == Age.CURRENT:
            if q_type == QType.ASK:
                return self.state[level + self.k].size
            else:
                return self.state[self.k - level - 1].size
        else:
            if q_type == QType.ASK:
                return self.last_state[level + self.k].size
            else:
                return self.last_state[self.k - level - 1].size

    def shift(self, invariant: List[List[float]], tick_size: float, direction: int):
        """
        Décale les prix dans le carnet d'ordres.

        :param invariant: Liste des états invariants
        :param tick_size: Taille du tick
        :param direction: Direction du décalage (positif ou négatif)
        """
        for i in range(self.k * 2):
            self.last_state[i].price = self.state[i].price
            self.last_state[i].size = self.state[i].size

        for i in range(self.k * 2):
            self.state[i].price += direction * tick_size
            if 0 <= i + direction < self.k * 2:
                self.state[i].size = self.last_state[i + direction].size

        if direction > 0:
            self.state[-1] = Queue(self.state[self.k - 2].price + tick_size, 
                                   pull_from_invariant(invariant, self.k * 2 - 1))
        else:
            self.state[0] = Queue(self.state[1].price - tick_size,
                                  pull_from_invariant(invariant, 0))

        self.p_ref += direction * tick_size
        assert len(self.state) == 6

    def get_dep_theta(self, theta: float) -> float:
        """
        Calcule le theta dépendant de l'état du carnet d'ordres.

        :param theta: Valeur de base de theta
        :return: Theta dépendant de l'état
        """
        bid_sum = sum(q.size for q in self.state[:self.k])
        ask_sum = sum(q.size for q in self.state[self.k:])
        ratio = ask_sum / (ask_sum + bid_sum)
        return max(0, min(theta - 0.0 + ratio * 0.0, 1.0))

class Prices:
    def __init__(self):
        """
        Initialise une structure pour stocker les prix et les bougies.
        """
        self.prices: List[Price] = []
        self.candles: List[Candle] = []

    def set(self, lob: OrderBook, time: int):
        """
        Enregistre les meilleurs prix bid et ask à un instant donné.

        :param lob: Carnet d'ordres
        :param time: Temps actuel
        """
        bid, ask = lob.get_best()
        self.prices.append(Price(bid, ask, time))

    def get_mid_price(self, i: int) -> float:
        """
        Calcule le prix moyen pour un indice donné.

        :param i: Indice du prix
        :return: Prix moyen
        """
        return (self.prices[i].ask + self.prices[i].bid) / 2

    def add_candle(self, time: int, candle_size: int):
        """
        Ajoute une nouvelle bougie à la liste des bougies.

        :param time: Temps actuel
        :param candle_size: Taille de la bougie
        """
        if not self.candles:
            first_price_index = 0
        else:
            first_price_index = self.candles[-1].last_price_index

        last_price_index = len(self.prices)
        open_price = self.get_mid_price(first_price_index)
        close_price = self.get_mid_price(last_price_index - 1)
        high = max(self.get_mid_price(i) for i in range(first_price_index, last_price_index))
        low = min(self.get_mid_price(i) for i in range(first_price_index, last_price_index))

        time = time - candle_size + 1
        self.candles.append(Candle(high, open_price, close_price, low, time, last_price_index - 1))

    def print_candle(self, index: int):
        """
        Affiche les informations d'une bougie spécifique.

        :param index: Indice de la bougie à afficher
        """
        cn = self.candles[index]
        print(f"open: {cn.open:.5f}, high: {cn.high:.5f}, low: {cn.low:.5f}, close: {cn.close:.5f}, time: {cn.time}")

    def get_mid_prices_by_time(self, start: int, end: int, simulation_time: int) -> List[float]:
        """
        Obtient les prix moyens dans un intervalle de temps donné.

        :param start: Temps de début
        :param end: Temps de fin
        :param simulation_time: Durée totale de la simulation
        :return: Liste des prix moyens
        """
        assert 0 <= start <= end <= simulation_time
        return [self.get_mid_price(i) for i in range(len(self.prices)) 
                if start <= self.prices[i].time < end]

    def get_10_min_cal_params(self, stf: int, simulation_time: int) -> Tuple[float, float]:
        """
        Calcule les paramètres de calibration sur une période de 10 minutes.

        :param stf: Facteur de mise à l'échelle du temps
        :param simulation_time: Durée totale de la simulation
        :return: Tuple contenant le taux de rendement moyen et la volatilité
        """
        min10 = 1000 * 60 * 10
        if simulation_time < min10:
            raise ValueError(f"Simulation time ({simulation_time}) must be at least {min10}")

        nc, na = 0, 0
        i = 2
        cur_time = self.prices[i].time
        while cur_time < min10 and i <= len(self.prices) - 1:
            cur_time = self.prices[i].time
            if (self.get_mid_price(i-2) - self.get_mid_price(i-1) < 0) == (self.get_mid_price(i-1) - self.get_mid_price(i) < 0):
                nc += 1
            else:
                na += 1
            i += 1

        mid_prices = self.get_mid_prices_by_time(0, min10, simulation_time)
        mrr = nc / (na * 2) if na > 0 else 0
        vol = np.std(mid_prices)
        return mrr, vol

def pull_from_invariant(invariant: List[List[float]], index: int) -> int:
    """
    Tire une valeur aléatoire de l'invariant pour un index donné.

    :param invariant: Liste des états invariants
    :param index: Index de l'état à tirer
    :return: Valeur tirée de l'invariant
    """
    return int(random.choice(invariant[index]))

def init(p_ref: float, invariant: List[List[float]], k: int, tick_size: float) -> OrderBook:
    """
    Initialise un nouveau carnet d'ordres.

    :param p_ref: Prix de référence
    :param invariant: Liste des états invariants
    :param k: Nombre de niveaux de prix de chaque côté
    :param tick_size: Taille du tick
    :return: Carnet d'ordres initialisé
    """
    p_lowest = p_ref - tick_size/2 - tick_size * k
    state = [Queue(p_lowest + i * tick_size, pull_from_invariant(invariant, i)) for i in range(k * 2)]
    return OrderBook(state, k, p_ref)

def reinit(lob: OrderBook, p_ref: float, invariant: List[List[float]], k: int, tick_size: float):
    """
    Réinitialise un carnet d'ordres existant.

    :param lob: Carnet d'ordres à réinitialiser
    :param p_ref: Nouveau prix de référence
    :param invariant: Liste des états invariants
    :param k: Nombre de niveaux de prix de chaque côté
    :param tick_size: Taille du tick
    """
    p_lowest = p_ref - tick_size/2 - tick_size * k
    for i in range(k * 2):
        lob.state[i] = Queue(p_lowest + i * tick_size, pull_from_invariant(invariant, i))
    lob.p_ref = p_ref

def iterate(lob: OrderBook, stf: int, k: int, lambda_: List[List[float]], mju: List[List[float]]):
    """
    Effectue une itération du modèle de carnet d'ordres.

    :param lob: Carnet d'ordres
    :param stf: Facteur de mise à l'échelle du temps
    :param k: Nombre de niveaux de prix de chaque côté
    :param lambda_: Taux d'arrivée des ordres
    :param mju: Taux d'annulation des ordres
    """
    for i in range(k * 2):
        lob.last_state[i].size = lob.state[i].size
        size = lob.state[i].size
        s_lambda = np.random.poisson(lambda_[i % k][size] * stf)
        s_mju = np.random.poisson(mju[i % k][size] * stf)
        proposed_change = s_lambda - s_mju
        lob.state[i].size = max(0, size + proposed_change)

def maybe_modify(lob: OrderBook, invariant: List[List[float]], tick_size: float, theta: float, theta_reinit: float):
    """
    Modifie potentiellement le carnet d'ordres en fonction de certaines conditions.

    :param lob: Carnet d'ordres
    :param invariant: Liste des états invariants
    :param tick_size: Taille du tick
    :param theta: Paramètre de contrôle pour les modifications
    :param theta_reinit: Probabilité de réinitialisation du carnet d'ordres
    """
    ask_size = lob.get_size(0, Age.CURRENT, QType.ASK)
    bid_size = lob.get_size(0, Age.CURRENT, QType.BID)
    z_ask = int(ask_size == 0)
    z_bid = int(bid_size == 0)
    c_ask = int(lob.get_size(0, Age.LAST, QType.ASK) != 0)
    c_bid = int(lob.get_size(0, Age.LAST, QType.BID) != 0)
    direction = z_ask * c_ask - z_bid * c_bid

    if direction != 0:
        r = random.random()
        lob_dep_theta = lob.get_dep_theta(theta)
        if r < lob_dep_theta:
            lob.shift(invariant, tick_size, direction)
            r = random.random()
            if r < theta_reinit:
                p_ref = lob.p_ref
                k = lob.k
                reinit(lob, p_ref, invariant, k, tick_size)

def maybe_save_prices(lob: OrderBook, prices: Prices, i: int):
    """
    Enregistre les prix si nécessaire.

    :param lob: Carnet d'ordres
    :param prices: Structure de stockage des prix
    :param i: Indice de temps actuel
    """
    bid, ask = lob.get_best()
    if not prices.prices or prices.prices[-1].bid != bid or prices.prices[-1].ask != ask:
        prices.set(lob, i)

def maybe_save_candle(prices: Prices, i: int, candle_size: int):
    """
    Enregistre une nouvelle bougie si nécessaire.

    :param prices: Structure de stockage des prix
    :param i: Indice de temps actuel
    :param candle_size: Taille de la bougie
    """
    if (i + 1) % candle_size == 0:
        prices.add_candle(i, candle_size)

def simulation(k: int, candle_size: int, resize: int, p_ref: float, tick_size: float, stf: int,
               simulation_time: int, theta_reinit: float, theta: float) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    """
    Cette fonction simule l'évolution d'un carnet d'ordres sur une période donnée.

    Paramètres:
    - k (int): Nombre de niveaux de prix de chaque côté du carnet d'ordres
    - candle_size (int): Taille des bougies en nombre d'itérations
    - resize (int): Taille maximale des files d'attente
    - p_ref (float): Prix de référence initial
    - tick_size (float): Taille minimale de variation de prix
    - stf (int): Facteur de mise à l'échelle du temps
    - simulation_time (int): Durée totale de la simulation
    - theta_reinit (float): Probabilité de réinitialisation du carnet d'ordres
    - theta (float): Paramètre de contrôle pour les modifications du carnet d'ordres

    Retourne:
    - Tuple[List[float], List[float], List[float], List[float], List[float]]:
      Contient les prix d'ouverture, les plus hauts, les plus bas, les prix de clôture,
      et une liste avec le taux de rendement moyen et la volatilité.
    """
    lambda_, mju = get_distributions(k, resize)
    invariant = assign_invariant(lambda_, mju, k)
    lob = init(p_ref, invariant, k, tick_size)
    prices = Prices()
    prices.set(lob, 0)

    for i in range(1, simulation_time // stf):
        iterate(lob, stf, k, lambda_, mju)
        maybe_modify(lob, invariant, tick_size, theta, theta_reinit)
        maybe_save_prices(lob, prices, i)
        maybe_save_candle(prices, i, candle_size)
        assert len(lob.state) == 6

    o = [c.open for c in prices.candles]
    h = [c.high for c in prices.candles]
    l = [c.low for c in prices.candles]
    c = [c.close for c in prices.candles]
    
    try:
        mrr, vol = prices.get_10_min_cal_params(stf, simulation_time)
    except ValueError as e:
        print(f"Warning: {str(e)}. Using default values for mrr and vol.")
        mrr, vol = 0, 0
    
    return o, h, l, c, [mrr, vol]

def get_distributions(k: int, resize: int) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Génère les distributions lambda et mju.

    :param k: Nombre de niveaux de prix de chaque côté
    :param resize: Taille maximale des files d'attente
    :return: Tuple contenant les distributions lambda et mju
    """
    # This function should be implemented to return lambda and mju distributions
    # For now, we'll return dummy values
    return [[random.random() for _ in range(resize)] for _ in range(k)], [[random.random() for _ in range(resize)] for _ in range(k)]

def assign_invariant(lambda_: List[List[float]], mju: List[List[float]], k: int) -> List[List[float]]:
    """
    Assigne l'invariant basé sur lambda et mju.

    :param lambda_: Distribution lambda
    :param mju: Distribution mju
    :param k: Nombre de niveaux de prix de chaque côté
    :return: Liste des états invariants
    """
    # This function should be implemented to assign invariant based on lambda and mju
    # For now, we'll return dummy values
    return [[random.randint(1, 10) for _ in range(10)] for _ in range(k * 2)]

def plot_results(mid_prices: List[float], bid_prices: List[float], ask_prices: List[float], bid_quantities: List[float], ask_quantities: List[float]) -> None:
    """
    Trace les résultats de la simulation.

    :param mid_prices: Liste des prix moyens
    :param bid_prices: Liste des prix bid
    :param ask_prices: Liste des prix ask
    :param bid_quantities: Liste des quantités bid
    :param ask_quantities: Liste des quantités ask
    """
    fig = go.Figure()
    
    # Tracer les prix
    fig.add_trace(go.Scatter(y=mid_prices, mode='lines', name='Prix moyen', line=dict(color='green')))
    fig.add_trace(go.Scatter(y=bid_prices, mode='lines', name='Prix bid', line=dict(color='blue')))
    fig.add_trace(go.Scatter(y=ask_prices, mode='lines', name='Prix ask', line=dict(color='red')))
    
    # Tracer les quantités
    fig.add_trace(go.Bar(y=bid_quantities, name='Quantité bid', marker_color='blue', opacity=0.3))
    fig.add_trace(go.Bar(y=ask_quantities, name='Quantité ask', marker_color='red', opacity=0.3))
    
    fig.update_layout(
        title="Carnet d'ordres MBP-1",
        xaxis_title="Pas de temps",
        yaxis_title="Prix / Quantité",
        barmode='overlay'
    )
    
    fig.show()

# Paramètres du modèle
params = {
    "k": 3,
    "candle_size": 100,
    "resize": 100,  
    "p_ref": 100.0,
    "tick_size": 0.01,
    "stf": 1,
    "simulation_time": 60000,
    "theta_reinit": 0.1,
    "theta": 0.5,
}

# Simulation
o, h, l, c, calib = simulation(**params)

# Affichage des résultats
plot_results([(hi + lo) / 2 for hi, lo in zip(h, l)], l, h, o, c)

print(f"MRR: {calib[0]:.5f}")
print(f"Volatility: {calib[1]:.5f}")
