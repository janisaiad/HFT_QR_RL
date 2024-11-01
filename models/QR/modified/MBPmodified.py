import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from enum import Enum

from Code import pull_from_invariant
from Code_simu_strat import (
    simulation,
    plot_results
)


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
            self.state[-1] = Queue(
                self.state[self.k - 2].price + tick_size,
                pull_from_invariant(invariant, self.k * 2 - 1)
            )
        else:
            self.state[0] = Queue(
                self.state[1].price - tick_size,
                pull_from_invariant(invariant, 0)
            )

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
        return [
            self.get_mid_price(i) for i in range(len(self.prices))
            if start <= self.prices[i].time < end
        ]

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
            condition = (
                (self.get_mid_price(i - 2) - self.get_mid_price(i - 1) < 0) ==
                (self.get_mid_price(i - 1) - self.get_mid_price(i) < 0)
            )
            if condition:
                nc += 1
            else:
                na += 1
            i += 1

        mid_prices = self.get_mid_prices_by_time(0, min10, simulation_time)
        mrr = nc / (na * 2) if na > 0 else 0
        vol = np.std(mid_prices)
        return mrr, vol


# Simulation Parameters and Execution
if __name__ == "__main__":
    # Paramètres du modèle
    params = {
        "k": 3,
        "candle_size": 100,
        "resize": 100,  # Reduced from 100 to 10
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
    mid_prices = [(hi + lo) / 2 for hi, lo in zip(h, l)]
    plot_results(mid_prices, l, h, o, c)

    print(f"MRR: {calib[0]:.5f}")
    print(f"Volatility: {calib[1]:.5f}")
