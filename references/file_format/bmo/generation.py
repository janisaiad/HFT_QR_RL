from datetime import datetime, timedelta
from typing import List
import random

class Order:
    def __init__(self, order_id: int, instrument_alias: str, bid: int, limit_price: float, size: int):
        self.order_id = order_id
        self.instrument_alias = instrument_alias
        self.bid = bid
        self.limit_price = limit_price
        self.size = size
        self.timestamp = datetime.now()

    def to_bmo_format(self) -> str:
        date_str = self.timestamp.strftime("%Y%m%d")
        time_str = self.timestamp.strftime("%H%M%S")
        subsecond_str = f"{self.timestamp.microsecond / 1e6:.9f}".split(".")[1]
        return f"S,{date_str},{time_str},{subsecond_str},{self.order_id},{self.instrument_alias},{self.bid},{self.limit_price},{self.size}"

class BMOGenerator:
    def __init__(self):
        self.orders: List[Order] = []
        self.events: List[str] = []

    def add_order(self, order: Order):
        self.orders.append(order)
        self.events.append(order.to_bmo_format())

    def simulate_events(self):
        for order in self.orders:
            # Simulate order updates
            for _ in range(random.randint(1, 5)):
                order.timestamp += timedelta(seconds=random.randint(1, 10))
                new_limit_price = order.limit_price + random.uniform(-1.0, 1.0)
                new_size = order.size + random.randint(-1, 1)
                self.events.append(f"U,{order.timestamp.strftime('%Y%m%d')},{order.timestamp.strftime('%H%M%S')},{f'{order.timestamp.microsecond / 1e6:.9f}'.split('.')[1]},{order.order_id},{new_limit_price},{new_size}")

            # Simulate order execution
            order.timestamp += timedelta(seconds=random.randint(1, 10))
            exec_price = order.limit_price + random.uniform(-0.5, 0.5)
            exec_size = random.randint(1, order.size)
            self.events.append(f"E,{order.timestamp.strftime('%Y%m%d')},{order.timestamp.strftime('%H%M%S')},{f'{order.timestamp.microsecond / 1e6:.9f}'.split('.')[1]},{order.order_id},{exec_price},{exec_size}")

            # Simulate order fill
            order.timestamp += timedelta(seconds=random.randint(1, 10))
            self.events.append(f"F,{order.timestamp.strftime('%Y%m%d')},{order.timestamp.strftime('%H%M%S')},{f'{order.timestamp.microsecond / 1e6:.9f}'.split('.')[1]},{order.order_id}")

    def generate_bmo_file(self, file_path: str):
        with open(file_path, "w") as file:
            file.write("!BOOKMAP_FORMAT_V1\n")
            file.write("!DO_NOT_UPDATE_AFTER_EXECUTION\n")
            for event in self.events:
                file.write(event + "\n")

# Example usage
if __name__ == "__main__":
    generator = BMOGenerator()
    for i in range(10000):
        order_id = 1105671100 + i
        instrument_alias = "ESU8.CME@RITHMIC"
        bid = random.randint(0, 1)
        limit_price = round(random.uniform(2800.0, 2900.0), 2)
        size = random.randint(1, 10)
        order = Order(order_id=order_id, instrument_alias=instrument_alias, bid=bid, limit_price=limit_price, size=size)
        generator.add_order(order)
    generator.simulate_events()
    generator.generate_bmo_file("/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/references/file_format/bmo/data.txt")

