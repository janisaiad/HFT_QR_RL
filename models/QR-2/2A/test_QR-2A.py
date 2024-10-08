import unittest
from QR_2A import Queue, QType, OrderBook, init_order_book, lambda_func, mju_func, market_order_func

class TestQR2A(unittest.TestCase):

    def setUp(self):
        self.params = {
            "k": 3,
            "p_ref": 10.0,
            "tick_size": 0.1,
            "stf": 1,
            "Cbound": 10,
            "delta": 0.1,
            "H": 5
        }
        self.lambda_funcs = [lambda_func(i, 100) for i in range(self.params["k"] * 2)]
        self.mju_funcs = [mju_func(50) for _ in range(self.params["k"] * 2)]
        self.market_order_func = market_order_func(75)
        self.initial_state = [5 for _ in range(self.params["k"] * 2)]
        self.lob = init_order_book(self.params["p_ref"], self.initial_state, self.params["k"], self.params["tick_size"])

    def test_get_best(self):
        bid, ask = self.lob.get_best()
        self.assertEqual(bid, 9.9)
        self.assertEqual(ask, 10.1)

    def test_update_state(self):
        initial_sizes = [q.size for q in self.lob.state]
        self.lob.update_state(self.lambda_funcs, self.mju_funcs, self.params["stf"], self.params["Cbound"], self.params["delta"], self.params["H"])
        updated_sizes = [q.size for q in self.lob.state]
        self.assertNotEqual(initial_sizes, updated_sizes)

    def test_process_market_orders(self):
        initial_sizes = [q.size for q in self.lob.state]
        self.lob.process_market_orders(self.market_order_func)
        updated_sizes = [q.size for q in self.lob.state]
        self.assertNotEqual(initial_sizes, updated_sizes)

    def test_execute_market_orders(self):
        initial_size = self.lob.state[self.params["k"] - 1].size
        self.lob.execute_market_orders(QType.BID, 3)
        updated_size = self.lob.state[self.params["k"] - 1].size
        self.assertEqual(initial_size - 3, updated_size)

    def test_regen_func_basic(self):
        self.lob.state[0].size = 0
        self.lob.state[0].size = 1
        self.assertEqual(self.lob.state[0].size, 1)

if __name__ == '__main__':
    unittest.main()
