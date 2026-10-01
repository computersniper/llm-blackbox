import unittest

from shop.cart import calc_total
from shop.report import receipt


class TestShop(unittest.TestCase):
    def test_total(self):
        self.assertEqual(calc_total([(10, 2), (5, 1)]), 25)

    def test_receipt(self):
        self.assertEqual(receipt([(100, 1)], 0.9), "合计 100 元，实付 90.0 元")


if __name__ == "__main__":
    unittest.main()
