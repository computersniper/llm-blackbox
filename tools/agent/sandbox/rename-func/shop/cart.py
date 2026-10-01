"""购物车。"""


def calc_total(items):
    """items 是 (单价, 数量) 的列表，返回总价。"""
    return sum(price * qty for price, qty in items)


def apply_discount(total, rate):
    """打折：rate=0.9 表示九折。"""
    return round(total * rate, 2)
