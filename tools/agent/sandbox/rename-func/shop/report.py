"""打印订单小票。"""
from shop.cart import apply_discount, calc_total


def receipt(items, rate=1.0):
    total = calc_total(items)
    pay = apply_discount(total, rate)
    return f"合计 {total} 元，实付 {pay} 元"
