"""一些简单的统计函数。"""


def mean(xs):
    """算术平均数。"""
    return sum(xs) / len(xs)


def median(xs):
    """中位数：排序后取中间的数；个数为偶数时，取中间两个数的平均。"""
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid] + s[mid + 1]) / 2
