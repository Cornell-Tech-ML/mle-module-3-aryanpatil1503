"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    "$f(x, y) = x * y$"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")
    return x * y


def id(x: float) -> float:
    "$f(x) = x$"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")
    return x


def add(x: float, y: float) -> float:
    "$f(x, y) = x + y$"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")
    return x + y


def neg(x: float) -> float:
    "$f(x) = -x$"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")
    return -1 * x


v: Callable[[float, float], float] = add


def lt(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is less than y else 0.0"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")
    if x < y:
        return 1.0
    else:
        return 0.0


def eq(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is equal to y else 0.0"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")
    if x == y:
        return 1.0
    else:
        return 0.0


def max(x: float, y: float) -> float:
    "$f(x) =$ x if x is greater than y else y"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")
    if x > y:
        return x
    else:
        return y


def is_close(x: float, y: float) -> float:
    "$f(x) = |x - y| < 1e-2$"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")
    return abs(x - y) < 1**-2


def sigmoid(x: float) -> float:
    r"""
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Calculate as

    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    for stability.
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")
    if x >= 0:
        return 1 / (1 + math.exp(-1 * x))
    else:
        return math.exp(x) / (1 + math.exp(x))


def relu(x: float) -> float:
    """
    $f(x) =$ x if x is greater than 0, else 0

    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) .)
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x + EPS)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r"If $f = log$ as above, compute $d \times f'(x)$"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")
    return d * 1 / (2.3025 * x)


def inv(x: float) -> float:
    "$f(x) = 1/x$"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")
    return 1 / x


def inv_back(x: float, d: float) -> float:
    r"If $f(x) = 1/x$ compute $d \times f'(x)$"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")
    return -1.0 * d * (1 / x**2)


def relu_back(x: float, d: float) -> float:
    r"If $f = relu$ compute $d \times f'(x)$"
    # TODO: Implement for Task 0.1.
    if x < 0:
        return 0.0
    else:
        return d * 1.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
        A function that takes a list, applies `fn` to each element, and returns a
         new list
    """

    # TODO: Implement for Task 0.3.
    # raise NotImplementedError("Need to implement for Task 0.3")
    def new_func(ls1: Iterable[float]) -> Iterable[float]:
        result_arr = []
        for i in ls1:
            result_arr.append(fn(i))
        return result_arr

    return new_func


def negList(ls: Iterable[float]) -> Iterable[float]:
    "Use `map` and `neg` to negate each element in `ls`"
    # TODO: Implement for Task 0.3.
    # raise NotImplementedError("Need to implement for Task 0.3")
    obj = map(neg)
    return obj(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
        Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """

    # TODO: Implement for Task 0.3.
    # raise NotImplementedError("Need to implement for Task 0.3")
    def new_func(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        result_arr = []
        list_ls1 = list(ls1)
        list_ls2 = list(ls2)
        for i in range(0, len(list_ls1)):
            result_arr.append(fn(list_ls1[i], list_ls2[i]))
        return result_arr

    return new_func


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    "Add the elements of `ls1` and `ls2` using `zipWith` and `add`"
    # TODO: Implement for Task 0.3.
    # raise NotImplementedError("Need to implement for Task 0.3")
    obj = zipWith(add)
    return obj(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
        Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """

    # TODO: Implement for Task 0.3.
    # raise NotImplementedError("Need to implement for Task 0.3")
    def new_func(ls: Iterable[float]) -> float:
        cur = 0.0
        list_ls = list(ls)
        if len(list_ls) == 0:
            return 0
        else:
            for i in range(0, len(list_ls)):
                if i > 0:
                    cur = fn(cur, list_ls[i])
                else:
                    cur = fn(start, list_ls[i])
            return cur

    return new_func


def sum(ls: Iterable[float]) -> float:
    "Sum up a list using `reduce` and `add`."
    # TODO: Implement for Task 0.3.
    # raise NotImplementedError("Need to implement for Task 0.3")
    obj = reduce(add, 0)
    return obj(ls)


def prod(ls: Iterable[float]) -> float:
    "Product of a list using `reduce` and `mul`."
    # TODO: Implement for Task 0.3.
    # raise NotImplementedError("Need to implement for Task 0.3")
    obj = reduce(mul, 1)
    return obj(ls)