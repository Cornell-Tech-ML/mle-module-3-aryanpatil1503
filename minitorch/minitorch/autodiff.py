from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple
from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    # raise NotImplementedError("Need to implement for Task 1.1")

    vals_add = []
    vals_sub = []
    for i in range(len(vals)):
        if i == arg:
            vals_add.append(vals[i] + epsilon)
            vals_sub.append(vals[i] - epsilon)
        else:
            vals_add.append(
                vals[i],
            )
            vals_sub.append(
                vals[i],
            )
    result = (f(*vals_add) - f(*vals_sub)) / (2 * epsilon)
    return result


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    # raise NotImplementedError("Need to implement for Task 1.4")
    passed: List[Any] = []
    seen = set()

    def dfs(variable: Variable) -> None:
        if variable.unique_id in seen or variable.is_constant():
            return
        if not variable.is_leaf():
            for i in variable.parents:
                if not i.is_constant():
                    dfs(i)
        passed.insert(0, variable)
        seen.add(variable.unique_id)

    dfs(variable)
    return passed


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    # raise NotImplementedError("Need to implement for Task 1.4")

    sort_list = topological_sort(variable)
    dict_sort = {}
    dict_sort[variable.unique_id] = deriv
    for i in sort_list:
        if i.is_leaf():
            i.accumulate_derivative(dict_sort[i.unique_id])
        else:
            for var, der in i.chain_rule(dict_sort[i.unique_id]):
                if var.unique_id in dict_sort.keys():
                    dict_sort[var.unique_id] = dict_sort[var.unique_id] + der
                else:
                    dict_sort[var.unique_id] = der


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
