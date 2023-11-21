from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple


class Module:
    """
    Modules form a tree that store parameters and other
    submodules. They make up the basis of neural network stacks.

    Attributes:
        _modules : Storage of the child modules
        _parameters : Storage of the module's parameters
        training : Whether the module is in training mode or evaluation mode

    """

    _modules: Dict[str, Module]
    _parameters: Dict[str, Parameter]
    training: bool

    def __init__(self) -> None:
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self) -> Sequence[Module]:
        "Return the direct child modules of this module."
        m: Dict[str, Module] = self.__dict__["_modules"]
        return list(m.values())

    def train(self) -> None:
        "Set the mode of this module and all descendent modules to `train`."
        # raise NotImplementedError("Need to include this file from past assignment.")

        self.training = True
        for i in self.modules():
            i.training = True

    def eval(self) -> None:
        "Set the mode of this module and all descendent modules to `eval`."

        # raise NotImplementedError("Need to include this file from past assignment.")
        self.training = False
        for i in self.modules():
            i.training = False

    def named_parameters(
        self, dict_param: dict[str, Parameter] = {}, prefix: str = "", iter: int = 0
    ) -> Sequence[Tuple[str, Parameter]]:
        """
        Collect all the parameters of this module and its descendents.


        Returns:
            The name and `Parameter` of each ancestor parameter.
        """
        # raise NotImplementedError("Need to include this file from past assignment.")
        if iter == 0:
            dict_param = {}

        tup_allpar_nam: Tuple[str, ...] = ()
        tup_allpar_val: Tuple[Parameter, ...] = ()
        q: Dict[str, Module] = self.__dict__["_modules"]

        tup_1 = tuple(q.keys())
        mod = self.modules()
        for i in range(len(mod)):
            r: Dict[str, Parameter] = mod[i].__dict__["_parameters"]
            if mod[i].modules() is not None:
                dict_param.update(
                    mod[i].named_parameters(dict_param, tup_1[i], iter + 1)
                )
            tup_keys = list(r.keys())
            tup_val = list(r.values())
            tup_allpar_nam = tuple(tup_keys)
            tup_allpar_val = tuple(tup_val)
            print("start", tup_allpar_nam, tup_1[i])
            print("end")
            for j in range(len(tup_1)):
                if len(tup_allpar_nam) > 1:
                    print("arr", tup_allpar_nam, tup_1[j])
                    for k in range(len(tup_allpar_nam)):
                        if prefix:
                            if j <= len(tup_allpar_nam) - 1:
                                dict_param[
                                    prefix + "." + tup_1[i] + "." + tup_allpar_nam[k]
                                ] = tup_allpar_val[k]
                        else:
                            if j <= len(tup_allpar_nam) - 1:
                                dict_param[
                                    tup_1[i] + "." + tup_allpar_nam[k]
                                ] = tup_allpar_val[k]
                else:
                    if prefix:
                        if j <= len(tup_allpar_nam) - 1:
                            dict_param[
                                prefix + "." + tup_1[i] + "." + tup_allpar_nam[j]
                            ] = tup_allpar_val[j]

                    else:
                        if j <= len(tup_allpar_nam) - 1:
                            dict_param[
                                tup_1[i] + "." + tup_allpar_nam[j]
                            ] = tup_allpar_val[j]
        iter = iter + 1
        if iter == 1:
            m: Dict[str, Parameter] = self.__dict__["_parameters"]
            list_3 = [(k, v) for k, v in m.items()]
            dict_param.update(list_3)
        print(list(dict_param.items()))
        return list(dict_param.items())

    def parameters(self) -> Sequence[Parameter]:
        "Enumerate over all the parameters of this module and its descendents."
        # raise NotImplementedError("Need to include this file from past assignment.")
        list_par = list()
        for i in self.named_parameters():
            list_par.append(i[1])
        return list(list_par)

    def add_parameter(self, k: str, v: Any) -> Parameter:
        """
        Manually add a parameter. Useful helper for scalar parameters.

        Args:
            k: Local name of the parameter.
            v: Value for the parameter.

        Returns:
            Newly created parameter.
        """
        val = Parameter(v, k)
        self.__dict__["_parameters"][k] = val
        return val

    def __setattr__(self, key: str, val: Parameter) -> None:
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Any:
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        def _addindent(s_: str, numSpaces: int) -> str:
            s2 = s_.split("\n")
            if len(s2) == 1:
                return s_
            first = s2.pop(0)
            s2 = [(numSpaces * " ") + line for line in s2]
            s = "\n".join(s2)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Parameter:
    """
    A Parameter is a special container stored in a `Module`.

    It is designed to hold a `Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, x: Any, name: Optional[str] = None) -> None:
        self.value = x
        self.name = name
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def update(self, x: Any) -> None:
        "Update the parameter value."
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self) -> str:
        return repr(self.value)

    def __str__(self) -> str:
        return str(self.value)
