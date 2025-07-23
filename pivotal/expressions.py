import math
import itertools
from collections import defaultdict
from typing import Literal, Self, TypeVar


class Expression:
    """Base class for all expressions."""


class Constraint:
    def __init__(self, left: Expression, right: Expression) -> None:
        self.left = left
        self.right = right


class Equal(Constraint):
    def __repr__(self) -> str:
        return f"{self.left} = {self.right}"


class GreaterOrEqual(Constraint):
    def __repr__(self) -> str:
        return f"{self.left} >= {self.right}"


class LessOrEqual(Constraint):
    def __repr__(self) -> str:
        return f"{self.left} <= {self.right}"


class ComparableMixin:
    def __eq__(self, other: Expression) -> Equal:
        return Equal(self, other)

    def __ge__(self, other: Expression) -> GreaterOrEqual:
        return GreaterOrEqual(self, other)

    def __le__(self, other: Expression) -> LessOrEqual:
        return LessOrEqual(self, other)


class Variable(Expression, ComparableMixin):
    _id_iter = itertools.count()

    def __init__(self, name: str | None = None, *, coeff: float = 1.0, min=-math.inf, max=math.inf) -> None:
        if name is None:
            _id = next(self._id_iter)
            self.name = f"_{_id}"
        else:
            self.name = name
        self.coeff = coeff

        if min == math.inf:
            raise ValueError("Variable minimum cannot be infinity")
        if max == -math.inf:
            raise ValueError("Variable maximum cannot be negative infinity")
        if min > max:
            raise ValueError("Variable minimum cannot be greater than maximum")

        self.min = min
        self.max = max

    def __abs__(self) -> "Abs":
        return Abs(self)

    def __neg__(self) -> Self:
        return Variable(name=self.name, coeff=-self.coeff)

    def __add__(self, other: "float | Self | Sum | Abs") -> "float | Self | Sum":
        match other:
            case Variable(name=name, coeff=coeff) if name == self.name:
                if self.coeff + coeff == 0:
                    return 0
                return Variable(name=name, coeff=self.coeff + coeff)
            case int() | float() if other == 0:
                return self
            case Variable() | Abs() | int() | float():
                return Sum(self, other)
            case Sum(elts=elts):
                return Sum(self, *elts)
            case _:
                return NotImplemented

    def __radd__(self, other: float) -> "Self | Sum":
        match other:
            case int() | float():
                if other == 0:
                    return self
                return Sum(other, self)
            case _:
                return NotImplemented

    def __sub__(self, other: "float | Self | Sum | Abs") -> "float | Self | Sum":
        return self.__add__(-other)

    def __rsub__(self, other: float) -> "Self | Sum":
        return (-self).__radd__(other)

    def __mul__(self, other: float) -> "float | Self":
        match other:
            case int() | float():
                if other == 0:
                    return 0
                if other == 1:
                    return self
                return Variable(self.name, coeff=self.coeff * other)
            case _:
                return NotImplemented

    __rmul__ = __mul__

    def __repr__(self) -> str:
        if self.coeff == 1.0:
            return self.name
        if self.coeff == -1.0:
            return f"-{self.name}"
        return f"{self.coeff}*{self.name}"


class Multiplication(Expression, ComparableMixin):
    def __init__(self, var: Variable, coeff: float | int) -> None:
        self.var = var
        self.coeff = coeff

    def __repr__(self) -> str:
        return f"({self.var}) * ({self.coeff})"

    def __abs__(self) -> "Abs":
        return Abs(self)

    def __neg__(self) -> Self:
        return Multiplication(self.var, -self.coeff)
    
    def __add__(self, other: float | Variable | Self | Abs) -> Self | Sum:
        match other:
            case Abs():
                return Sum(self, other)
            case Sum(elts=elts):
                return Sum(self, *elts)
            case int() | float():
                if other == 0:
                    return self
                return Sum(self, other)
            case Variable() | int() | float():
                return Sum(self, other)
            case _:
                return NotImplemented
            
    def __sub__(self, other: float | Variable | Self | Abs) -> Self | Sum:
        return self.__add__(-other)
    
    def __radd__(self, other: float) -> Self | Sum:
        match other:
            case int() | float():
                if other == 0:
                    return self
                return Sum(other, self)
            case _:
                return NotImplemented
            
    def __rsub__(self, other: float) -> Self | Sum:
        return (-self).__radd__(other)
    
    def __mul__(self, other: float) -> float | Self:
        match other:
            case int() | float():
                if other == 1:
                    return self
                if other == 0:
                    return 0
                return Multiplication(self.var, self.coeff * other)
            case _:
                return NotImplemented
            
    __rmul__ = __mul__


class Sum(Expression, ComparableMixin):
    def __init__(self, *elts: list[Expression]) -> None:
        self.elts = elts

    def __abs__(self) -> "Abs":
        return Abs(self)

    def __neg__(self) -> Self:
        return Sum(*(-expr for expr in self.elts))

    def __add__(self, other: "float | Variable | Self | Abs") -> Self:
        match other:
            case Sum(elts=elts):
                return Sum(*self.elts, *elts)
            case int() | float() if other == 0:
                return self
            case Variable() | Abs() | int() | float():
                return Sum(*self.elts, other)
            case _:
                return NotImplemented

    def __sub__(self, other: "float | Variable | Self | Abs") -> Self:
        return self.__add__(-other)

    def __radd__(self, other: float) -> Self:
        match other:
            case int() | float() if other == 0:
                if other == 0:
                    return self
                return Sum(other, *self.elts)
            case _:
                return NotImplemented

    def __rsub__(self, other: float) -> Self:
        return (-self).__radd__(other)

    def __mul__(self, other: float) -> float | Self:
        match other:
            case int() | float():
                if other == 0:
                    return 0
                if other == 1:
                    return self
                return Sum(*(other * elt for elt in self.elts))
            case _:
                return NotImplemented

    __rmul__ = __mul__

    def __repr__(self) -> str:
        return " + ".join(repr(x) for x in self.elts)
    
    # def _simplify(self) -> Self:
    #     """Simplify the sum by combining like terms."""
    #     coeffs = defaultdict(float)
    #     const = 0
    #     for elt in self.elts:
    #         match elt:
    #             case Variable(name=name, coeff=coeff):
    #                 coeffs[name] += coeff
    #             case int() | float() as c:
    #                 const += c
    #             case _:
    #                 continue
    #     # Rebuild the simplified expression
    #     new_elts = [Variable(name, coeff) for name, coeff in coeffs.items() if coeff != 0]
    #     if const != 0:
    #         new_elts.append(const)
    #     return Sum(*new_elts) if new_elts else 0


class Abs(Expression):
    def __init__(self, arg: Expression, sign: Literal[-1, 1] = 1) -> None:
        self.arg = arg
        self.sign = sign

    def __abs__(self) -> Self:
        return Abs(self.arg)

    def __neg__(self) -> Self:
        return Abs(self.arg, -self.sign)

    def __add__(self, other: float | Variable | Self | Sum) -> Self | Sum:
        match other:
            case Abs():
                return Sum(self, other)
            case Sum(elts=elts):
                return Sum(self, *elts)
            case int() | float():
                if other == 0:
                    return self
                return Sum(self, other)
            case Variable() | int() | float():
                return Sum(self, other)
            case _:
                return NotImplemented

    def __sub__(self, other: float | Variable | Self | Sum) -> Self | Sum:
        return self.__add__(-other)

    def __radd__(self, other: float) -> Self | Sum:
        match other:
            case int() | float():
                if other == 0:
                    return self
                return Sum(other, self)
            case _:
                return NotImplemented

    def __rsub__(self, other: float) -> Self | Sum:
        return (-self).__radd__(other)

    def __mul__(self, other: float) -> float | Self:
        match other:
            case int() | float():
                if other == 1:
                    return self
                if other == 0:
                    return 0
                if other > 0:
                    return Abs(other * self.arg)
                return Abs(abs(other) * self.arg, sign=-1)
            case _:
                return NotImplemented

    __rmul__ = __mul__

    def __repr__(self) -> str:
        sign = "-" if self.sign == -1 else ""
        return f"{sign}|{self.arg}|"


def get_variable_coeffs(elem: Expression | Constraint) -> tuple[dict[str, float], float]:
    c = 0
    variables = defaultdict(float)
    match elem:
        case int() | float() as const:
            c += const
        case Variable(name=name, coeff=coeff):
            variables[name] += coeff
        case Sum(elts=elts):
            for expr in elts:
                _variables, _c = get_variable_coeffs(expr)
                c += _c
                for v in _variables:
                    variables[v] += _variables[v]
        case Constraint(left=left, right=right):
            vars_left, c_left = get_variable_coeffs(left)
            vars_right, c_right = get_variable_coeffs(right)

            c += c_right - c_left
            for v in vars_left:
                variables[v] += vars_left[v]
            for v in vars_right:
                variables[v] -= vars_right[v]
    return variables, c


def get_variable_names(elems: list[Expression | Constraint]) -> list[str]:
    variables = set()
    for elem in elems:
        coeffs, _ = get_variable_coeffs(elem)
        variables |= set(coeffs.keys())
    return sorted(variables)


def get_variables(elem: Expression | Constraint) -> set[Variable]:
    variables = set()
    match elem:
        case Variable() as v:
            variables.add(v)
        case Sum(elts=elts):
            for expr in elts:
                variables |= get_variables(expr)
        case Constraint(left=left, right=right):
            vars_left = get_variables(left)
            vars_right = get_variables(right)
            variables |= vars_left
            variables |= vars_right
    return variables


_T = TypeVar("T", bound=Expression | Constraint)


def substitute(elem: _T, old: Variable, new: Expression) -> _T:
    match elem:
        case Variable() as v:
            if v.name == old.name:
                return new
            return v
        case Sum(elts=elts):
            return Sum(*(substitute(elt, old, new) for elt in elts))
        case Constraint(left=left, right=right) as c:
            return type(c)(
                substitute(left, old, new),
                substitute(right, old, new)
            )
        case other:
            # If no match, return the original element unchanged
            return other