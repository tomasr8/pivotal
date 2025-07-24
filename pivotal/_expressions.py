from __future__ import annotations

import abc
import itertools
import math
from collections import defaultdict
from typing import Self, TypeAlias, TypeVar


Assignment: TypeAlias = dict[str, float]


def evaluate(expr: Expression | float, assignment: Assignment) -> float:
    """Evaluate an expression given a variable assignment."""
    if isinstance(expr, Expression):
        return expr.evaluate(assignment)
    return expr


class EvaluationError(Exception):
    """Raised when an expression cannot be evaluated due to missing variable assignments."""


class Expression:
    """Base class for all expressions."""

    @abc.abstractmethod
    def evaluate(self, assignment: Assignment) -> float:
        """Evaluate the expression given a variable assignment."""
        raise NotImplementedError


class Constraint(Expression):
    def __init__(self, left: Expression | float, right: Expression | float) -> None:
        self.left = left
        self.right = right

    def _simplify(self) -> None:
        """Simplify the constraint by combining like terms."""
        self.left = Sum(self.left) if not isinstance(self.left, Sum) else self.left
        self.right = Sum(self.right) if not isinstance(self.right, Sum) else self.right


class Equal(Constraint):
    def __repr__(self) -> str:
        return f"{self.left} = {self.right}"

    def evaluate(self, assignment: Assignment) -> bool:
        return math.isclose(evaluate(self.left, assignment), evaluate(self.right, assignment), rel_tol=1e-9)


class GreaterOrEqual(Constraint):
    def __repr__(self) -> str:
        return f"{self.left} >= {self.right}"

    def evaluate(self, assignment: Assignment) -> bool:
        return evaluate(self.left, assignment) >= evaluate(self.right, assignment)


class LessOrEqual(Constraint):
    def __repr__(self) -> str:
        return f"{self.left} <= {self.right}"

    def evaluate(self, assignment: Assignment) -> bool:
        return evaluate(self.left, assignment) <= evaluate(self.right, assignment)


class ComparableMixin:
    def __eq__(self, other: Expression) -> Equal:
        return Equal(self, other)

    def __ge__(self, other: Expression) -> GreaterOrEqual:
        return GreaterOrEqual(self, other)

    def __le__(self, other: Expression) -> LessOrEqual:
        return LessOrEqual(self, other)


class Variable(Expression, ComparableMixin):
    _id_iter = itertools.count()

    def __init__(self, name: str | None = None, *, min=-math.inf, max=math.inf) -> None:
        if name is None:
            _id = next(self._id_iter)
            self.name = f"_{_id}"
        else:
            self.name = name

        if min == math.inf:
            msg = "Variable minimum cannot be infinity"
            raise ValueError(msg)
        if max == -math.inf:
            msg = "Variable maximum cannot be negative infinity"
            raise ValueError(msg)
        if min > max:
            msg = "Variable minimum cannot be greater than maximum"
            raise ValueError(msg)

        self.min = min
        self.max = max

    def __abs__(self) -> Abs:
        return Abs(self)

    def __neg__(self) -> Self:
        return Multiplication(self, -1)

    def __add__(self, other: float | Self | Sum | Abs) -> float | Self | Sum:
        match other:
            case Variable(name=name) if name == self.name:
                return Multiplication(self, 2)
            case int() | float() if other == 0:
                return self
            case Variable() | Abs() | int() | float():
                return Sum(self, other)
            case Sum(elts=elts):
                return Sum(self, *elts)
            case Multiplication(expr=Variable(name=name), coeff=coeff):
                if name == self.name:
                    return Multiplication(self, coeff + 1)
                return Sum(self, other)
            case Multiplication(expr=Abs(), coeff=coeff):
                return Sum(self, other)
            case _:
                return NotImplemented

    def __radd__(self, other: float) -> Self | Sum:
        match other:
            case int() | float():
                if other == 0:
                    return self
                return Sum(other, self)
            case _:
                return NotImplemented

    def __sub__(self, other: float | Self | Sum | Abs) -> float | Self | Sum:
        return self.__add__(-other)

    def __rsub__(self, other: float) -> Self | Sum:
        return (-self).__radd__(other)

    def __mul__(self, other: float) -> float | Self:
        match other:
            case int() | float():
                if other == 0:
                    return 0
                if other == 1:
                    return self
                return Multiplication(self, coeff=other)
            case _:
                return NotImplemented

    __rmul__ = __mul__

    def __repr__(self) -> str:
        return self.name

    def evaluate(self, assignment: Assignment) -> float:
        value = assignment.get(self.name)
        if value is None:
            msg = f"Variable '{self.name}' is not assigned."
            raise EvaluationError(msg)
        return value


class Multiplication(Expression, ComparableMixin):
    def __init__(self, expr: Variable | Abs, coeff: float) -> None:
        self.expr = expr
        self.coeff = coeff

    def __repr__(self) -> str:
        if self.coeff == 1:
            return repr(self.expr)
        if self.coeff == -1:
            return f"-{self.expr}"
        return f"{self.coeff}*{self.expr}"

    def __abs__(self) -> Abs:
        return Multiplication(Abs(self.expr), abs(self.coeff))

    def __neg__(self) -> Self:
        return Multiplication(self.expr, -self.coeff)

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
            case Variable():
                return Sum(self, other)
            case Multiplication():
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
                return Multiplication(self.expr, self.coeff * other)
            case _:
                return NotImplemented

    __rmul__ = __mul__

    def evaluate(self, assignment: Assignment) -> float:
        value = self.expr.evaluate(assignment)
        return value * self.coeff


class Sum(Expression, ComparableMixin):
    def __init__(self, *elts: list[Expression | float]) -> None:
        self.elts = self._simplify(elts)

    def __abs__(self) -> Abs:
        return Abs(self)

    def __neg__(self) -> Self:
        return Sum(*(-expr for expr in self.elts))

    def __add__(self, other: float | Variable | Self | Abs) -> Self:
        match other:
            case Sum(elts=elts):
                return Sum(*self.elts, *elts)
            case int() | float() if other == 0:
                return self
            case Variable() | Abs() | int() | float():
                return Sum(*self.elts, other)
            case Multiplication():
                return Sum(*self.elts, other)
            case _:
                return NotImplemented

    def __sub__(self, other: float | Variable | Self | Abs) -> Self:
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
        if not self.elts:
            return ""
        elts = [repr(x) for x in sorted(self.elts, key=self._sort_key)]
        out = elts[0]
        for elt in elts[1:]:
            if elt.startswith("-"):
                out += f" - {elt[1:]}"
            else:
                out += " + " + elt
        return out

    @staticmethod
    def _flatten(elts: list[Expression | float]) -> list[Expression | float]:
        """Flatten nested sums into a single list of expressions."""
        flattened = []
        for elt in elts:
            if isinstance(elt, Sum):
                flattened.extend(elt.elts)
            else:
                flattened.append(elt)
        return flattened

    @classmethod
    def _simplify(cls, elts: list[Expression | float]) -> list[Expression | float]:
        """Simplify the sum by combining like terms."""
        if not elts:
            return []

        elts = cls._flatten(elts)

        variables = {}
        variable_coeffs = defaultdict(int)
        abs_coeffs = defaultdict(int)
        const_coeff = 0
        for elt in elts:
            match elt:
                case Variable(name=name) as v:
                    variables[name] = v
                    variable_coeffs[name] += 1
                case Abs(var=Variable(name=name) as v):
                    variables[name] = v
                    abs_coeffs[name] += 1
                case Multiplication(expr=Variable(name=name) as v, coeff=coeff):
                    variables[name] = v
                    variable_coeffs[name] += coeff
                case Multiplication(expr=Abs(var=Variable(name=name) as v), coeff=coeff):
                    variables[name] = v
                    abs_coeffs[name] += coeff
                case int() | float() as c:
                    const_coeff += c
                case _:
                    continue
        # Rebuild the simplified expression
        simplified = []
        for name, coeff in variable_coeffs.items():
            if coeff != 0:
                simplified.append(variables[name] * coeff)
        for name, coeff in abs_coeffs.items():
            if coeff != 0:
                simplified.append(Abs(variables[name]) * coeff)
        if const_coeff != 0:
            simplified.append(const_coeff)
        if len(simplified) == 1:
            return [simplified[0]]
        return simplified

    @staticmethod
    def _sort_key(expr: Expression) -> str:
        match expr:
            case Variable(name=name):
                return (0, name)
            case Abs(var=Variable(name=name)):
                return (1, name)
            case Multiplication(expr=Variable(name=name)):
                return (2, name)
            case int() | float():
                return (3, "")
            case _:
                msg = f"Unsupported expression type: {type(expr)}"
                raise TypeError(msg)

    def evaluate(self, assignment: Assignment) -> float:
        """Evaluate the sum given a variable assignment."""
        return sum(evaluate(elt, assignment) for elt in self.elts)


class Abs(Expression):
    def __init__(self, var: Variable) -> None:
        self.var = var

    def __abs__(self) -> Self:
        return self

    def __neg__(self) -> Self:
        return Multiplication(self, -1)

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
            case Variable():
                return Sum(self, other)
            case Multiplication():
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
                return Multiplication(self, other)
            case _:
                return NotImplemented

    __rmul__ = __mul__

    def __repr__(self) -> str:
        return f"|{self.var}|"

    def evaluate(self, assignment: Assignment) -> float:
        """Evaluate the absolute value expression given a variable assignment."""
        value = self.var.evaluate(assignment)
        return abs(value)


def get_variable_coeffs(elem: Expression) -> tuple[dict[str, float], float]:
    c = 0
    variables = defaultdict(float)
    match elem:
        case int() | float() as const:
            c += const
        case Variable(name=name):
            variables[name] += 1
        case Multiplication(expr=Variable(name=name), coeff=coeff):
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


def get_variable_names(elems: list[Expression]) -> list[str]:
    variables = set()
    for elem in elems:
        coeffs, _ = get_variable_coeffs(elem)
        variables |= set(coeffs.keys())
    return sorted(variables)


def get_variables(elems: Expression | list[Expression]) -> list[Variable]:
    variables = {}

    class VariableCollector(ExpressionVisitor):
        def visit_variable(self, var: Variable) -> None:
            variables[var.name] = var

    if isinstance(elems, Expression):
        elems = [elems]
    for elem in elems:
        VariableCollector().visit(elem)
    return list(variables.values())


_T = TypeVar("_T", bound=Expression)


def substitute(elem: _T, old: Variable, new: Expression) -> _T:
    class Substitutor(ExpressionTransformer):
        def visit_variable(self, var: Variable) -> Expression:
            if var.name == old.name:
                return new
            return var

    return Substitutor().visit(elem)


class ExpressionVisitor:
    """Base class for visiting expressions."""

    def visit(self, elem: Expression) -> None:
        """Visit the given expression or constraint."""
        match elem:
            case int() | float():
                self.visit_constant(elem)
            case Variable():
                self.visit_variable(elem)
            case Multiplication():
                self.visit_multiplication(elem)
            case Abs():
                self.visit_abs(elem)
            case Sum():
                self.visit_sum(elem)
            case Constraint():
                self.visit_constraint(elem)
            case _:
                msg = f"Unsupported expression type: {type(elem)} {elem!r}"
                raise TypeError(msg)

    def visit_constant(self, const: float) -> None:
        """Visit a constant."""

    def visit_variable(self, var: Variable) -> None:
        """Visit a variable."""

    def visit_multiplication(self, mul: Multiplication) -> None:
        """Visit a multiplication."""
        self.visit(mul.expr)

    def visit_abs(self, abs: Abs) -> None:
        """Visit an absolute value."""
        self.visit(abs.var)

    def visit_sum(self, sum: Sum) -> None:
        """Visit a sum."""
        for elt in sum.elts:
            self.visit(elt)

    def visit_constraint(self, constraint: Constraint) -> None:
        """Visit a constraint."""
        self.visit(constraint.left)
        self.visit(constraint.right)


class ExpressionTransformer:
    """Base class for transforming expressions."""

    def visit(self, elem: Expression) -> Expression:
        """Transform the given expression or constraint."""
        match elem:
            case Variable():
                return self.visit_variable(elem)
            case Multiplication():
                return self.visit_multiplication(elem)
            case Abs():
                return self.visit_abs(elem)
            case Sum():
                return self.visit_sum(elem)
            case Constraint():
                return self.visit_constraint(elem)
            case _:
                msg = f"Unsupported expression type: {type(elem)}"
                raise TypeError(msg)

    def visit_variable(self, var: Variable) -> Variable:
        """Transform a variable."""
        return var

    def visit_multiplication(self, mul: Multiplication) -> Multiplication:
        """Transform a multiplication."""
        new_expr = self.visit(mul.expr)
        return Multiplication(new_expr, mul.coeff)

    def visit_abs(self, abs: Abs) -> Abs:
        """Transform an absolute value."""
        new_var = self.visit(abs.var)
        return Abs(new_var)

    def visit_sum(self, sum: Sum) -> Sum:
        """Transform a sum."""
        new_elts = [self.visit(elt) for elt in sum.elts]
        return Sum(*new_elts)

    def visit_constraint(self, constraint: Constraint) -> Constraint:
        """Transform a constraint."""
        new_left = self.visit(constraint.left)
        new_right = self.visit(constraint.right)
        return type(constraint)(new_left, new_right)
