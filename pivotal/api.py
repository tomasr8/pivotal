import math
from typing import TypeAlias

from pivotal._expressions import Constraint, Expression, get_variable_names
from pivotal.simplex import ProgramType, solve


ResultType: TypeAlias = tuple[float, dict[str, float]]


def optimize(
    _type: ProgramType, objective: Expression, constraints: list[Constraint], max_iterations: int, tolerance: float
) -> ResultType:
    value, variables = solve(_type, objective, constraints, max_iterations=max_iterations, tolerance=tolerance)
    value = value if _type == ProgramType.MIN else -value

    user_defined_vars = get_variable_names((objective, *constraints))
    variables = variables[: len(user_defined_vars)]
    return value, ({user_defined_vars[i]: v for i, v in enumerate(variables)})


def minimize(
    objective: Expression, constraints: list[Constraint], *, max_iterations: int = math.inf, tolerance: float = 1e-6
) -> ResultType:
    return optimize(ProgramType.MIN, objective, constraints, max_iterations, tolerance)


def maximize(
    objective: Expression, constraints: list[Constraint], *, max_iterations: int = math.inf, tolerance: float = 1e-6
) -> ResultType:
    return optimize(ProgramType.MAX, objective, constraints, max_iterations, tolerance)
