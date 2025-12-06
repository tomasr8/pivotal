import math
from typing import TypeAlias

from pivotal.expressions import Constraint, Expression
from pivotal.simplex import ProgramType, solve


ResultType: TypeAlias = tuple[float, dict[str, float]]


def optimize(
    _type: ProgramType,
    objective: Expression,
    constraints: list[Constraint] | tuple[Constraint, ...],
    max_iterations: float,
    tolerance: float,
) -> ResultType:
    value, all_variables, all_var_names, user_defined_vars = solve(
        _type, objective, constraints, max_iterations=max_iterations, tolerance=tolerance
    )
    value = value if _type == ProgramType.MIN else -value

    # Map user-defined variables to their values
    result_vars = {}
    for var_name in user_defined_vars:
        idx = all_var_names.index(var_name)
        result_vars[var_name] = all_variables[idx]

    return value, result_vars


def minimize(
    objective: Expression,
    constraints: list[Constraint] | tuple[Constraint, ...],
    *,
    max_iterations: float = math.inf,
    tolerance: float = 1e-6,
) -> ResultType:
    return optimize(ProgramType.MIN, objective, constraints, max_iterations, tolerance)


def maximize(
    objective: Expression,
    constraints: list[Constraint] | tuple[Constraint, ...],
    *,
    max_iterations: float = math.inf,
    tolerance: float = 1e-6,
) -> ResultType:
    return optimize(ProgramType.MAX, objective, constraints, max_iterations, tolerance)
