import math

from pivotal.expressions import Constraint, ExprOrNumber, collect_integer_variables
from pivotal.milp import solve_milp
from pivotal.simplex import ProgramType
from pivotal.simplex import solve as lp_solve


type ResultType = tuple[float, dict[str, float]]


def optimize(
    _type: ProgramType,
    objective: ExprOrNumber,
    constraints: list[Constraint] | tuple[Constraint, ...],
    *,
    max_iterations: float,
    tolerance: float,
    node_limit: int,
) -> ResultType:
    integer_vars = collect_integer_variables(objective, list(constraints))

    if integer_vars:
        value, all_variables, all_var_names, user_defined_vars = solve_milp(
            _type,
            objective,
            constraints,
            integer_vars=integer_vars,
            max_iterations=max_iterations,
            tolerance=tolerance,
            node_limit=node_limit,
        )
    else:
        value, all_variables, all_var_names, user_defined_vars = lp_solve(
            _type, objective, constraints, max_iterations=max_iterations, tolerance=tolerance
        )

    value = value if _type == ProgramType.MIN else -value

    # Map user-defined variables to their values
    result_vars = {}
    for var_name in user_defined_vars:
        idx = all_var_names.index(var_name)
        val = all_variables[idx]
        # Round integer/binary variables to exact integers
        if var_name in integer_vars:
            val = float(round(val))
        result_vars[var_name] = val

    return value, result_vars


def minimize(
    objective: ExprOrNumber,
    constraints: list[Constraint] | tuple[Constraint, ...],
    *,
    max_iterations: float = math.inf,
    tolerance: float = 1e-6,
    node_limit: int = 10_000,
) -> ResultType:
    return optimize(
        ProgramType.MIN,
        objective,
        constraints,
        max_iterations=max_iterations,
        tolerance=tolerance,
        node_limit=node_limit,
    )


def maximize(
    objective: ExprOrNumber,
    constraints: list[Constraint] | tuple[Constraint, ...],
    *,
    max_iterations: float = math.inf,
    tolerance: float = 1e-6,
    node_limit: int = 10_000,
) -> ResultType:
    return optimize(
        ProgramType.MAX,
        objective,
        constraints,
        max_iterations=max_iterations,
        tolerance=tolerance,
        node_limit=node_limit,
    )
