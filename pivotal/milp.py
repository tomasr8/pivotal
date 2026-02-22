from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pivotal.errors import Infeasible, NodeLimitReached, Unbounded
from pivotal.expressions import Constraint, ExprOrNumber, Variable, VarType
from pivotal.simplex import ProgramType
from pivotal.simplex import solve as lp_solve


if TYPE_CHECKING:
    import numpy as np


@dataclass
class BBNode:
    """A node in the branch-and-bound tree."""

    extra_constraints: list[Constraint]
    depth: int = 0
    bound: float = -math.inf
    node_id: int = 0


def solve_milp(
    _type: ProgramType,
    objective: ExprOrNumber,
    constraints: list[Constraint] | tuple[Constraint, ...],
    *,
    integer_vars: dict[str, VarType],
    max_iterations: float = math.inf,
    tolerance: float = 1e-6,
    node_limit: int = 10_000,
) -> tuple[float, np.ndarray, list[str], list[str]]:
    """
    Solve a Mixed-Integer Linear Program using branch and bound.

    Uses the existing LP simplex solver for relaxations at each node.
    Returns the same shape as simplex.solve():
        (value, solution, all_var_names, original_var_names)
    """
    constraints = list(constraints)

    # Incumbent: best integer-feasible solution found so far
    incumbent_value = math.inf if _type == ProgramType.MIN else -math.inf
    incumbent_result: tuple[float, np.ndarray, list[str], list[str]] | None = None

    # Priority queue: (priority, node_id, node)
    node_counter = 0
    root = BBNode(extra_constraints=[], depth=0, bound=-math.inf, node_id=node_counter)

    heap: list[tuple[float, int, BBNode]] = []
    heapq.heappush(heap, (_heap_priority(root.bound, _type), node_counter, root))

    nodes_explored = 0

    while heap:
        if nodes_explored >= node_limit:
            if incumbent_result is not None:
                break
            msg = f"Node limit ({node_limit}) reached without finding an integer-feasible solution."
            raise NodeLimitReached(msg)

        _, _, node = heapq.heappop(heap)
        nodes_explored += 1

        # Solve LP relaxation at this node
        node_constraints = constraints + node.extra_constraints

        try:
            value, solution, all_vars, orig_vars = lp_solve(
                _type,
                objective,
                node_constraints,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )
        except Infeasible:
            continue
        except Unbounded:
            raise

        # simplex.solve() negates value for MAX; un-negate for internal comparisons
        natural_value = value if _type == ProgramType.MIN else -value

        # Prune by bound: if this node's relaxation is worse than incumbent
        if _type == ProgramType.MIN and natural_value >= incumbent_value - tolerance:
            continue
        if _type == ProgramType.MAX and natural_value <= incumbent_value + tolerance:
            continue

        # Map variable names to solution values
        var_values = {name: solution[i] for i, name in enumerate(orig_vars)}

        # Check integrality
        branching_var = _select_branching_variable(var_values, integer_vars, tolerance)

        if branching_var is None:
            # All integer variables are integral — update incumbent
            if _is_better(natural_value, incumbent_value, _type, tolerance):
                incumbent_value = natural_value
                incumbent_result = (value, solution, all_vars, orig_vars)
            continue

        # Branch on the selected variable
        frac_value = var_values[branching_var]
        floor_val = math.floor(frac_value)
        ceil_val = math.ceil(frac_value)

        # Left child: branching_var <= floor(frac_value)
        left_constraint = Variable(branching_var) <= floor_val
        left_constraints = [*node.extra_constraints, left_constraint]

        # Right child: branching_var >= ceil(frac_value)
        right_constraint = Variable(branching_var) >= ceil_val
        right_constraints = [*node.extra_constraints, right_constraint]

        node_counter += 1
        left_node = BBNode(
            extra_constraints=left_constraints,
            depth=node.depth + 1,
            bound=natural_value,
            node_id=node_counter,
        )

        node_counter += 1
        right_node = BBNode(
            extra_constraints=right_constraints,
            depth=node.depth + 1,
            bound=natural_value,
            node_id=node_counter,
        )

        priority = _heap_priority(natural_value, _type)
        heapq.heappush(heap, (priority, left_node.node_id, left_node))
        heapq.heappush(heap, (priority, right_node.node_id, right_node))

    if incumbent_result is None:
        msg = "The MILP is infeasible — no integer-feasible solution exists."
        raise Infeasible(msg)

    return incumbent_result


def _heap_priority(bound: float, _type: ProgramType) -> float:
    """Convert a bound to a heap priority (lower = explored first)."""
    # For MIN: lower bound = better → use bound directly
    # For MAX: higher bound = better → negate
    return bound if _type == ProgramType.MIN else -bound


def _select_branching_variable(
    var_values: dict[str, float],
    integer_vars: dict[str, VarType],
    tolerance: float,
) -> str | None:
    """
    Select the most fractional integer variable to branch on.

    Returns None if all integer variables are integral.
    """
    best_var = None
    best_fractionality = -1.0

    for var_name in integer_vars:
        value = var_values.get(var_name, 0.0)
        frac = abs(value - round(value))

        if frac <= tolerance:
            continue

        # Most fractional: closest to 0.5
        fractionality = 0.5 - abs(frac - 0.5)
        if fractionality > best_fractionality:
            best_fractionality = fractionality
            best_var = var_name

    return best_var


def _is_better(new_val: float, old_val: float, _type: ProgramType, tolerance: float) -> bool:
    """Check if new_val is strictly better than old_val for the given optimization direction."""
    if _type == ProgramType.MIN:
        return new_val < old_val - tolerance
    return new_val > old_val + tolerance
