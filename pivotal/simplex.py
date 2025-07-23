import math
from os import replace
import warnings
from collections.abc import Callable
from enum import Enum, auto
from typing import Literal, TypeVar

import numpy as np

from pivotal.errors import Infeasible, Unbounded
from pivotal.expressions import (
    Constraint,
    Equal,
    Expression,
    GreaterOrEqual,
    LessOrEqual,
    Variable,
    get_variable_coeffs,
    get_variable_names,
    get_variables,
    substitute,
)


class ProgramType(Enum):
    MIN = auto()
    MAX = auto()


_T = TypeVar("_T")


def suppress_divide_by_zero_warning(fn: Callable[..., _T]) -> None:
    def _fn_suppressed(*args, **kwargs) -> _T:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return fn(*args, **kwargs)

    return _fn_suppressed


class Pivots:
    """
    Helper class to manage the positions of pivots in the tableau.

    The pivots can be accessed by row or column index using either `rc` or `cr` respectively.
    The row and columns indices are based on the full M matrix. If you want to get the position
    in the A matrix, you need to subtract 1 from the row index.
    """

    def __init__(self, cr: dict[int, int]) -> None:
        self.cr = cr
        self.rc = {cr[c]: c for c in cr}

    def get(self, *, row: int | None = None, column: int | None = None) -> int:
        if row is not None:
            return self.rc[row]
        return self.cr[column]

    def get_pivot(self, *, row: int | None = None, column: int | None = None) -> tuple[int, int]:
        if row is not None:
            return row, self.rc[row]
        return self.cr[column], column

    def set(self, *, row: int, column: int) -> None:
        self.rc[row] = column
        self.cr[column] = row

    def has(self, *, row: int | None = None, column: int | None = None) -> bool:
        if row is not None:
            return row in self.rc
        return column in self.cr

    def delete(self, *, row: int | None = None, column: int | None = None) -> None:
        if row is not None:
            column = self.rc[row]
            del self.rc[row]
            del self.cr[column]
        else:
            row = self.cr[column]
            del self.rc[row]
            del self.cr[column]

    def __repr__(self) -> str:
        return repr(self.rc)


class Tableau:
    def __init__(  # noqa: PLR0913
        self, A: np.ndarray, b: np.ndarray, c: np.ndarray, pivots: Pivots | None = None, *, tolerance: float = 1e-6
    ) -> None:
        self.M = np.block([[np.atleast_2d(c), np.atleast_2d(0)], [A, np.atleast_2d(b).T]])
        self.pivots = pivots if pivots else Pivots({})
        self.tolerance = tolerance

    @property
    def n_vars(self) -> int:
        return self.A.shape[1]

    @property
    def n_constraints(self) -> int:
        return self.A.shape[0]

    @property
    def A(self) -> np.ndarray:
        return self.M[1:, :-1]

    @property
    def b(self) -> np.ndarray:
        return self.M[1:, -1]

    @property
    def c(self) -> np.ndarray:
        return self.M[0, :-1]

    @property
    def value(self) -> float:
        """The value of the objective function."""
        return -self.M[0, -1]

    @property
    def solution(self) -> np.ndarray:
        solution = np.zeros(self.n_vars, dtype=float)
        for col in range(self.n_vars):
            if self.pivots.has(column=col):
                row = self.pivots.get(column=col)
                solution[col] = self.b[row - 1]
        return solution

    def __repr__(self) -> str:
        return f"Tableau:\n{self.M}\nPivots:\n{self.pivots}"


def solve(
    _type: ProgramType, objective: Expression, constraints: list[Constraint], *, max_iterations: int, tolerance: float
) -> tuple[float, np.ndarray]:
    """
    The main entrypoint to the simplex algorithm.

    This function accepts a high-level representation of the LP and returns the optimal value and the solution.
    """
    program = as_tableau(_type, objective, constraints, tolerance=tolerance)
    return _solve(program, max_iterations=max_iterations, tolerance=tolerance)


def _solve(program: Tableau, *, max_iterations: int, tolerance: float) -> tuple[float, np.ndarray]:
    aux_program = create_phase_one_program(program)
    run_simplex(aux_program)

    # The auxiliary problem minimizes the sum of slack variables.
    # If the sum is not zero, the original problem is infeasible.
    if not is_zero(aux_program.value, tolerance):
        msg = "The program is infeasible, try removing some constraints."
        raise Infeasible(msg)

    if has_slack_variables_in_basis(aux_program, program.n_vars):
        # Either the problem is degenerate (one or more original variables is zero)
        # or there are redundant constraints (or both).
        # Before we proceed with the second phase, we need to remove the redundant constraints and
        # move the slack variables out of the base.
        remove_redundant_constraints(aux_program, program.n_vars, tolerance)
        move_pivots_from_slack_variables(aux_program, program.n_vars, tolerance)

    new_prog = create_phase_two_program(program, aux_program)
    run_simplex(new_prog, max_iterations=max_iterations)
    return new_prog.value, new_prog.solution


def as_tableau(
    _type: ProgramType, objective: Expression, constraints: list[Constraint], *, tolerance: float
) -> Tableau:
    """
    Convert the high-level representation of the LP to a Tableau backed by a numpy matrix.

    The LP is converted to its canonical form in the process.
    """

    variables = get_variables((objective, *constraints))
    substitutions = {}
    for v in variables:
        if v.min == 0 and v.max == math.inf:
            # This is what simplex wants, no need to do anything
            continue
        if v.min == -math.inf and v.max == math.inf:
            # TODO: free var
            pass
        elif v.min == v.max:
            if v.min >= 0:
                constraints = [*constraints, v == v.min]
            else:
                new_var = Variable(coeff=-v.coeff, min=-v.min, max=-v.max)
                constraints = substitute(constraints, v, new_var)
                objective = replace(objective, v, new_var)
                constraints = [*constraints, new_var == -v.min]
        elif v.max == math.inf and v.min > 0:
            constraints = [*constraints, v >= v.min]
        elif v.min == -math.inf and v.max < 0:
            new_var = Variable(coeff=-v.coeff, min=-v.max, max=-v.min)
            constraints = replace(constraints, v, new_var)
            objective = replace(objective, v, new_var)
            constraints = [*constraints, new_var <= -v.max]


    variables = get_variable_names((objective, *constraints))
    n_vars = len(variables)
    n_constraints = len(constraints)
    A = np.zeros((n_constraints, n_vars), dtype=float)
    b = np.zeros(n_constraints, dtype=float)
    c = np.zeros(n_vars, dtype=float)
    constraint_types = np.zeros(n_constraints, dtype=int)

    # Create the objective function row
    coeffs, _ = get_variable_coeffs(objective)
    for name, coeff in coeffs.items():
        i = variables.index(name)
        c[i] = coeff

    constraint_map = {
        Equal: 0,
        GreaterOrEqual: 1,
        LessOrEqual: -1,
    }
    # Create the constraints
    for row, constraint in enumerate(constraints):
        coeffs, _b = get_variable_coeffs(constraint)
        for name, coeff in coeffs.items():
            column = variables.index(name)
            A[row, column] = coeff
        b[row] = _b
        constraint_types[row] = constraint_map[type(constraint)]

    A, b, c = canonicalize(_type, A, b, c, constraint_types)
    return Tableau(A, b, c, tolerance=tolerance)


def canonicalize(
    _type: ProgramType, A: np.ndarray, b: np.ndarray, c: np.ndarray, constraint_types: list[Literal[-1, 0, 1]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert the LP to the canonical form:

    min c^T*x
      s.t.
        Ax <= b
        x >= 0
        b >= 0
    """
    if _type == ProgramType.MAX:
        c *= -1

    # convert <= to >=
    sel = constraint_types == -1
    b[sel] *= -1
    A[sel] *= -1
    constraint_types[sel] = 1

    # convert >= to ==
    sel = constraint_types == 1
    n = np.sum(sel)
    if n > 0:
        A = np.hstack((A, np.zeros((A.shape[0], n))))
        c = np.hstack((c, np.zeros(n)))
        A[sel, -n:] = -np.eye(n)
        constraint_types[sel] = 0

    # convert -b to b
    sel = b < 0
    b[sel] *= -1
    A[sel] *= -1

    return A, b, c


def has_slack_variables_in_basis(aux_program: Tableau, n_original_vars: int) -> bool:
    return any(j >= n_original_vars for j in aux_program.pivots.cr)


def create_phase_one_program(program: Tableau) -> Tableau:
    """
    Create the auxiliary problem for the first phase of the simplex algorithm.

    This adds additional slack variables to create a valid basis.
    The sum of the slack variables is then minimized.

    For example, this problem:

    min 2*x + y + 3*z s.t.
        x - y   == 4
        y + 2*z == 2

    is converted to:

    min s_1 + s_2 s.t.
        x - y   + s_1      == 4
        y + 2*z +     s_2  == 2
    """
    m = program.n_constraints
    A = np.copy(np.c_[program.A, np.eye(m)])
    b = np.copy(program.b)
    c = np.concatenate((np.zeros(program.n_vars), np.ones(m)))
    pivots = Pivots({program.n_vars + i: i + 1 for i in range(m)})

    aux_program = Tableau(A, b, c, pivots)
    for j in range(program.n_vars, program.n_vars + m):
        zero_out_cj(aux_program, j)
    return aux_program


def create_phase_two_program(original_program: Tableau, aux_program: Tableau) -> Tableau:
    """
    Create the phase two program using the optimized phase one program.

    To do so, we discard the slack variables used by the auxiliary problem and
    restore the original objective function.
    """
    # Discard slack variables
    A = aux_program.A[:, : original_program.n_vars]
    b = np.copy(aux_program.b)
    # Use the orginal objective function
    c = np.copy(original_program.c)
    pivots = aux_program.pivots
    new_prog = Tableau(A, b, c, pivots)

    # The simplex algorithm expects reduced costs to be zeroed out
    for column in pivots.cr:
        zero_out_cj(new_prog, column)

    return new_prog


def remove_redundant_constraints(aux_program: Tableau, n_original_vars: int, tolerance: float) -> None:
    """
    Remove redundant constraints from the auxiliary problem.

    A constraint is redundant if all the coefficients of the original variables are zero.
    For example in this tableau, the last constraint is redundant:

      x1 x2 x3  s1 s2
    | 1  0  0 | 0  0 |
    | 0  1  0 | 0  0 |
    | 0  0  0 | 0  1 |

    """
    zeros = is_zero(aux_program.A[:, :n_original_vars], tolerance)
    sel = np.all(zeros, axis=1)
    sel = np.concatenate(([False], sel))

    # Delete pivots in the same rows as the redundant constraints
    aux_program.M = aux_program.M[~sel]
    for i in sel.nonzero()[0]:
        col = aux_program.pivots.get(row=i)
        aux_program.pivots.delete(column=col)


def move_pivots_from_slack_variables(aux_program: Tableau, n_original_vars: int, tolerance: float) -> None:
    """
    Move pivots from the slack variables to the original variables.

    This happens when the original solution is degenrate (one or more original variables is zero).
    For example, in this tableau, the pivot is in the slack variable s2:

      x1 x2 x3  s1 s2
    | 1  0  0 | 0  0 |
    | 0  1  4 | 0  0 |
    | 0  0  2 | 0  1 |

    After moving the pivot, the tableau becomes:

      x1 x2 x3'  s1 s2'
    | 1  0  0  | 0   0  |
    | 0  1  0  | 0  -2  |
    | 0  0  1  | 0  1/2 |

    """
    # Find the pivots in the slack variables
    pivots = [aux_program.pivots.get_pivot(column=col) for col in aux_program.pivots.cr if (col >= n_original_vars)]

    # Move pivots out of the slack variables
    for p in pivots:
        row, col = p
        # Search through the original variables
        for candidate_col in range(n_original_vars):
            # Already in the base
            if candidate_col in aux_program.pivots.cr:
                continue
            # The new pivot location is zero, not possible
            if is_zero(aux_program.M[row, candidate_col], tolerance):
                continue
            # Set as the new pivot
            aux_program.pivots.delete(column=col)
            aux_program.pivots.set(row=row, column=candidate_col)
            # Update the tableau
            normalize_pivot(aux_program, (row, candidate_col))
            zero_out_cj(aux_program, candidate_col)
            break


def zero_out_cj(program: Tableau, column: int) -> None:
    """Make the reduced cost in the given column zero by subtracting a multiple of the pivot row."""
    row = program.pivots.get(column=column)
    cj = program.c[column]
    if is_zero(cj, program.tolerance):
        return
    program.M[0] -= cj * program.M[row]


def normalize_pivot(program: Tableau, pivot: tuple[int, int]) -> None:
    """Make the pivot equal to 1 by dividing the row by the pivot value."""
    row, column = pivot
    program.M[row] /= program.M[row, column]

    for i in range(1, program.n_constraints + 1):
        if i == row:
            continue
        program.M[i] -= program.M[i, column] * program.M[row]


@suppress_divide_by_zero_warning
def find_pivot(program: Tableau) -> tuple[int, int] | None:
    """Find a new pivot which will improve the objective function."""
    for j in range(program.n_vars):
        if program.pivots.has(column=j):
            continue
        # cj must be negative
        if program.c[j] < -program.tolerance:
            # Pivot candidates must positive
            positive = program.M[1:, j] > program.tolerance
            frac = program.M[1:, -1] / program.M[1:, j]
            frac[~positive] = np.inf
            # Pick a pivot which minimizes b_i/p_ij
            min_i = np.argmin(frac)
            if frac[min_i] != np.inf:
                return min_i + 1, j
    return None


def run_simplex(program: Tableau, *, max_iterations: int | None = math.inf) -> None:
    """Run the simplex algorithm starting from a feasible solution."""
    iterations = 0
    while (pivot := find_pivot(program)) is not None:
        row, column = pivot
        program.pivots.delete(column=program.pivots.get(row=row))
        program.pivots.set(row=row, column=column)
        normalize_pivot(program, pivot)
        zero_out_cj(program, column)

        iterations += 1
        if iterations > max_iterations:
            return

    if np.any(program.c < -program.tolerance):
        msg = "The program is unbounded, try adding more constraints."
        raise Unbounded(msg)


def is_zero(v: float, tol: float) -> bool:
    return np.isclose(v, 0, rtol=0, atol=tol)
