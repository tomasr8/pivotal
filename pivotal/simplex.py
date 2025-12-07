import math
import warnings
from collections.abc import Callable
from enum import Enum, auto
from typing import TypeVar

import numpy as np

from pivotal.errors import AbsoluteValueRequiresMILP, Infeasible, Unbounded
from pivotal.expressions import (
    Abs,
    Constraint,
    Equal,
    ExprOrNumber,
    Expression,
    GreaterOrEqual,
    LessOrEqual,
    Sum,
    Variable,
    get_variable_coeffs,
    get_variable_names,
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
    def __init__(
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


def linearize_abs(
    objective: ExprOrNumber, constraints: list[Constraint], program_type: ProgramType
) -> tuple[ExprOrNumber, list[Constraint], list[str]]:
    """
    Linearize absolute value expressions in the objective function and constraints.

    For each abs(expr) in the objective:
    - Create a new auxiliary variable t
    - Replace abs(expr) with t (or -t if the sign is negative)
    - Add constraints: t >= expr and t >= -expr

    For constraints:
    - abs(expr) <= C is replaced by: expr <= C and -expr <= C
    - abs(expr) = 0 is replaced by: expr = 0
    - abs(expr) >= C requires MILP (raises error)
    - abs(expr) = C (C != 0) requires MILP (raises error)

    This linearization only works when "sign constraints are satisfied":
    - For minimization: coefficients must be positive (e.g., minimize(abs(x)) or minimize(2*abs(x)))
    - For maximization: coefficients must be negative (e.g., maximize(-abs(x)))

    Otherwise, the problem requires Mixed-Integer Linear Programming (MILP).

    Returns the transformed objective, constraints, and list of original variable names.
    """
    abs_expressions = []

    def collect_abs(expr: Expression) -> None:
        """Recursively collect all Abs expressions."""
        match expr:
            case Abs():
                abs_expressions.append(expr)
            case Sum(elts=elts):
                for elt in elts:
                    collect_abs(elt)

    collect_abs(objective)

    # Get original variable names before transformation
    original_vars = get_variable_names((objective, *constraints))

    # Linearize constraints containing absolute values
    constraints = _linearize_abs_in_constraints(constraints)

    if not abs_expressions:
        return objective, constraints, original_vars

    # Validate sign constraints for objective
    for abs_expr in abs_expressions:
        if program_type == ProgramType.MIN and abs_expr.sign == -1:
            msg = (
                "Cannot minimize negative absolute values (e.g., minimize(-abs(x))). "
                "This requires Mixed-Integer Linear Programming (MILP). "
                "Consider using maximize(abs(x)) instead, or negate the entire objective."
            )
            raise AbsoluteValueRequiresMILP(msg)
        if program_type == ProgramType.MAX and abs_expr.sign == 1:
            msg = (
                "Cannot maximize positive absolute values (e.g., maximize(abs(x))). "
                "This requires Mixed-Integer Linear Programming (MILP). "
                "Consider using maximize(-abs(x)) or minimize(abs(x)) instead."
            )
            raise AbsoluteValueRequiresMILP(msg)

    # Create auxiliary variables and constraints for each absolute value in objective
    new_constraints = list(constraints)
    replacements = {}

    for abs_expr in abs_expressions:
        # Create auxiliary variable
        aux_var = Variable()

        # If the sign is negative (e.g., -abs(x)), we use -aux_var
        replacement = -aux_var if abs_expr.sign == -1 else aux_var
        replacements[id(abs_expr)] = replacement

        # Add constraints: aux_var >= arg and aux_var >= -arg
        new_constraints.append(aux_var >= abs_expr.arg)
        new_constraints.append(aux_var >= -abs_expr.arg)

    # Replace Abs expressions in the objective
    def replace_abs(expr: Expression) -> Expression:
        """Replace Abs expressions with auxiliary variables."""
        match expr:
            case Abs():
                return replacements[id(expr)]
            case Sum(elts=elts):
                new_elts = [replace_abs(elt) for elt in elts]
                return Sum(*new_elts)
            case _:
                return expr

    new_objective = replace_abs(objective)
    return new_objective, new_constraints, original_vars


def _contains_abs(expr: Expression) -> bool:
    """Check if an expression contains any Abs terms."""
    match expr:
        case Abs():
            return True
        case Sum(elts=elts):
            return any(_contains_abs(elt) for elt in elts)
        case _:
            return False


def _linearize_abs_in_constraints(constraints: list[Constraint]) -> list[Constraint]:
    """
    Linearize absolute value expressions in constraints.

    Transformations:
    - abs(expr) <= C becomes: expr <= C and -expr <= C
    - abs(expr) = 0 becomes: expr = 0
    - abs(expr) >= C raises AbsoluteValueRequiresMILP
    - abs(expr) = C (C != 0) raises AbsoluteValueRequiresMILP
    """
    new_constraints = []

    for constraint in constraints:
        # Check if constraint contains absolute values
        has_abs_left = _contains_abs(constraint.left)
        has_abs_right = _contains_abs(constraint.right)

        if not has_abs_left and not has_abs_right:
            # No absolute values, keep as is
            new_constraints.append(constraint)
            continue

        if has_abs_left and has_abs_right:
            msg = f"Constraints with absolute values on both sides are not supported. Constraint: {constraint}"
            raise AbsoluteValueRequiresMILP(msg)

        # Normalize so abs is on the left side
        normalized_constraint = _swap_constraint_sides(constraint) if has_abs_right else constraint

        # Now abs is on the left side
        # Get the coefficients from the right side to check if it's a constant
        right_coeffs, right_const = get_variable_coeffs(normalized_constraint.right)

        # Check if left side is a single Abs expression (possibly with coefficient)
        abs_expr = _extract_single_abs(normalized_constraint.left)
        if abs_expr is None:
            msg = f"Only constraints with a single absolute value term are supported. Constraint: {constraint}"
            raise AbsoluteValueRequiresMILP(msg)

        # The RHS should be a constant (all variable coeffs should be zero)
        if any(coeff != 0 for coeff in right_coeffs.values()):
            msg = f"Absolute value constraints must have a constant on the right-hand side. Constraint: {constraint}"
            raise AbsoluteValueRequiresMILP(msg)

        rhs_value = right_const

        # Handle different constraint types
        match normalized_constraint:
            case LessOrEqual():
                # abs(expr) <= C becomes: expr <= C and -expr <= C
                if rhs_value < 0:
                    msg = (
                        f"Constraint |expr| <= {rhs_value} is infeasible (absolute value cannot be negative). "
                        f"Constraint: {constraint}"
                    )
                    raise Infeasible(msg)
                # Create constraints using the right-hand side from the original constraint
                new_constraints.append(LessOrEqual(abs_expr.arg, normalized_constraint.right))
                new_constraints.append(LessOrEqual(-abs_expr.arg, normalized_constraint.right))

            case Equal():
                # abs(expr) = 0 becomes: expr = 0
                # abs(expr) = C (C != 0) requires MILP
                if rhs_value != 0:
                    msg = (
                        f"Constraint |expr| = {rhs_value} requires Mixed-Integer Linear Programming (MILP). "
                        "Only |expr| = 0 is supported, which simplifies to expr = 0."
                    )
                    raise AbsoluteValueRequiresMILP(msg)
                new_constraints.append(Equal(abs_expr.arg, 0))

            case GreaterOrEqual():
                # abs(expr) >= C requires MILP
                msg = (
                    f"Constraint |expr| >= {rhs_value} requires Mixed-Integer Linear Programming (MILP). "
                    "Consider rewriting as |expr| <= C if possible."
                )
                raise AbsoluteValueRequiresMILP(msg)
            case _:
                # This should never happen, but satisfy type checker
                msg = f"Unknown constraint type: {type(normalized_constraint)}"
                raise ValueError(msg)

    return new_constraints


def _swap_constraint_sides(constraint: Constraint) -> Constraint:
    """Swap left and right sides of a constraint, reversing the comparison."""
    match constraint:
        case Equal():
            return Equal(constraint.right, constraint.left)
        case LessOrEqual():
            return GreaterOrEqual(constraint.right, constraint.left)
        case GreaterOrEqual():
            return LessOrEqual(constraint.right, constraint.left)
        case _:
            # This should never happen, but satisfy type checker
            msg = f"Unknown constraint type: {type(constraint)}"
            raise ValueError(msg)


def _extract_single_abs(expr: Expression) -> Abs | None:
    """
    Extract a single Abs expression from an expression.
    Returns None if the expression is not a single Abs (possibly with coefficient).
    """
    match expr:
        case Abs():
            return expr
        case Sum(elts=elts):
            # Check if it's a single Abs with other constant terms
            abs_exprs = [elt for elt in elts if isinstance(elt, Abs)]
            if len(abs_exprs) == 1 and all(isinstance(elt, (int, float, Abs)) for elt in elts):
                return abs_exprs[0]
            return None
        case _:
            return None


def transform_variable_bounds(
    objective: ExprOrNumber, constraints: list[Constraint]
) -> tuple[ExprOrNumber, list[Constraint], dict[str, dict]]:
    """
    Transform variables with non-default bounds.

    Transformations:
    - For free variables (lower=None, upper=None): x = x+ - x- where x+, x- >= 0
    - For lower != 0: x = lower + x' where x' >= 0
    - For upper bounds: add constraint x <= upper (or x' <= upper - lower after substitution)

    Returns:
    - Transformed objective
    - Transformed constraints (with additional bound constraints)
    - Variable mapping for inverse transformation
    """
    # Collect all unique variables and their bounds
    vars_dict = {}

    def collect_variables(expr: Expression | Constraint) -> None:
        """Recursively collect all variables with their bounds."""
        match expr:
            case Variable(name=name, lower=lower, upper=upper):
                if name not in vars_dict:
                    vars_dict[name] = {"lower": lower, "upper": upper}
            case Sum(elts=elts):
                for elt in elts:
                    collect_variables(elt)
            case Abs(arg=arg):
                collect_variables(arg)
            case Constraint(left=left, right=right):
                collect_variables(left)
                collect_variables(right)

    collect_variables(objective)
    for constraint in constraints:
        collect_variables(constraint)

    # Create variable mappings and new constraints
    var_mapping = {}
    new_constraints = list(constraints)
    substitutions = {}

    for var_name, bounds in vars_dict.items():
        lower = bounds["lower"]
        upper = bounds["upper"]

        if lower is None and upper is None:
            # Free variable: x = x_pos - x_neg where both >= 0
            x_pos = Variable(f"{var_name}_pos")
            x_neg = Variable(f"{var_name}_neg")
            substitutions[var_name] = x_pos - x_neg
            var_mapping[var_name] = {"type": "free", "pos": f"{var_name}_pos", "neg": f"{var_name}_neg"}

        elif lower is not None and lower != 0:
            # Shifted variable: x = lower + x'
            x_prime = Variable(f"{var_name}_shifted")
            substitutions[var_name] = lower + x_prime
            var_mapping[var_name] = {"type": "shifted", "shift": lower, "var": f"{var_name}_shifted"}

            # Add upper bound constraint if specified
            if upper is not None:
                # x' <= upper - lower
                new_constraints.append(x_prime <= upper - lower)

        elif upper is not None:
            # Only upper bound (lower = 0): add constraint x <= upper
            x = Variable(var_name)
            new_constraints.append(x <= upper)
            var_mapping[var_name] = {"type": "upper_only", "upper": upper}

    # If no transformations needed, return early
    if not substitutions:
        return objective, new_constraints, var_mapping

    # Apply substitutions to objective and constraints
    def substitute(expr: Expression) -> Expression:
        """Substitute variables according to the mapping."""
        match expr:
            case Variable(name=name, coeff=coeff):
                if name in substitutions:
                    return coeff * substitutions[name] if coeff != 1 else substitutions[name]
                return expr
            case Sum(elts=elts):
                new_elts = [substitute(elt) for elt in elts]
                # Flatten nested sums and combine constants
                flattened = []
                for elt in new_elts:
                    if isinstance(elt, Sum):
                        flattened.extend(elt.elts)
                    else:
                        flattened.append(elt)
                return Sum(*flattened) if len(flattened) > 1 else flattened[0]
            case Abs(arg=arg, sign=sign):
                return Abs(substitute(arg), sign)
            case _:
                return expr

    def substitute_constraint(constraint: Constraint) -> Constraint:
        """Substitute variables in a constraint."""
        new_left = substitute(constraint.left)
        new_right = substitute(constraint.right)
        match constraint:
            case Equal():
                return Equal(new_left, new_right)
            case LessOrEqual():
                return LessOrEqual(new_left, new_right)
            case GreaterOrEqual():
                return GreaterOrEqual(new_left, new_right)
            case _:
                msg = f"Unknown constraint type: {type(constraint)}"
                raise ValueError(msg)

    new_objective = substitute(objective)
    new_constraints = [substitute_constraint(c) for c in new_constraints]

    return new_objective, new_constraints, var_mapping


def apply_variable_mapping(
    solution: np.ndarray, all_vars: list[str], var_mapping: dict[str, dict], original_vars: list[str]
) -> np.ndarray:
    """
    Apply inverse variable mapping to get original variable values.

    For free variables: x = x_pos - x_neg
    For shifted variables: x = lower + x'
    For upper-only variables: x = x (no transformation)
    """
    # Create a dictionary of transformed variable values
    var_values = {name: solution[i] for i, name in enumerate(all_vars)}

    # If no variable transformations, just extract original variables
    if not var_mapping:
        # Just extract values for original variables
        return np.array([var_values.get(name, 0.0) for name in original_vars])

    # Calculate original variable values with transformations applied
    result_values = {}
    for var_name in original_vars:
        if var_name not in var_mapping:
            # No transformation, use value directly
            result_values[var_name] = var_values.get(var_name, 0.0)
        else:
            mapping = var_mapping[var_name]
            match mapping["type"]:
                case "free":
                    # x = x_pos - x_neg
                    x_pos = var_values.get(mapping["pos"], 0.0)
                    x_neg = var_values.get(mapping["neg"], 0.0)
                    result_values[var_name] = x_pos - x_neg
                case "shifted":
                    # x = lower + x'
                    x_prime = var_values.get(mapping["var"], 0.0)
                    result_values[var_name] = mapping["shift"] + x_prime
                case "upper_only":
                    # No transformation needed
                    result_values[var_name] = var_values.get(var_name, 0.0)

    # Convert back to array in the order of original_vars
    return np.array([result_values[name] for name in original_vars])


def evaluate_objective(objective: ExprOrNumber, solution: np.ndarray, var_names: list[str]) -> float:
    """
    Evaluate the objective function value given a solution.

    Args:
        objective: The objective expression
        solution: Array of variable values
        var_names: List of variable names corresponding to solution values

    Returns:
        The objective function value
    """
    var_values = {name: solution[i] for i, name in enumerate(var_names)}

    def evaluate(expr: Expression) -> float:
        """Recursively evaluate an expression."""
        match expr:
            case int() | float():
                return expr
            case Variable(name=name, coeff=coeff):
                return coeff * var_values.get(name, 0.0)
            case Sum(elts=elts):
                return sum(evaluate(elt) for elt in elts)
            case Abs(arg=arg, sign=sign):
                return sign * abs(evaluate(arg))
            case _:
                return 0.0

    return evaluate(objective)


def solve(
    _type: ProgramType,
    objective: ExprOrNumber,
    constraints: list[Constraint] | tuple[Constraint, ...],
    *,
    max_iterations: float,
    tolerance: float,
) -> tuple[float, np.ndarray, list[str], list[str]]:
    """
    The main entrypoint to the simplex algorithm.

    This function accepts a high-level representation of the LP and returns:
    - optimal value
    - solution (values for all variables)
    - all variable names (including auxiliary)
    - original variable names (user-defined only)
    """
    # Store original objective for final value computation
    original_objective = objective

    # Get original variable names before any transformations
    original_vars = get_variable_names([objective, *constraints])

    # Convert constraints to list for transformations
    constraints = list(constraints)

    # Linearize absolute values in the objective
    objective, constraints, _ = linearize_abs(objective, constraints, _type)

    # Transform variable bounds
    objective, constraints, var_mapping = transform_variable_bounds(objective, constraints)

    # Get all variable names after transformation
    all_vars = get_variable_names([objective, *constraints])

    program = as_tableau(_type, objective, constraints, tolerance=tolerance)
    value, solution = _solve(program, max_iterations=max_iterations, tolerance=tolerance)

    # Map transformed variables back to original variables
    original_solution = apply_variable_mapping(solution, all_vars, var_mapping, original_vars)

    # Compute objective value using original objective and mapped solution
    original_value = evaluate_objective(original_objective, original_solution, original_vars)

    # For maximization problems, negate the value (api.py will negate it back)
    if _type == ProgramType.MAX:
        original_value = -original_value

    # Return the mapped solution with original variable names
    return original_value, original_solution, original_vars, original_vars


def _solve(program: Tableau, *, max_iterations: float, tolerance: float) -> tuple[float, np.ndarray]:
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
    _type: ProgramType, A: np.ndarray, b: np.ndarray, c: np.ndarray, constraint_types: np.ndarray
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
            # If no valid pivot candidates (empty or all infinite), skip this column
            if frac.size == 0 or np.all(np.isinf(frac)):
                continue
            # Pick a pivot which minimizes b_i/p_ij
            min_i = np.argmin(frac)
            if frac[min_i] != np.inf:
                return min_i + 1, j
    return None


def run_simplex(program: Tableau, *, max_iterations: float | None = math.inf) -> None:
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
