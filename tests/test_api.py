import numpy as np
import pytest
from numpy.testing import assert_allclose

from pivotal.api import maximize, minimize
from pivotal.errors import AbsoluteValueRequiresMILP, Infeasible
from pivotal.expressions import Variable


def assert_solution_almost_equal(expected, actual):
    __traceback_hide__ = True  # noqa: F841
    assert np.isclose(expected[0], actual[0])
    assert_allclose(expected[1], [actual[1][name] for name in sorted(actual[1].keys())], atol=1e-8)


def assert_equal(a, b):
    __traceback_hide__ = True  # noqa: F841
    assert object.__eq__(a, b)


def assert_array_equal(a, b):
    __traceback_hide__ = True  # noqa: F841
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert_equal(x, y)


def test_variables():
    x = Variable()

    assert x + 0 is x
    assert 0 + x is x
    assert x - 0 is x

    assert 1 * x is x
    assert x * 1 is x

    assert (-x).coeff == -1
    assert (2 * x).coeff == 2
    assert (-2 * x).coeff == -2
    assert (x * 2).coeff == 2
    assert (x * -2).coeff == -2

    assert (5 * x).name == x.name


def test_sum():
    x = Variable()
    y = Variable()

    assert_array_equal((x + 1).elts, (x, 1))
    assert_array_equal((x - 3).elts, (x, -3))
    assert_array_equal((1 + x).elts, (1, x))
    assert_array_equal((3 + x).elts, (3, x))

    s = 1 - x
    assert_equal(s.elts[0], 1)
    assert s.elts[1].coeff == -1

    s = -x + 1
    assert s.elts[0].coeff == -1
    assert_equal(s.elts[1], 1)

    s = 1 + x + 2
    assert_equal(s.elts[0], 1)
    assert s.elts[1].coeff == 1
    assert_equal(s.elts[2], 2)

    s = x + y
    assert s.elts[0].coeff == 1
    assert s.elts[1].coeff == 1
    assert s.elts[0].name == x.name
    assert s.elts[1].name == y.name

    s = x - y + 3 * x
    assert s.elts[0].coeff == 1
    assert s.elts[1].coeff == -1
    assert s.elts[2].coeff == 3
    assert s.elts[0].name == x.name
    assert s.elts[1].name == y.name
    assert s.elts[2].name == x.name

    s = 5 * (x + 2 * y)
    assert s.elts[0].coeff == 5
    assert s.elts[1].coeff == 10
    assert s.elts[0].name == x.name
    assert s.elts[1].name == y.name

    s = (x + 2 * y) * 5
    assert s.elts[0].coeff == 5
    assert s.elts[1].coeff == 10
    assert s.elts[0].name == x.name
    assert s.elts[1].name == y.name

    s = x + y
    assert 1 * s is s
    assert s * 1 is s

    x1 = Variable("x")
    x2 = Variable("x")
    assert (x1 + 3 * x2).coeff == 4


def test_abs():
    x = Variable()

    assert abs(x).sign == 1
    assert abs(x).arg.coeff == 1
    assert abs(x).arg.name == x.name

    assert (-abs(x)).sign == -1
    assert (-abs(x)).arg.coeff == 1
    assert (-abs(x)).arg.name == x.name


def test_mixing_task():
    # opt.pdf page 143, example 11.4
    b = Variable("b")
    m = Variable("m")
    z = Variable("z")

    solution = minimize(
        25 * b + 50 * m + 80 * z,
        (
            2 * b + m + z >= 8,
            2 * b + 6 * m + z >= 16,
            b + 3 * m + 6 * z >= 8,
        ),
    )

    assert_solution_almost_equal((160, [3.2, 1.6, 0]), solution)


def test_assignment_problem():
    # Assignment problem
    # technically ILP, but the LP relaxation is always optimal
    # opt.pdf page 149, section 11.4.1
    C = np.array([[1, 2, 3], [2, 3, 1], [10, 2, 1]], dtype=float)

    X = [[Variable(f"x_{i}{j}") for j in range(3)] for i in range(3)]

    solution = minimize(
        sum(sum(C[i, j] * X[i][j] for j in range(3)) for i in range(3)),
        (
            sum(X[0][j] for j in range(3)) == 1,
            sum(X[1][j] for j in range(3)) == 1,
            sum(X[2][j] for j in range(3)) == 1,
            sum(X[i][0] for i in range(3)) == 1,
            sum(X[i][1] for i in range(3)) == 1,
            sum(X[i][2] for i in range(3)) == 1,
        ),
    )

    assert_solution_almost_equal((4, [1, 0, 0, 0, 0, 1, 0, 1, 0]), solution)


def test_redundant_constraints():
    # OCW, page 82
    X = [Variable(f"x{i + 1}") for i in range(3)]

    solution = maximize(
        X[0] + 2 * X[1] - X[2],
        (
            2 * X[0] - X[1] + X[2] == 12,
            -X[0] + 2 * X[1] + X[2] == 10,
            X[0] + X[1] + 2 * X[2] == 22,
        ),
    )

    assert_solution_almost_equal((98 / 3, [34 / 3, 32 / 3, 0]), solution)


def test_degenerate_solution():
    # OCW, page 84
    X = [Variable(f"x{i + 1}") for i in range(4)]

    solution = minimize(
        X[0] + X[1] + 3 * X[2],
        (
            X[0] + 5 * X[1] + X[2] + X[3] == 7,
            X[0] - X[1] + X[2] == 5,
            0.5 * X[0] - 2 * X[1] + X[2] == 5,
        ),
    )

    assert_solution_almost_equal((15, [0, 0, 5, 2]), solution)


def test_abs_simple():
    # Minimize abs(x) with x == 5
    # Expected: x = 5, objective = 5
    x = Variable("x")

    solution = minimize(abs(x), (x == 5,))

    assert np.isclose(solution[0], 5.0)
    assert np.isclose(solution[1]["x"], 5.0)


def test_abs_negative():
    # Minimize abs(x) with x == -3
    # Since all variables are non-negative, we need to use x - 3 == 0
    # So we're minimizing abs(x - 3) with x == 0, giving |0 - 3| = 3
    x = Variable("x")

    solution = minimize(abs(x - 3), (x == 0,))

    assert np.isclose(solution[0], 3.0)
    assert np.isclose(solution[1]["x"], 0.0)


def test_abs_multiple():
    # Minimize abs(x) + abs(y) with x == 2, y == 3
    # Expected: objective = 5
    x = Variable("x")
    y = Variable("y")

    solution = minimize(abs(x) + abs(y), (x == 2, y == 3))

    assert np.isclose(solution[0], 5.0)
    assert np.isclose(solution[1]["x"], 2.0)
    assert np.isclose(solution[1]["y"], 3.0)


def test_abs_with_coefficient():
    # Minimize 2*abs(x) + y with x == 3, y == 1
    # Expected: objective = 2*3 + 1 = 7
    x = Variable("x")
    y = Variable("y")

    solution = minimize(2 * abs(x) + y, (x == 3, y == 1))

    assert np.isclose(solution[0], 7.0)
    assert np.isclose(solution[1]["x"], 3.0)
    assert np.isclose(solution[1]["y"], 1.0)


def test_abs_unconstrained():
    # Minimize abs(x - 5) where x >= 0
    # Since all variables are non-negative, the minimum is at x = 5
    # giving abs(5 - 5) = 0
    x = Variable("x")

    solution = minimize(abs(x - 5), ())

    assert np.isclose(solution[0], 0.0, atol=1e-5)
    assert np.isclose(solution[1]["x"], 5.0, atol=1e-5)


def test_abs_with_linear_term():
    # Minimize abs(x - 2) + y with x + y == 10
    # The optimal solution minimizes the absolute value
    x = Variable("x")
    y = Variable("y")

    solution = minimize(abs(x - 2) + y, (x + y == 10,))

    # At optimum, x should be as close to 2 as possible
    # Given x + y = 10 and x >= 0, y >= 0, the optimal is x = 2, y = 8
    # giving abs(2 - 2) + 8 = 8
    assert np.isclose(solution[0], 8.0, atol=1e-5)
    assert np.isclose(solution[1]["x"], 2.0, atol=1e-5)
    assert np.isclose(solution[1]["y"], 8.0, atol=1e-5)


def test_abs_maximize():
    # Maximize -abs(x - 5) with 0 <= x <= 10
    # This is equivalent to minimize abs(x - 5)
    # Expected: x = 5, objective = 0
    x = Variable("x")

    solution = maximize(-abs(x - 5), (x <= 10,))

    assert np.isclose(solution[0], 0.0, atol=1e-5)
    assert np.isclose(solution[1]["x"], 5.0, atol=1e-5)


def test_abs_complex_expression():
    # Minimize abs(2*x + 3*y - 10) with constraints
    x = Variable("x")
    y = Variable("y")

    solution = minimize(abs(2 * x + 3 * y - 10), (x == 2, y == 2))

    # At x=2, y=2: abs(2*2 + 3*2 - 10) = abs(4 + 6 - 10) = abs(0) = 0
    assert np.isclose(solution[0], 0.0, atol=1e-5)
    assert np.isclose(solution[1]["x"], 2.0)
    assert np.isclose(solution[1]["y"], 2.0)


def test_abs_maximize_positive_raises_error():
    # maximize(abs(x)) should raise AbsoluteValueRequiresMILP
    x = Variable("x")

    with pytest.raises(AbsoluteValueRequiresMILP):
        maximize(abs(x), (x <= 10,))


def test_abs_minimize_negative_raises_error():
    # minimize(-abs(x)) should raise AbsoluteValueRequiresMILP
    x = Variable("x")

    with pytest.raises(AbsoluteValueRequiresMILP):
        minimize(-abs(x), (x <= 10,))


############ Tests for abs() in constraints


def test_abs_constraint_less_equal():
    # Test abs(x - 5) <= 3
    # This means -3 <= x - 5 <= 3, so 2 <= x <= 8
    # Minimize x subject to abs(x - 5) <= 3
    # Expected: x = 2 (since x >= 0 and we minimize)
    x = Variable("x")

    solution = minimize(x, (abs(x - 5) <= 3,))

    assert np.isclose(solution[0], 2.0, atol=1e-5)
    assert np.isclose(solution[1]["x"], 2.0, atol=1e-5)


def test_abs_constraint_less_equal_maximize():
    # Maximize x subject to abs(x - 5) <= 3
    # This means 2 <= x <= 8
    # Expected: x = 8
    x = Variable("x")

    solution = maximize(x, (abs(x - 5) <= 3,))

    assert np.isclose(solution[0], 8.0, atol=1e-5)
    assert np.isclose(solution[1]["x"], 8.0, atol=1e-5)


def test_abs_constraint_less_equal_two_variables():
    # Minimize x + y subject to abs(x - y) <= 2
    # This means -2 <= x - y <= 2
    x = Variable("x")
    y = Variable("y")

    solution = minimize(x + y, (abs(x - y) <= 2, x + y >= 4))

    # Optimal: x and y should be as small as possible while satisfying constraints
    # With x + y = 4 and |x - y| <= 2, we get x = 1, y = 3 or x = 3, y = 1
    # But we want to minimize x + y, so x + y = 4
    assert np.isclose(solution[0], 4.0, atol=1e-5)


def test_abs_constraint_equal_zero():
    # Test abs(x - 3) = 0
    # This simplifies to x - 3 = 0, so x = 3
    x = Variable("x")

    solution = minimize(x, (abs(x - 3) == 0,))

    assert np.isclose(solution[0], 3.0, atol=1e-5)
    assert np.isclose(solution[1]["x"], 3.0, atol=1e-5)


def test_abs_constraint_equal_zero_complex():
    # Test abs(2*x + 3*y - 12) = 0
    # This means 2*x + 3*y = 12
    x = Variable("x")
    y = Variable("y")

    solution = minimize(x + y, (abs(2 * x + 3 * y - 12) == 0,))

    # With 2*x + 3*y = 12 and minimize x + y
    # At x = 0: 3*y = 12, y = 4, x + y = 4
    # At y = 0: 2*x = 12, x = 6, x + y = 6
    # Optimal is x = 0, y = 4
    assert np.isclose(solution[0], 4.0, atol=1e-5)
    assert np.isclose(solution[1]["x"], 0.0, atol=1e-5)
    assert np.isclose(solution[1]["y"], 4.0, atol=1e-5)


def test_abs_constraint_equal_nonzero_raises_error():
    # abs(x) = 5 should raise AbsoluteValueRequiresMILP
    x = Variable("x")

    with pytest.raises(AbsoluteValueRequiresMILP):
        minimize(x, (abs(x) == 5,))


def test_abs_constraint_greater_equal_raises_error():
    # abs(x) >= 3 should raise AbsoluteValueRequiresMILP
    x = Variable("x")

    with pytest.raises(AbsoluteValueRequiresMILP):
        minimize(x, (abs(x) >= 3,))


def test_abs_constraint_negative_rhs_raises_error():
    # abs(x) <= -1 should raise Infeasible (absolute value can't be negative)
    x = Variable("x")

    with pytest.raises(Infeasible):
        minimize(x, (abs(x) <= -1,))


def test_abs_constraint_with_variable_rhs_raises_error():
    # abs(x) <= y should raise error (RHS must be constant)
    x = Variable("x")
    y = Variable("y")

    with pytest.raises(AbsoluteValueRequiresMILP):
        minimize(x, (abs(x) <= y,))


def test_abs_constraint_both_sides_raises_error():
    # abs(x) <= abs(y) should raise error
    x = Variable("x")
    y = Variable("y")

    with pytest.raises(AbsoluteValueRequiresMILP):
        minimize(x + y, (abs(x) <= abs(y),))


def test_abs_constraint_multiple_abs_raises_error():
    # abs(x) + abs(y) <= 5 should raise error (only single abs supported)
    x = Variable("x")
    y = Variable("y")

    with pytest.raises(AbsoluteValueRequiresMILP):
        minimize(x + y, (abs(x) + abs(y) <= 5,))


def test_abs_constraint_right_side():
    # Test 3 <= abs(x - 5) (abs on right side, should be normalized)
    # This is abs(x - 5) >= 3, which should raise error
    x = Variable("x")

    with pytest.raises(AbsoluteValueRequiresMILP):
        minimize(x, (3 <= abs(x - 5),))


def test_abs_constraint_combined_with_objective():
    # Test using abs in both objective and constraints
    # Minimize abs(x) subject to abs(x - 10) <= 2
    # This means 8 <= x <= 12
    # Minimizing abs(x) with x in [8, 12] gives x = 8, abs(x) = 8
    x = Variable("x")

    solution = minimize(abs(x), (abs(x - 10) <= 2,))

    assert np.isclose(solution[0], 8.0, atol=1e-5)
    assert np.isclose(solution[1]["x"], 8.0, atol=1e-5)


def test_abs_constraint_simple_variable():
    # Test abs(x) <= 5
    # This means -5 <= x <= 5, but since x >= 0, it's 0 <= x <= 5
    # Maximize x subject to abs(x) <= 5
    # Expected: x = 5
    x = Variable("x")

    solution = maximize(x, (abs(x) <= 5,))

    assert np.isclose(solution[0], 5.0, atol=1e-5)
    assert np.isclose(solution[1]["x"], 5.0, atol=1e-5)
