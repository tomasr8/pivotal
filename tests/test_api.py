import numpy as np
from numpy.testing import assert_allclose

from pivotal.api import maximize, minimize
from pivotal.expressions import Expression, LessOrEqual, Variable, substitute


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


def test_substitute_1():
    x = Variable("x")
    y = Variable("y")

    sub = substitute(x, x, y)
    assert sub is y
    assert repr(sub) == "y"


def test_substitute_2():
    x = Variable("x")
    y = Variable("y")
    z = Variable("z")

    expr = x + y
    new_expr = substitute(expr, y, z)

    assert isinstance(new_expr, Expression)
    assert repr(new_expr) == "x + z"


def test_substitute_3():
    x = Variable("x")
    y = Variable("y")
    z = Variable("z")

    expr = x + y + z
    new_expr = substitute(expr, y, z)

    assert isinstance(new_expr, Expression)
    assert repr(new_expr) == "x + z + z"


def test_substitute_4():
    x = Variable("x")
    y = Variable("y")
    z = Variable("z")

    expr = x + y <= z
    new_expr = substitute(expr, y, z)

    assert isinstance(new_expr, LessOrEqual)
    assert repr(new_expr) == "x + z <= z"


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
    X = [Variable(f"x{i+1}") for i in range(3)]

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
    X = [Variable(f"x{i+1}") for i in range(4)]

    solution = minimize(
        X[0] + X[1] + 3 * X[2],
        (
            X[0] + 5 * X[1] + X[2] + X[3] == 7,
            X[0] - X[1] + X[2] == 5,
            0.5 * X[0] - 2 * X[1] + X[2] == 5,
        ),
    )

    assert_solution_almost_equal((15, [0, 0, 5, 2]), solution)
