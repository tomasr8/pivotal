import numpy as np
from numpy.testing import assert_allclose

from pivotal._expressions import Variable
from pivotal.api import maximize, minimize


def assert_solution_almost_equal(expected, actual):
    __traceback_hide__ = True  # noqa: F841
    assert np.isclose(expected[0], actual[0])
    assert_allclose(expected[1], [actual[1][name] for name in sorted(actual[1].keys())], atol=1e-8)


def test_mixing_task():
    # opt.pdf page 143, example 11.4
    b = Variable("b", min=0)
    m = Variable("m", min=0)
    z = Variable("z", min=0)

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

    X = [[Variable(f"x_{i}{j}", min=0) for j in range(3)] for i in range(3)]

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
    X = [Variable(f"x{i + 1}", min=0) for i in range(3)]

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
    X = [Variable(f"x{i + 1}", min=0) for i in range(4)]

    solution = minimize(
        X[0] + X[1] + 3 * X[2],
        (
            X[0] + 5 * X[1] + X[2] + X[3] == 7,
            X[0] - X[1] + X[2] == 5,
            0.5 * X[0] - 2 * X[1] + X[2] == 5,
        ),
    )

    assert_solution_almost_equal((15, [0, 0, 5, 2]), solution)
