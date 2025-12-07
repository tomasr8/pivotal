import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pivotal.simplex import (
    Pivots,
    ProgramType,
    Tableau,
    _solve,
    canonicalize,
    find_pivot,
    normalize_pivot,
    zero_out_cj,
)


def test_zero_out_cj_1():
    A = np.eye(2)
    b = np.ones(2)
    c = np.zeros(2)
    program = Tableau(A, b, c, Pivots({0: 1, 1: 2}))
    M = np.copy(program.M)
    zero_out_cj(program, 0)
    assert_array_equal(M, program.M)
    zero_out_cj(program, 1)
    assert_array_equal(M, program.M)


def test_zero_out_cj_2():
    A = np.eye(2)
    b = np.ones(2)
    c = 2 * np.ones(2)
    program = Tableau(A, b, c, Pivots({0: 1, 1: 2}))
    zero_out_cj(program, 0)
    assert_array_equal(
        program.M,
        np.array(
            [
                [0, 2, -2],
                [1, 0, 1],
                [0, 1, 1],
            ]
        ),
    )
    zero_out_cj(program, 1)
    assert_array_equal(
        program.M,
        np.array(
            [
                [0, 0, -4],
                [1, 0, 1],
                [0, 1, 1],
            ]
        ),
    )


def test_normalize_pivot_1():
    A = np.array([[1, 0], [0, 1]], dtype=float)
    b = [3, 5]
    c = [1, 1]
    program = Tableau(A, b, c, Pivots({}))
    normalize_pivot(program, (1, 0))

    assert_array_equal(
        program.M,
        np.array(
            [
                [1, 1, 0],
                [1, 0, 3],
                [0, 1, 5],
            ]
        ),
    )


def test_normalize_pivot_2():
    A = np.array([[12, 3], [0, 2]], dtype=float)
    b = [3, 5]
    c = [1, 1]
    program = Tableau(A, b, c, Pivots({}))
    normalize_pivot(program, (1, 1))

    assert_array_equal(
        program.M,
        np.array(
            [
                [1, 1, 0],
                [4, 1, 1],
                [-8, 0, 3],
            ]
        ),
    )


def test_find_pivot_no_pivot_cj_positive():
    A = np.array([[1, 0, 2], [0, 1, 7]], dtype=float)
    b = [3, 5]
    c = [1, 1, 1]
    program = Tableau(A, b, c, Pivots({}))
    pivot = find_pivot(program)
    assert pivot is None


def test_find_pivot_no_pivot_negative_column():
    A = np.array([[1, 0, -2], [0, 1, -7]], dtype=float)
    b = [3, 5]
    c = [1, 1, -3]
    program = Tableau(A, b, c, Pivots({}))
    pivot = find_pivot(program)
    assert pivot is None


def test_find_pivot():
    A = np.array([[1, 0, 2], [0, 1, 7]], dtype=float)
    b = [3, 5]
    c = [1, 1, -2]
    program = Tableau(A, b, c, Pivots({0: 1, 1: 2}))
    pivot = find_pivot(program)
    assert_array_equal(pivot, [2, 2])


def test_get_solution_1():
    A = np.array([[1, 0], [0, 1]], dtype=float)
    b = [3, 5]
    c = [1, 1]
    program = Tableau(A, b, c, Pivots({0: 1, 1: 2}))
    solution = program.solution
    assert_array_equal(solution, [3, 5])

    program = Tableau(A, b, c, Pivots({1: 2, 0: 1}))
    solution = program.solution
    assert_array_equal(solution, [3, 5])


def test_canonicalize_1():
    A = np.eye(2)
    b = np.array([1, 2])
    c = np.array([3, 4])
    types = np.array([0, 0])

    Ac, bc, cc = canonicalize(ProgramType.MIN, A, b, c, types)
    assert_array_equal(A, Ac)
    assert_array_equal(b, bc)
    assert_array_equal(c, cc)


def test_canonicalize_2():
    A = np.array([[1, 1]])
    b = np.array([-2])
    c = np.array([3, 4])
    types = np.array([0])

    Ac, bc, cc = canonicalize(ProgramType.MIN, A, b, c, types)
    assert_array_equal(Ac, [[-1, -1]])
    assert_array_equal(bc, [2])
    assert_array_equal(cc, [3, 4])


def test_canonicalize_3():
    A = np.array([[1, 1]])
    b = np.array([2])
    c = np.array([3, 4])
    types = np.array([1])

    Ac, bc, cc = canonicalize(ProgramType.MIN, A, b, c, types)
    assert_array_equal(Ac, [[1, 1, -1]])
    assert_array_equal(bc, [2])
    assert_array_equal(cc, [3, 4, 0])


def test_canonicalize_4():
    A = np.array([[1, 1]])
    b = np.array([2])
    c = np.array([3, 4])
    types = np.array([-1])

    Ac, bc, cc = canonicalize(ProgramType.MIN, A, b, c, types)
    assert_array_equal(Ac, [[1, 1, 1]])
    assert_array_equal(bc, [2])
    assert_array_equal(cc, [3, 4, 0])


def test_program_1():
    A = np.array(
        [
            [1, 3],
            [4, 1],
        ],
        dtype=float,
    )

    b = np.array([6, 5], dtype=float)
    c = np.array([1, 2], dtype=float)

    value, X = _solve(Tableau(A, b, c, Pivots({})), max_iterations=math.inf, tolerance=1e-6)
    X_true = [0.81818182, 1.72727273]
    assert value == pytest.approx(4.27272727)
    for x, xt in zip(X, X_true, strict=True):
        assert x == pytest.approx(xt)


def test_program_2():
    A = np.array(
        [
            [3, 2, 1],
            [1, 2, 2],
        ],
        dtype=float,
    )

    b = np.array([10, 15], dtype=float)
    c = np.array([-20, -30, -40], dtype=float)

    value, X = _solve(Tableau(A, b, c, Pivots({})), max_iterations=math.inf, tolerance=1e-6)
    X_true = [1, 0, 7]
    assert value == pytest.approx(-300)
    for x, xt in zip(X, X_true, strict=True):
        assert x == pytest.approx(xt)
