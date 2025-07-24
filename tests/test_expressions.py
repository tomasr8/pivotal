from pivotal._expressions import Variable, substitute


def assert_equal(a, b):
    __traceback_hide__ = True  # noqa: F841
    assert object.__eq__(a, b)


def assert_array_equal(a, b):
    __traceback_hide__ = True  # noqa: F841
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert_equal(x, y)


def test_variables():
    x = Variable("x")

    assert x + 0 is x
    assert 0 + x is x
    assert x - 0 is x


def test_multiplication():
    x = Variable("x")

    assert (x * 0) == 0
    assert (0 * x) == 0

    assert 1 * x is x
    assert x * 1 is x

    assert (-x).coeff == -1
    assert (2 * x).coeff == 2
    assert (-2 * x).coeff == -2
    assert (x * 2).coeff == 2
    assert (x * -2).coeff == -2

    assert (2 * (3 * x)).coeff == 6
    assert ((3 * x) * 2).coeff == 6


def test_sum():
    x = Variable("x")
    y = Variable("y")

    assert_array_equal((x + 1).elts, (x, 1))
    assert_array_equal((x - 3).elts, (x, -3))
    assert_array_equal((1 + x).elts, (1, x))
    assert_array_equal((3 + x).elts, (3, x))

    s = 1 - x
    assert s.elts[0].coeff == -1
    assert_equal(s.elts[1], 1)

    s = -x + 1
    assert s.elts[0].coeff == -1
    assert_equal(s.elts[1], 1)

    s = 1 + x + 2
    assert s.elts[0].name == "x"
    assert_equal(s.elts[1], 3)

    s = x + y
    assert s.elts[0].name == "x"
    assert s.elts[1].name == "y"

    s = x - y + 3 * x
    assert s.elts[0].coeff == 4
    assert s.elts[1].coeff == -1

    s = 5 * (x + 2 * y)
    assert s.elts[0].coeff == 5
    assert s.elts[1].coeff == 10
    assert s.elts[0].expr.name == x.name
    assert s.elts[1].expr.name == y.name

    s = (x + 2 * y) * 5
    assert s.elts[0].coeff == 5
    assert s.elts[1].coeff == 10
    assert s.elts[0].expr.name == x.name
    assert s.elts[1].expr.name == y.name

    s = x + y
    assert 1 * s is s
    assert s * 1 is s

    x1 = Variable("x")
    x2 = Variable("x")
    assert (x1 + 3 * x2).coeff == 4


def test_abs():
    x = Variable("x")

    assert abs(x).var.name == x.name

    assert (-abs(x)).coeff == -1
    assert (-abs(x)).expr.var.name == x.name


def test_simplify():
    x = Variable("x")
    y = Variable("y")

    assert repr(x + y) == "x + y"
    assert repr(x - y) == "x - y"
    assert repr(x + x + x) == "3*x"
    assert repr(x + 2 * x) == "3*x"
    assert repr(x + 2 * y) == "x + 2*y"
    assert repr(2 * x + 3 * y) == "2*x + 3*y"
    assert repr(2 * x + 3 * y + x + 2 * y) == "3*x + 5*y"


def test_substitute_1():
    x = Variable("x")
    y = Variable("y")

    sub = substitute(x, x, y)
    assert sub is y


def test_substitute_2():
    x = Variable("x")
    y = Variable("y")
    z = Variable("z")

    expr = x + y
    new_expr = substitute(expr, y, z)

    assert repr(new_expr) == "x + z"


def test_substitute_3():
    x = Variable("x")
    y = Variable("y")
    z = Variable("z")

    expr = x + y + z
    new_expr = substitute(expr, y, z)

    assert repr(new_expr) == "x + 2*z"


def test_substitute_4():
    x = Variable("x")
    y = Variable("y")
    z = Variable("z")

    expr = x + y <= z
    new_expr = substitute(expr, y, z)

    assert repr(new_expr) == "x + z <= z"
