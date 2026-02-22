import pytest

from pivotal import Infeasible, NodeLimitReached, Variable, maximize, minimize


class TestVariableType:
    def test_default_is_continuous(self):
        x = Variable("x")
        assert x.var_type == "continuous"

    def test_integer_variable(self):
        x = Variable("x", var_type="integer")
        assert x.var_type == "integer"

    def test_binary_variable_bounds(self):
        x = Variable("x", var_type="binary")
        assert x.lower == 0
        assert x.upper == 1
        assert x.var_type == "binary"

    def test_binary_variable_invalid_lower(self):
        with pytest.raises(ValueError, match="lower bound"):
            Variable("x", var_type="binary", lower=1)

    def test_binary_variable_invalid_upper(self):
        with pytest.raises(ValueError, match="upper bound"):
            Variable("x", var_type="binary", upper=5)

    def test_propagation_neg(self):
        x = Variable("x", var_type="integer")
        assert (-x).var_type == "integer"

    def test_propagation_mul(self):
        x = Variable("x", var_type="integer")
        assert (3 * x).var_type == "integer"

    def test_propagation_add_same_name(self):
        x = Variable("x", var_type="integer")
        result = x + Variable("x", var_type="integer")
        assert result.var_type == "integer"


class TestIntegerProgramming:
    def test_simple_integer_min(self):
        x = Variable("x", var_type="integer")
        value, sol = minimize(x, (x >= 3.5,))
        assert value == 4.0
        assert sol["x"] == 4.0

    def test_simple_integer_max(self):
        x = Variable("x", var_type="integer")
        value, sol = maximize(x, (x <= 5.7,))
        assert value == 5.0
        assert sol["x"] == 5.0

    def test_already_integer_relaxation(self):
        x = Variable("x", var_type="integer")
        y = Variable("y", var_type="integer")
        value, sol = minimize(x + y, (x == 3, y == 5))
        assert value == 8.0
        assert sol["x"] == 3.0
        assert sol["y"] == 5.0

    def test_two_integer_variables(self):
        x = Variable("x", var_type="integer")
        y = Variable("y", var_type="integer")
        value, sol = minimize(x + y, (x + y >= 3.5,))
        assert value == 4.0
        assert sol["x"] == sol["x"] // 1  # integer
        assert sol["y"] == sol["y"] // 1  # integer

    def test_infeasible_milp(self):
        x = Variable("x", var_type="integer")
        with pytest.raises(Infeasible):
            minimize(x, (x >= 1.5, x <= 1.9))


class TestMixedInteger:
    def test_mixed_integer_continuous(self):
        x = Variable("x", var_type="integer")
        y = Variable("y")
        value, sol = minimize(x + y, (x + y >= 3.5, y <= 1))
        assert value == pytest.approx(3.5)
        assert sol["x"] == 3.0
        assert sol["y"] == pytest.approx(0.5)

    def test_mixed_integer_max(self):
        x = Variable("x", var_type="integer")
        y = Variable("y")
        value, sol = maximize(x + y, (x + y <= 5.7, x <= 4))
        assert value == pytest.approx(5.7)
        assert sol["x"] == 4.0
        assert sol["y"] == pytest.approx(1.7)


class TestBinaryProgramming:
    def test_simple_binary(self):
        x = Variable("x", var_type="binary")
        y = Variable("y", var_type="binary")
        value, sol = maximize(x + y, (x + y <= 1,))
        assert value == 1.0
        assert sol["x"] in (0.0, 1.0)
        assert sol["y"] in (0.0, 1.0)
        assert sol["x"] + sol["y"] == 1.0

    def test_knapsack(self):
        weights = [2, 3, 4, 5]
        values = [3, 4, 5, 6]
        capacity = 8

        items = [Variable(f"x{i}", var_type="binary") for i in range(4)]

        obj = sum(values[i] * items[i] for i in range(4))
        cap_constraint = sum(weights[i] * items[i] for i in range(4)) <= capacity

        value, sol = maximize(obj, (cap_constraint,))

        # All variables should be 0 or 1
        for i in range(4):
            assert sol[f"x{i}"] in (0.0, 1.0)

        # Capacity constraint should be respected
        total_weight = sum(weights[i] * sol[f"x{i}"] for i in range(4))
        assert total_weight <= capacity

        # Optimal: items 0, 1, 2 (weight=9 > 8) or items 0, 1, 3 (weight=10 > 8)
        # or items 0, 2 (weight=6, value=8) or items 1, 2 (weight=7, value=9)
        # or items 0, 1 (weight=5, value=7) or items 1, 3 (weight=8, value=10)
        # Optimal is items 1, 3: weight=8, value=10
        assert value == pytest.approx(10.0)

    def test_set_cover(self):
        # 3 sets, need to cover all elements with minimum cost
        x = [Variable(f"x{i}", var_type="binary") for i in range(3)]

        # Set 0 covers element 0, cost 3
        # Set 1 covers elements 0,1, cost 2
        # Set 2 covers element 1, cost 4
        value, sol = minimize(
            3 * x[0] + 2 * x[1] + 4 * x[2],
            (
                x[0] + x[1] >= 1,  # element 0 covered
                x[1] + x[2] >= 1,  # element 1 covered
            ),
        )
        # Optimal: just set 1 (covers both), cost 2
        assert value == pytest.approx(2.0)
        assert sol["x1"] == 1.0


class TestBigM:
    def test_either_or_constraints(self):
        # A job must be scheduled on exactly one of two machines.
        # Machine 1 processes in interval [2, 5], machine 2 in [8, 11].
        # We want to minimize the start time.
        #
        # Using big-M to model: either 2 <= x <= 5  OR  8 <= x <= 11
        #   y = 0  =>  2 <= x <= 5   (machine 1)
        #   y = 1  =>  8 <= x <= 11  (machine 2)
        #
        #   x >= 2 + 6*y        (if y=0: x>=2,  if y=1: x>=8)
        #   x <= 5 + 6*y        (if y=0: x<=5,  if y=1: x<=11)
        M = 6
        x = Variable("x")
        y = Variable("y", var_type="binary")

        value, sol = minimize(
            x,
            (
                x >= 2 + M * y,
                x <= 5 + M * y,
            ),
        )
        # Optimal: y=0 (machine 1), x=2
        assert value == pytest.approx(2.0)
        assert sol["x"] == pytest.approx(2.0)
        assert sol["y"] == 0.0

    def test_fixed_cost_production(self):
        # A factory has a fixed setup cost of 10 to open, plus variable cost of 1 per unit.
        # It can produce up to 20 units. We need at least 5 units.
        # Minimize total cost = 10*y + x, where y = 1 if factory is open.
        #
        # Big-M constraints:
        #   x <= M*y   (can only produce if open)
        #   x >= 5     (demand)
        M = 20
        x = Variable("x", upper=M)
        y = Variable("y", var_type="binary")

        value, sol = minimize(
            10 * y + x,
            (
                x <= M * y,  # production only if open
                x >= 5,  # demand
            ),
        )
        # Must open (y=1), produce exactly 5: cost = 10 + 5 = 15
        assert value == pytest.approx(15.0)
        assert sol["y"] == 1.0
        assert sol["x"] == pytest.approx(5.0)


class TestNodeLimit:
    def test_node_limit_sufficient(self):
        x = Variable("x", var_type="integer")
        value, _sol = minimize(x, (x >= 3.5,), node_limit=100)
        assert value == 4.0

    def test_node_limit_reached(self):
        # Create a problem that needs branching, with node_limit=1
        x = Variable("x", var_type="integer")
        y = Variable("y", var_type="integer")
        with pytest.raises((NodeLimitReached, Infeasible)):
            minimize(x + y, (x + y >= 3.5, x <= 10, y <= 10), node_limit=1)

    def test_node_limit_returns_best_found(self):
        # LP relaxation is fractional (3.5), branching finds integer solution (4.0),
        # then node limit triggers with remaining unexplored nodes — returns best found
        x = Variable("x", var_type="integer")
        y = Variable("y", var_type="integer")
        value, _sol = minimize(x + y, (x + y >= 3.5,), node_limit=3)
        assert value == 4.0


class TestContinuousRegression:
    """Ensure pure LP problems still work correctly through the updated API."""

    def test_simple_lp(self):
        x = Variable("x")
        y = Variable("y")
        value, _sol = minimize(x + y, (x + y >= 3.5,))
        assert value == pytest.approx(3.5)

    def test_simple_lp_max(self):
        x = Variable("x")
        value, _sol = maximize(x, (x <= 5.7,))
        assert value == pytest.approx(5.7)
