from nose.tools import assert_equal
from nose import SkipTest

from Goulib.optim import *

class TestHillclimb:
    def test_hillclimb(self):
        # assert_equal(expected, hillclimb(init_function, move_operator, objective_function, max_evaluations))
        raise SkipTest # TODO: implement your test here

class TestHillclimbAndRestart:
    def test_hillclimb_and_restart(self):
        # assert_equal(expected, hillclimb_and_restart(init_function, move_operator, objective_function, max_evaluations))
        raise SkipTest # TODO: implement your test here

class TestP:
    def test_p(self):
        # assert_equal(expected, P(prev_score, next_score, temperature))
        raise SkipTest # TODO: implement your test here

class TestObjectiveFunction:
    def test___call__(self):
        # objective_function = ObjectiveFunction(objective_function)
        # assert_equal(expected, objective_function.__call__(solution))
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # objective_function = ObjectiveFunction(objective_function)
        raise SkipTest # TODO: implement your test here

class TestKirkpatrickCooling:
    def test_kirkpatrick_cooling(self):
        # assert_equal(expected, kirkpatrick_cooling(start_temp, alpha))
        raise SkipTest # TODO: implement your test here

class TestAnneal:
    def test_anneal(self):
        # assert_equal(expected, anneal(init_function, move_operator, objective_function, max_evaluations, start_temp, alpha))
        raise SkipTest # TODO: implement your test here

class TestReversedSections:
    def test_reversed_sections(self):
        # assert_equal(expected, reversed_sections(tour))
        raise SkipTest # TODO: implement your test here

class TestSwappedCities:
    def test_swapped_cities(self):
        # assert_equal(expected, swapped_cities(tour))
        raise SkipTest # TODO: implement your test here

class TestTourLength:
    def test_tour_length(self):
        # assert_equal(expected, tour_length(points, dist, tour))
        raise SkipTest # TODO: implement your test here

class TestTsp:
    def test_tsp(self):
        # assert_equal(expected, tsp(points, dist, max_iterations, start_temp, alpha, close, rand))
        raise SkipTest # TODO: implement your test here
    
if __name__ == "__main__":
    import nose
    nose.runmodule()

