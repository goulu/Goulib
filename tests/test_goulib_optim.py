#!/usr/bin/env python
# coding: utf8
from nose.tools import assert_equal
from nose import SkipTest
# lines above are inserted automatically by pythoscope. Line below overrides them
from goulib.tests import *

from goulib.optim import *

import logging
import math


class TestNelderMead:
    def test_nelder_mead(self):
        def f(x):
            return math.sin(x[0])*math.cos(x[1])*(1./(abs(x[2])+1))

        logging.info(nelder_mead(f, [0., 0., 0.]))


class TestBinDict:

    def setup(self):
        # this is run once before each test below
        self.bin = BinDict(1)  # simplest Bin
        self.bin[0.1] = 0.1
        self.bin[0.3] = 0.3
        # can contain only strings that have max 10 chars in commmon
        self.alpha = BinDict(10, f=lambda x: set(x)if x else set())
        self.alpha['alpha'] = 'alpha'
        self.alpha['hello'] = 'hello'

    def test___init__(self):
        pass  # tested above

    def test___repr__(self):
        assert repr(self.bin) == 'BinDict(0.4/1)'
        # assert_equal(repr(self.alpha),"BinDict(set(['a', 'e', 'h', 'l', 'o', 'p'])/10)")  # TODO find a test working under Python2 and Python3

    def test_fits(self):
        assert self.bin.fits(0.6)
        assert not self.bin.fits(0.61)
        assert self.alpha.fits('more')
        assert not self.alpha.fits('whisky')

    def test___setitem__(self):
        self.bin[0.6] = 0.6
        assert self.bin.size() == 0
        self.bin[0.6] = 0.2
        assert self.bin.size() == pytest.approx(0.4, abs=1e-7)

        self.alpha['more'] = 'more'
        assert self.alpha.size() == 2
        self.alpha['more'] = 'more!'
        assert self.alpha.size() == 1

    @raises(OverflowError)
    def test___setitem2__(self):
        self.bin[0.61] = 0.61

    @raises(OverflowError)
    def test___setitem3__(self):
        self.alpha['whisky'] = 'whisky'

    def test___delitem__(self):
        # bin = Bin(capacity, items, f)
        # assert_equal(expected, bin.__delitem__(key))
        pass  # TODO: implement

    def test___iadd__(self):
        # bin_dict = BinDict()
        # assert_equal(expected, bin_dict.__iadd__(key, item))
        pass  # TODO: implement

    def test___isub__(self):
        # bin_dict = BinDict()
        # assert_equal(expected, bin_dict.__isub__(key))
        pass  # TODO: implement


class TestBinList:
    def setup(self):
        # this is run once before each test below
        self.bin = BinList(1)  # simplest Bin
        self.bin.extend([0.1, 0.3])
        # can contain only strings that have max 10 chars in commmon
        self.alpha = BinList(10, f=lambda x: set(x)if x else set())
        self.alpha.extend(['alpha', 'hello'])

    def test___init__(self):
        pass  # tested above

    def test___repr__(self):
        assert repr(self.bin) == 'BinList(0.4/1)'
        # assert_equal(repr(self.alpha),"BinList(set(['a', 'e', 'h', 'l', 'o', 'p'])/10)") # TODO find a test working under Python2 and Python3

    def test_fits(self):
        assert self.bin.fits(0.1)
        assert not self.bin.fits(1.1)
        assert self.alpha.fits('alpha')
        assert not self.alpha.fits('abcefghiklmnopqrtuvwxyz')

    def test_extend(self):
        pass  # tested in setup

    def test_append(self):
        self.bin.append(0.6)
        assert self.bin.size() == 0
        self.bin[-1] = 0.2
        assert self.bin.size() == pytest.approx(0.4, abs=1e-7)

        self.alpha.append('more')
        assert self.alpha.size() == 2
        self.alpha[-1] = 'more!'
        assert self.alpha.size() == 1

    def test_insert(self):
        self.bin.insert(0, 0.6)
        assert self.bin.size() == 0
        self.bin[0] = 0.2
        assert self.bin.size() == pytest.approx(0.4, abs=1e-7)

        self.alpha.insert(0, 'more')
        assert self.alpha.size() == 2
        self.alpha[0] = 'more!'
        assert self.alpha.size() == 1

    def test___iadd__(self):
        self.bin += 0.6
        assert self.bin.size() == 0
        self.bin.pop()
        self.bin += 0.2
        assert self.bin.size() == pytest.approx(0.4, abs=1e-7)

        self.alpha += 'more'
        assert self.alpha.size() == 2
        self.alpha[-1] = 'more!'
        assert self.alpha.size() == 1

    def test_pop(self):
        pass  # tested above

    @raises(OverflowError)
    def test___setitem2__(self):
        self.bin.append(0.61)

    @raises(OverflowError)
    def test___setitem3__(self):
        self.alpha.append('whisky')

    def test_remove(self):
        # bin = Bin(capacity, items, f)
        # assert_equal(expected, bin.remove(item))
        pass  # TODO: implement

    def test___isub__(self):
        # bin_list = BinList()
        # assert_equal(expected, bin_list.__isub__(item))
        pass  # TODO: implement

    def test___setitem__(self):
        # bin_list = BinList()
        # assert_equal(expected, bin_list.__setitem__(i, item))
        pass  # TODO: implement


class TestFirstFitDecreasing:
    def test_first_fit_decreasing(self):
        from random import random
        bins = [BinList(1)]
        items = [random() for _ in range(30)]
        nofit = first_fit_decreasing(items, bins, 10)
        for item in nofit:
            for bin in bins:
                assert not bin.fits(item)
        pass


class TestHillclimb:
    def test_hillclimb(self):
        # assert_equal(expected, hillclimb(init_function, move_operator, objective_function, max_evaluations))
        pass  # TODO: implement


class TestHillclimbAndRestart:
    def test_hillclimb_and_restart(self):
        # assert_equal(expected, hillclimb_and_restart(init_function, move_operator, objective_function, max_evaluations))
        pass  # TODO: implement


class TestP:
    def test_p(self):
        # assert_equal(expected, P(prev_score, next_score, temperature))
        pass  # TODO: implement


class TestObjectiveFunction:
    def test___call__(self):
        # objective_function = ObjectiveFunction(objective_function)
        # assert_equal(expected, objective_function.__call__(solution))
        pass  # TODO: implement

    def test___init__(self):
        # objective_function = ObjectiveFunction(objective_function)
        pass  # TODO: implement


class TestKirkpatrickCooling:
    def test_kirkpatrick_cooling(self):
        # assert_equal(expected, kirkpatrick_cooling(start_temp, alpha))
        pass  # TODO: implement


class TestAnneal:
    def test_anneal(self):
        # assert_equal(expected, anneal(init_function, move_operator, objective_function, max_evaluations, start_temp, alpha))
        pass  # TODO: implement


class TestReversedSections:
    def test_reversed_sections(self):
        # assert_equal(expected, reversed_sections(tour))
        pass  # TODO: implement


class TestSwappedCities:
    def test_swapped_cities(self):
        # assert_equal(expected, swapped_cities(tour))
        pass  # TODO: implement


class TestTourLength:
    def test_tour_length(self):
        # assert_equal(expected, tour_length(points, dist, tour))
        pass  # TODO: implement


class TestTsp:
    def test_tsp(self):
        words = ['geneva', 'london', 'new-york', 'paris', 'tokyo', 'rome',
                 'zurich', 'bern', 'berlin', 'mokba', 'washington', 'wien', 'biel']
        n = 2000
        from goulib.math2 import levenshtein
        iterations, score, best = tsp(words, levenshtein, n)
        logging.info('TSP hill climbing closed score=%d, best=%s' %
                     (score, [words[i] for i in best]))
        iterations, score, best = tsp(words, levenshtein, n, 2, .9)
        logging.info('TSP annealing closed score=%d, best=%s' %
                     (score, [words[i] for i in best]))
        iterations, score, best = tsp(words, levenshtein, n, close=False)
        logging.info('TSP hill climbing open score=%d, best=%s' %
                     (score, [words[i] for i in best]))
        iterations, score, best = tsp(
            words, levenshtein, n, 2, .9, close=False)
        logging.info('TSP annealing open score=%d, best=%s' %
                     (score, [words[i] for i in best]))


class TestSize:
    def test_size(self):
        # assert_equal(expected, size(self))
        pass  # TODO: implement


class TestFits:
    def test_fits(self):
        # assert_equal(expected, fits(self, item))
        pass  # TODO: implement


class TestDifferentialEvolution:
    def test___init__(self):
        pass

    def test_function(self):
        from math import cos
        from goulib.math2 import vecmul, norm_2

        class function(object):
            def __init__(self):
                self.x = None
                self.n = 9
                self.domain = [(-100, 100)]*self.n
                self.optimizer = DifferentialEvolution(
                    self, population_size=100, n_cross=5,
                    show_progress=True
                )

            def target(self, vector):
                result = (sum(map(cos, vecmul(vector, 10))) +
                          self.n+1)*norm_2(vector)
                return result

            def print_status(self, mins, means, vector, txt):
                logging.info('%s %s %s %s' % (txt, mins, means, vector))

        v = function()
        logging.info('%s' % v.x)
        assert norm_2(v.x) < 1e-5

    def test_rosenbrock_function(self):
        class rosenbrock(object):
            # http://en.wikipedia.org/wiki/Rosenbrock_function
            def __init__(self, dim=5):
                self.x = None
                self.n = 2*dim
                self.dim = dim
                self.domain = [(1, 3)]*self.n
                self.optimizer = DifferentialEvolution(
                    self, population_size=min(self.n*10, 40),
                    n_cross=self.n,
                    cr=0.9,
                    eps=1e-8,
                    show_progress=True
                )

            def target(self, vector):
                x_vec = vector[0:self.dim]
                y_vec = vector[self.dim:]
                result = 0
                for x, y in zip(x_vec, y_vec):
                    result += 100.0*((y-x*x)**2.0) + (1-x)**2.0
                # print list(x_vec), list(y_vec), result
                return result

            def print_status(self, mins, means, vector, txt):
                logging.info('%s %s %s %s' % (txt, mins, means, vector))

        v = rosenbrock(1)  # single dimension to be faster...
        logging.info('%s' % v.x)
        for x in v.x:
            assert abs(x-1.0) < 1e-2

    def test_evolve(self):
        pass  # tested above

    def test_make_random_population(self):
        pass  # tested above

    def test_optimize(self):
        pass  # tested above

    def test_score_population(self):
        pass  # tested above

    def test_show_population(self):
        pass  # tested above


if __name__ == "__main__":
    runmodule()
