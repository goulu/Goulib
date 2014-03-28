from nose.tools import assert_equal, assert_true, assert_false
from nose import SkipTest

import logging

from Goulib.optim import *

class TestBin:
    @classmethod
    def setup_class(self):
        self.bin=Bin(1) #simplest Bin
        self.alpha=Bin(10,f=lambda x:set(x)if x else set()) # can contain only strings that have max 10 chars in commmon
        
    def test___init__(self):
        pass #tested above
    
    def test___repr__(self):
        print(self.bin)
        print(self.alpha)
        
    def test_fits(self):
        assert_true(self.bin.fits(0.1))
        assert_false(self.bin.fits(1.1))
        assert_true(self.alpha.fits('alpha'))
        assert_false(self.alpha.fits('abcefghiklmnopqrtuvwxyz'))
        
    def test___delitem__(self):
        # bin = Bin(capacity, items, f)
        # assert_equal(expected, bin.__delitem__(key))
        raise SkipTest # TODO: implement your test here




    def test___setitem__(self):
        # bin = Bin(capacity, items, f)
        # assert_equal(expected, bin.__setitem__(key, item))
        raise SkipTest # TODO: implement your test here

    def test_append(self):
        # bin = Bin(capacity, items, f)
        # assert_equal(expected, bin.append(item))
        raise SkipTest # TODO: implement your test here

    def test_extend(self):
        # bin = Bin(capacity, items, f)
        # assert_equal(expected, bin.extend(more))
        raise SkipTest # TODO: implement your test here

    def test_insert(self):
        # bin = Bin(capacity, items, f)
        # assert_equal(expected, bin.insert(i, item))
        raise SkipTest # TODO: implement your test here

    def test_pop(self):
        # bin = Bin(capacity, items, f)
        # assert_equal(expected, bin.pop(i))
        raise SkipTest # TODO: implement your test here

    def test_remove(self):
        # bin = Bin(capacity, items, f)
        # assert_equal(expected, bin.remove(item))
        raise SkipTest # TODO: implement your test here

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
        words=['geneva','london','new-york','paris','tokyo','rome','zurich','bern','berlin','mokba','washington','wien','biel']
        n=2000
        from Goulib.math2 import levenshtein 
        iterations,score,best=tsp(words,levenshtein,n)
        logging.info('TSP hill climbing closed score=%d, best=%s'%(score,[words[i] for i in best]))
        iterations,score,best=tsp(words,levenshtein,n,2,.9)
        logging.info('TSP annealing closed score=%d, best=%s'%(score,[words[i] for i in best]))
        iterations,score,best=tsp(words,levenshtein,n,close=False)
        logging.info('TSP hill climbing open score=%d, best=%s'%(score,[words[i] for i in best]))
        iterations,score,best=tsp(words,levenshtein,n,2,.9,close=False)
        logging.info('TSP annealing open score=%d, best=%s'%(score,[words[i] for i in best]))
    


if __name__=="__main__":
    import sys
    import nose
    from cStringIO import StringIO  
    
    module_name = sys.modules[__name__].__file__

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    result = nose.run(argv=[sys.argv[0], module_name, '-s'])
    sys.stdout = old_stdout
    print mystdout.getvalue()

