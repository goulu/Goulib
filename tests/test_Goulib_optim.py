from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *

from Goulib.optim import *

import logging

class TestBinDict:

    def setup(self):
        #this is run once before each test below
        self.bin=BinDict(1) #simplest Bin
        self.bin[0.1]=0.1
        self.bin[0.3]=0.3
        self.alpha=BinDict(10,f=lambda x:set(x)if x else set()) # can contain only strings that have max 10 chars in commmon
        self.alpha['alpha']='alpha'
        self.alpha['hello']='hello'
        
    def test___init__(self):
        pass #tested above
    
    def test___repr__(self):
        assert_equal(repr(self.bin),'BinDict(0.4/1)')
        assert_equal(repr(self.alpha),"BinDict(set(['a', 'e', 'h', 'l', 'o', 'p'])/10)")
        
    def test_fits(self):
        assert_true(self.bin.fits(0.6))
        assert_false(self.bin.fits(0.61))
        assert_true(self.alpha.fits('more'))
        assert_false(self.alpha.fits('whisky'))
        
    def test___setitem__(self):
        self.bin[0.6]=0.6
        assert_equal(self.bin.size(),0)
        self.bin[0.6]=0.2
        assert_almost_equal(self.bin.size(),0.4)
        
        self.alpha['more']='more'
        assert_equal(self.alpha.size(),2)
        self.alpha['more']='more!'
        assert_equal(self.alpha.size(),1)
        
    @raises(OverflowError)
    def test___setitem2__(self):
        self.bin[0.61]=0.61
        
    @raises(OverflowError)
    def test___setitem3__(self):
        self.alpha['whisky']='whisky'

    def test___delitem__(self):
        # bin = Bin(capacity, items, f)
        # assert_equal(expected, bin.__delitem__(key))
        raise SkipTest 

    def test___iadd__(self):
        # bin_dict = BinDict()
        # assert_equal(expected, bin_dict.__iadd__(key, item))
        raise SkipTest 

    def test___isub__(self):
        # bin_dict = BinDict()
        # assert_equal(expected, bin_dict.__isub__(key))
        raise SkipTest 

class TestBinList:
    def setup(self):
        #this is run once before each test below
        self.bin=BinList(1) #simplest Bin
        self.bin.extend([0.1,0.3])
        self.alpha=BinList(10,f=lambda x:set(x)if x else set()) # can contain only strings that have max 10 chars in commmon
        self.alpha.extend(['alpha','hello'])
         
    def test___init__(self):
        pass #tested above
    
    def test___repr__(self):
        assert_equal(repr(self.bin),'BinList(0.4/1)')
        assert_equal(repr(self.alpha),"BinList(set(['a', 'e', 'h', 'l', 'o', 'p'])/10)")
        
    def test_fits(self):
        assert_true(self.bin.fits(0.1))
        assert_false(self.bin.fits(1.1))
        assert_true(self.alpha.fits('alpha'))
        assert_false(self.alpha.fits('abcefghiklmnopqrtuvwxyz'))
        
    def test_extend(self):
        pass #tested in setup
        
    def test_append(self):
        self.bin.append(0.6)
        assert_equal(self.bin.size(),0)
        self.bin[-1]=0.2
        assert_almost_equal(self.bin.size(),0.4)
        
        self.alpha.append('more')
        assert_equal(self.alpha.size(),2)
        self.alpha[-1]='more!'
        assert_equal(self.alpha.size(),1)
        
    def test_insert(self):
        self.bin.insert(0,0.6)
        assert_equal(self.bin.size(),0)
        self.bin[0]=0.2
        assert_almost_equal(self.bin.size(),0.4)
        
        self.alpha.insert(0,'more')
        assert_equal(self.alpha.size(),2)
        self.alpha[0]='more!'
        assert_equal(self.alpha.size(),1)
        
    def test___iadd__(self):
        self.bin+=0.6
        assert_equal(self.bin.size(),0)
        self.bin.pop()
        self.bin+=0.2
        assert_almost_equal(self.bin.size(),0.4)
        
        self.alpha+='more'
        assert_equal(self.alpha.size(),2)
        self.alpha[-1]='more!'
        assert_equal(self.alpha.size(),1)
        
    def test_pop(self):
        pass # tested above
    
    @raises(OverflowError)
    def test___setitem2__(self):
        self.bin.append(0.61)
        
    @raises(OverflowError)
    def test___setitem3__(self):
        self.alpha.append('whisky')


    
    def test_remove(self):
        # bin = Bin(capacity, items, f)
        # assert_equal(expected, bin.remove(item))
        raise SkipTest 

    def test___isub__(self):
        # bin_list = BinList()
        # assert_equal(expected, bin_list.__isub__(item))
        raise SkipTest 
    
    def test___setitem__(self):
        # bin_list = BinList()
        # assert_equal(expected, bin_list.__setitem__(i, item))
        raise SkipTest 

class TestFirstFitDecreasing:
    def test_first_fit_decreasing(self):
        from random import random
        bins=[BinList(1)]
        items=[random() for _ in range(30)]
        nofit=first_fit_decreasing(items, bins,10)
        for item in nofit:
            for bin in bins:
                assert_false(bin.fits(item))
        pass

class TestHillclimb:
    def test_hillclimb(self):
        # assert_equal(expected, hillclimb(init_function, move_operator, objective_function, max_evaluations))
        raise SkipTest 

class TestHillclimbAndRestart:
    def test_hillclimb_and_restart(self):
        # assert_equal(expected, hillclimb_and_restart(init_function, move_operator, objective_function, max_evaluations))
        raise SkipTest 

class TestP:
    def test_p(self):
        # assert_equal(expected, P(prev_score, next_score, temperature))
        raise SkipTest 

class TestObjectiveFunction:
    def test___call__(self):
        # objective_function = ObjectiveFunction(objective_function)
        # assert_equal(expected, objective_function.__call__(solution))
        raise SkipTest 

    def test___init__(self):
        # objective_function = ObjectiveFunction(objective_function)
        raise SkipTest 

class TestKirkpatrickCooling:
    def test_kirkpatrick_cooling(self):
        # assert_equal(expected, kirkpatrick_cooling(start_temp, alpha))
        raise SkipTest 

class TestAnneal:
    def test_anneal(self):
        # assert_equal(expected, anneal(init_function, move_operator, objective_function, max_evaluations, start_temp, alpha))
        raise SkipTest 

class TestReversedSections:
    def test_reversed_sections(self):
        # assert_equal(expected, reversed_sections(tour))
        raise SkipTest 

class TestSwappedCities:
    def test_swapped_cities(self):
        # assert_equal(expected, swapped_cities(tour))
        raise SkipTest 

class TestTourLength:
    def test_tour_length(self):
        # assert_equal(expected, tour_length(points, dist, tour))
        raise SkipTest 

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

class TestSize:
    def test_size(self):
        # assert_equal(expected, size(self))
        raise SkipTest 

class TestFits:
    def test_fits(self):
        # assert_equal(expected, fits(self, item))
        raise SkipTest 

if __name__=="__main__":
    runmodule()

