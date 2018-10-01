#!/usr/bin/env python
# coding: utf8
from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them

from Goulib.tests import *
from Goulib.container import *

from Goulib import math2

class TestRecord:
    @classmethod
    def setup_class(self):
        self.d={'first':'Albert', 'last':'Einstein'}
        self.r=Record(self.d)
        assert_raises(AttributeError,lambda:self.r.birth)
        
    def test___init__(self):
        pass
    
    def test___getattr__(self):
        assert_equal(self.r.first,self.d['first'])
        assert_equal(self.r['first'],self.d['first'])

    def test___setattr__(self):
        self.r.born=1879
        assert_equal(self.r['born'],1879)
        
        r2=self.r.copy()
        r2.first='Franck' # Franck Enstein ;-)
        #check the fields are copied
        assert_equal(self.r.first,self.d['first'])
        

    def test___str__(self):
        # record = Record(*args, **kwargs)
        # assert_equal(expected, record.__str__())
        raise SkipTest # implement your test here




class TestSequence:
    @classmethod
    def setup_class(self):
        self.A000040=Sequence(containf=math2.is_prime)
        
        #check that we can iterate twice in the same Sequence
        l1=list(itertools2.take(20,self.A000040))
        l2=list(itertools2.take(20,self.A000040))
        assert_equal(l1,l2)
        
    def test___add__(self):
        # sequence = Sequence(iterf, itemf, containf, desc)
        # assert_equal(expected, sequence.__add__(other))
        raise SkipTest # implement your test here

    def test___and__(self):
        # sequence = Sequence(iterf, itemf, containf, desc)
        # assert_equal(expected, sequence.__and__(other))
        raise SkipTest # implement your test here

    def test___contains__(self):
        # sequence = Sequence(iterf, itemf, containf, desc)
        # assert_equal(expected, sequence.__contains__(n))
        raise SkipTest # implement your test here

    def test___getitem__(self):
        # sequence = Sequence(iterf, itemf, containf, desc)
        # assert_equal(expected, sequence.__getitem__(i))
        raise SkipTest # implement your test here

    def test___init__(self):
        # sequence = Sequence(iterf, itemf, containf, desc)
        raise SkipTest # implement your test here

    def test___iter__(self):
        # sequence = Sequence(iterf, itemf, containf, desc)
        # assert_equal(expected, sequence.__iter__())
        raise SkipTest # implement your test here

    def test___mod__(self):
        # sequence = Sequence(iterf, itemf, containf, desc)
        # assert_equal(expected, sequence.__mod__(other))
        raise SkipTest # implement your test here

    def test___repr__(self):
        # sequence = Sequence(iterf, itemf, containf, desc)
        # assert_equal(expected, sequence.__repr__())
        raise SkipTest # implement your test here

    def test___sub__(self):
        # sequence = Sequence(iterf, itemf, containf, desc)
        # assert_equal(expected, sequence.__sub__(other))
        raise SkipTest # implement your test here

    def test_accumulate(self):
        A007504=self.A000040.accumulate()

    def test_apply(self):
        A001248=self.A000040.apply(lambda n:n*n)

    def test_filter(self):
        A000043=self.A000040.filter(math2.lucas_lehmer)

    def test_index(self):
        # sequence = Sequence(iterf, itemf, containf, desc)
        # assert_equal(expected, sequence.index(v))
        raise SkipTest # implement your test here

    def test_pairwise(self):
        A001223=self.A000040.pairwise(operator.sub)

    def test_sort(self):
        # sequence = Sequence(iterf, itemf, containf, desc)
        # assert_equal(expected, sequence.sort(key, buffer))
        raise SkipTest # implement your test here

    def test_unique(self):
        # sequence = Sequence(iterf, itemf, containf, desc)
        # assert_equal(expected, sequence.unique(buffer))
        raise SkipTest # implement your test here

    def test___or__(self):
        # sequence = Sequence(iterf, itemf, containf, desc)
        # assert_equal(expected, sequence.__or__(other))
        raise SkipTest # implement your test here

if __name__ == "__main__":
    runmodule()

