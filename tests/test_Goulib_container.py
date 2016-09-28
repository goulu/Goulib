#!/usr/bin/env python
# coding: utf8
from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *

from Goulib.container import *

def ve2no(f, *args):
    'Convert ValueError result to -1'
    try:
        return f(*args)
    except ValueError:
        return -1

class TestSortedCollection:
    
    @classmethod
    def setup_class(self):
        from random import choice
        self.pool = [1.5, 2, 2.0, 3, 3.0, 3.5, 4, 4.0, 4.5]
        self.testSC=[]
        for i in range(500):
            for n in range(6):
                s = [choice(self.pool) for i in range(n)]
                sc = SortedCollection(s)
                s.sort()
                self.testSC.append((sc,s))
                
    def test_index(self):
        def slow_index(seq, k):
            'Location of match or -1 if not found'
            for i, item in enumerate(seq):
                if item == k:
                    return i
            return -1
        for sc,s in self.testSC:
            for probe in self.pool:
                assert_equal(ve2no(sc.index, probe),slow_index(s, probe))
        
    def test___contains__(self):
        # sorted_collection = SortedCollection(iterable, key)
        # assert_equal(expected, sorted_collection.__contains__(item))
        raise SkipTest 

    def test___getitem__(self):
        # sorted_collection = SortedCollection(iterable, key)
        # assert_equal(expected, sorted_collection.__getitem__(i))
        raise SkipTest 

    def test___init__(self):
        # sorted_collection = SortedCollection(iterable, key)
        raise SkipTest 

    def test___iter__(self):
        # sorted_collection = SortedCollection(iterable, key)
        # assert_equal(expected, sorted_collection.__iter__())
        raise SkipTest 

    def test___len__(self):
        # sorted_collection = SortedCollection(iterable, key)
        # assert_equal(expected, sorted_collection.__len__())
        raise SkipTest 

    def test___reduce__(self):
        # sorted_collection = SortedCollection(iterable, key)
        # assert_equal(expected, sorted_collection.__reduce__())
        raise SkipTest 

    def test___repr__(self):
        # sorted_collection = SortedCollection(iterable, key)
        # assert_equal(expected, sorted_collection.__repr__())
        raise SkipTest 

    def test___reversed__(self):
        # sorted_collection = SortedCollection(iterable, key)
        # assert_equal(expected, sorted_collection.__reversed__())
        raise SkipTest 

    def test_clear(self):
        # sorted_collection = SortedCollection(iterable, key)
        # assert_equal(expected, sorted_collection.clear())
        raise SkipTest 

    def test_copy(self):
        # sorted_collection = SortedCollection(iterable, key)
        # assert_equal(expected, sorted_collection.copy())
        raise SkipTest 

    def test_count(self):
        # sorted_collection = SortedCollection(iterable, key)
        # assert_equal(expected, sorted_collection.count(item))
        raise SkipTest 

    def test_find(self):
        # sorted_collection = SortedCollection(iterable, key)
        # assert_equal(expected, sorted_collection.find(k))
        raise SkipTest 

    def test_find_ge(self):
        # sorted_collection = SortedCollection(iterable, key)
        # assert_equal(expected, sorted_collection.find_ge(k))
        raise SkipTest 

    def test_find_gt(self):
        # sorted_collection = SortedCollection(iterable, key)
        # assert_equal(expected, sorted_collection.find_gt(k))
        raise SkipTest 

    def test_find_le(self):
        # sorted_collection = SortedCollection(iterable, key)
        # assert_equal(expected, sorted_collection.find_le(k))
        raise SkipTest 

    def test_find_lt(self):
        # sorted_collection = SortedCollection(iterable, key)
        # assert_equal(expected, sorted_collection.find_lt(k))
        raise SkipTest 



    def test_insert(self):
        # sorted_collection = SortedCollection(iterable, key)
        # assert_equal(expected, sorted_collection.insert(item))
        raise SkipTest 

    def test_insert_right(self):
        # sorted_collection = SortedCollection(iterable, key)
        # assert_equal(expected, sorted_collection.insert_right(item))
        raise SkipTest 

    def test_pop(self):
        # sorted_collection = SortedCollection(iterable, key)
        # assert_equal(expected, sorted_collection.pop(i))
        raise SkipTest 

    def test_remove(self):
        # sorted_collection = SortedCollection(iterable, key)
        # assert_equal(expected, sorted_collection.remove(item))
        raise SkipTest 
    
class TestRecord:
    def test___getattr__(self):
        # record = Record(*args, **kwargs)
        # assert_equal(expected, record.__getattr__(name))
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # record = Record(*args, **kwargs)
        raise SkipTest # TODO: implement your test here

    def test___setattr__(self):
        # record = Record(*args, **kwargs)
        # assert_equal(expected, record.__setattr__(name, value))
        raise SkipTest # TODO: implement your test here

    def test___str__(self):
        # record = Record(*args, **kwargs)
        # assert_equal(expected, record.__str__())
        raise SkipTest # TODO: implement your test here

if __name__ == "__main__":
    runmodule()


"""
    

    def slow_find(seq, k):
        'First item with a key equal to k. -1 if not found'
        for item in seq:
            if item == k:
                return item
        return -1

    def slow_find_le(seq, k):
        'Last item with a key less-than or equal to k.'
        for item in reversed(seq):
            if item <= k:
                return item
        return -1

    def slow_find_lt(seq, k):
        'Last item with a key less-than k.'
        for item in reversed(seq):
            if item < k:
                return item
        return -1

    def slow_find_ge(seq, k):
        'First item with a key-value greater-than or equal to k.'
        for item in seq:
            if item >= k:
                return item
        return -1

    def slow_find_gt(seq, k):
        'First item with a key-value greater-than or equal to k.'
        for item in seq:
            if item > k:
                return item
        return -1
    
    
            for probe in pool:
                assert repr(ve2no(sc.index, probe)) == repr(slow_index(s, probe))
                assert repr(ve2no(sc.find, probe)) == repr(slow_find(s, probe))
                assert repr(ve2no(sc.find_le, probe)) == repr(slow_find_le(s, probe))
                assert repr(ve2no(sc.find_lt, probe)) == repr(slow_find_lt(s, probe))
                assert repr(ve2no(sc.find_ge, probe)) == repr(slow_find_ge(s, probe))
                assert repr(ve2no(sc.find_gt, probe)) == repr(slow_find_gt(s, probe))
            for i, item in enumerate(s):
                assert repr(item) == repr(sc[i])        # test __getitem__
                assert item in sc                       # test __contains__ and __iter__
                assert s.count(item) == sc.count(item)  # test count()
            assert len(sc) == n                         # test __len__
            assert list(map(repr, reversed(sc))) == list(map(repr, reversed(s)))    # test __reversed__
            assert list(sc.copy()) == list(sc)          # test copy()
            sc.clear()                                  # test clear()
            assert len(sc) == 0

    sd = SortedCollection('The quick Brown Fox jumped'.split(), key=str.lower)
    assert sd._keys == ['brown', 'fox', 'jumped', 'quick', 'the']
    assert sd._items == ['Brown', 'Fox', 'jumped', 'quick', 'The']
    assert sd._key == str.lower
    assert repr(sd) == "SortedCollection(['Brown', 'Fox', 'jumped', 'quick', 'The'], key=lower)"
    sd.key = str.upper
    assert sd._key == str.upper
    assert len(sd) == 5
    assert list(reversed(sd)) == ['The', 'quick', 'jumped', 'Fox', 'Brown']
    for item in sd:
        assert item in sd
    for i, item in enumerate(sd):
        assert item == sd[i]
    sd.insert('jUmPeD')
    sd.insert_right('QuIcK')
    assert sd._keys ==['BROWN', 'FOX', 'JUMPED', 'JUMPED', 'QUICK', 'QUICK', 'THE']
    assert sd._items == ['Brown', 'Fox', 'jUmPeD', 'jumped', 'quick', 'QuIcK', 'The']
    assert sd.find_le('JUMPED') == 'jumped', sd.find_le('JUMPED')
    assert sd.find_ge('JUMPED') == 'jUmPeD'
    assert sd.find_le('GOAT') == 'Fox'
    assert sd.find_ge('GOAT') == 'jUmPeD'
    assert sd.find('FOX') == 'Fox'
    assert sd[3] == 'jumped'
    assert sd[3:5] ==['jumped', 'quick']
    assert sd[-2] == 'QuIcK'
    assert sd[-4:-2] == ['jumped', 'quick']
    for i, item in enumerate(sd):
        assert sd.index(item) == i
    try:
        sd.index('xyzpdq')
    except ValueError:
        pass
    else:
        assert 0, 'Oops, failed to notify of missing value'
    sd.remove('jumped')
    assert list(sd) == ['Brown', 'Fox', 'jUmPeD', 'quick', 'QuIcK', 'The']

    import doctest
    from operator import itemgetter
    print(doctest.testmod())
"""