from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *

from Goulib.interval import *

class TestInInterval:
    def test_in_interval(self):
        assert_equal(in_interval([1,2], 1),True)
        assert_equal(in_interval([2,1], 1),True) #interval might be unordered
        assert_equal(in_interval((2,1), 1),True) #or defined by a tuple
        assert_equal(in_interval([1,2], 2,closed=True),True)
        assert_equal(in_interval([1,2], 2,closed=False),False)

class TestIntersect:
    def test_intersect(self):
        assert_equal(intersect([1,3],[2,4]),True)
        assert_equal(intersect([3,1],(4,2)),True)
        assert_equal(intersect((1,2),[2,4]),False)
        assert_equal(intersect((5,1),(2,3)),True)

class TestIntersection:
    def test_intersection(self):
        assert_equal(intersection([1,3],(4,2)),(2,3))
        assert_equal(intersection([1,5],(3,2)),(2,3))
        assert_equal(intersection((1,2),[2,4]),(2,2))
        assert_equal(intersection((1,2),[3,4]),None)

class TestIntersectlen:
    def test_intersectlen(self):
        assert_equal(intersectlen([1,5],(3,2)),1)
        assert_equal(intersectlen((1,2),[2,4]),0)
        assert_equal(intersectlen((1,2),[3,4],None),None)

class TestInterval:
    @classmethod
    def setup_class(self):
        self.none = Interval(None,None) #required for Box, equivalent t
        self.i12 = Interval(1,2)
        self.i13 = Interval(1,3)
        self.i23 = Interval(2,3)
        self.i24 = Interval(2,4)
        self.i25 = Interval(5,2)
        assert_equal(self.i25,Interval(2,5)) #check order
        self.i33 = Interval(3,3) #empty
        self.i34 = Interval(3,4)
        
    def test___init__(self):
        pass #tested above
    
    def test___repr__(self):
        assert_equal(repr(self.i12),'[1,2)')
                     
    def test___str__(self):
        assert_equal(str(self.i12),'[1,2)')
        
    def test___hash__(self):
        """test that we can use an Interval as key in a dict and retrieve it with a different Interval with same values"""
        dict={}
        dict[self.i12]=self.i12
        assert_equal(dict[Interval(2,1)],self.i12)
        
    def test___lt__(self):
        assert_equal(self.i12<self.i34,True)
        assert_equal(self.i12>self.i34,False)

    def test___contains__(self):
        assert_true(2 in self.i13)
        assert_false(3 in self.i13)

    def test_empty(self):
        assert_true(self.i33.empty())
        assert_false(self.i13.empty())

    def test_hull(self):
        assert_equal(self.i12.hull(self.i34),Interval(1,4))

    def test_intersection(self):
        assert_equal(self.i12.intersection(self.i34),None)
        assert_equal(self.i13.intersection(self.i25),self.i23)
        assert_equal(self.i25.intersection(self.i13),self.i23)

    def test_overlap(self):
        assert_false(Interval(1,2).overlap(Interval(3,4)))
        assert_true(Interval(1,3).overlap(Interval(2,5)))
        
    def test_separation(self):
        assert_equal(self.i12.separation(self.i23),0)
        assert_equal(self.i12.separation(self.i34),3-2)
        assert_equal(self.i34.separation(self.i12),3-2)
    
    def test_subset(self):
        assert_true(Interval(1,3).subset(Interval(1,3)))
        assert_false(Interval(1,3).subset(Interval(1,2)))
        assert_false(Interval(2,3).subset(Interval(1,2)))

    def test_proper_subset(self):
        assert_false(Interval(1,3).proper_subset(Interval(1,3)))
        eps=1E-12
        assert_true(Interval(1,3).proper_subset(Interval(1-eps,3+eps)))

    def test_singleton(self):
        assert_true(Interval(1,2).singleton())
        assert_false(Interval(1,3).singleton())

    def test___add__(self):
        assert_equal(Interval(1,3)+Interval(2,4),Interval(1,4))
        i24=Interval(2,3)+Interval(3,4)
        assert_equal(i24,self.i24)
        assert_equal(Interval(4,5)+Interval(2,3),Intervals([Interval(4,5),Interval(2,3)]))
        a=Interval(5,6)+Interval(2,3)
        a+=Interval(3,4)
        b=Intervals([Interval(5,6),Interval(2,4)])
        assert_equal(a,b)

    def test___eq__(self):
        pass #tested in other tests...

    def test___iadd__(self):
        pass #tested in other tests...

    def test_center(self):
        pass #tested in other tests...
    def test_size(self):
        pass #tested in other tests...
    
    def test___call__(self):
        # interval = Interval(start, end)
        # assert_equal(expected, interval.__call__())
        raise SkipTest 

    def test___nonzero__(self):
        # interval = Interval(start, end)
        # assert_equal(expected, interval.__nonzero__())
        raise SkipTest 

class TestIntervals:
    @classmethod
    def setup_class(self):
        i12 = Interval(1,2)
        i13 = Interval(1,3)
        i24 = Interval(2,4)
        i56 = Interval(5,6)
        self.intervals=Intervals([i24,i13,i12,i56])
        assert_equal(str(self.intervals),'[[1,4), [5,6)]')
    
    def test___init__(self):
        pass #tested above
        
    def test___call__(self):
        assert_equal(self.intervals(2),Interval(1,4))
        assert_equal(self.intervals(4),None)
        assert_equal(self.intervals(5),Interval(5,6))

    def test_insert(self):
        pass #tested above
    
    def test_extend(self):
        pass #tested above

    def test___add__(self):
        i=self.intervals+Interval(-1,-3)
        assert_equal(str(i),'[[-3,-1), [1,4), [5,6)]')

    def test___iadd__(self):
        i=Intervals(self.intervals)
        i+=Interval(-1,-3)
        assert_equal(str(i),'[[-3,-1), [1,4), [5,6)]')

    def test___repr__(self):
        # intervals = Intervals()
        # assert_equal(expected, intervals.__repr__())
        raise SkipTest 

class TestBox:
    @classmethod
    def setup_class(self):
        self.empty=Box(2)
        self.unit=Box(Interval(0,1),Interval(0,1))
        self.box=Box((-1,4),[3,-2])
        self.copy=Box(self.box)
        assert_equal(self.box,self.copy)
        
    def test___init__(self):
        pass #tested in setup_class
    
    def test___repr__(self):
        assert_equal(repr(self.box),'[[-1,3), [-2,4)]')

    def test_min(self):
        assert_equal(self.unit.min, (0,0))
        assert_equal(self.box.min, (-1,-2))

    def test_max(self):
        assert_equal(self.unit.max, (1,1))
        assert_equal(self.box.max, (3,4))
        
    def test_size(self):
        assert_equal(self.box.size, (4,6))
        
    def test_center(self):
        assert_equal(self.box.center, (1,1))

    def test___add__(self):
        box=self.unit+(2,0)
        assert_equal(repr(box),'[[0,2), [0,1)]')
        box=box+Box((-2,-1),(.5,.5))
        assert_equal(repr(box),'[[-2,2), [-1,1)]')
        
    def test___iadd__(self):
        box=Box(self.unit)
        box+=(2,0)
        assert_equal(repr(box),'[[0,2), [0,1)]')
        box+=Box((-2,-1),(.5,.5))
        assert_equal(repr(box),'[[-2,2), [-1,1)]')
        
    def test_end(self):
        pass #tested in other tests...

    def test_start(self):
        pass #tested in other tests...

    def test___contains__(self):
        # box = Box(*args)
        # assert_equal(expected, box.__contains__(other))
        raise SkipTest 

    def test___nonzero__(self):
        # box = Box(*args)
        # assert_equal(expected, box.__nonzero__())
        raise SkipTest 

    def test_empty(self):
        # box = Box(*args)
        # assert_equal(expected, box.empty())
        raise SkipTest 

    def test_corner(self):
        # box = Box(*args)
        # assert_equal(expected, box.corner(n))
        raise SkipTest

if __name__ == "__main__":
    runmodule()
