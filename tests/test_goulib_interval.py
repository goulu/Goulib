from goulib.tests import *

from goulib.interval import *


class TestInInterval:
    def test_in_interval(self):
        assert in_interval([1, 2], 1) == True
        assert in_interval([2, 1], 1) == True  # interval might be unordered
        assert in_interval((2, 1), 1) == True  # or defined by a tuple
        assert in_interval([1, 2], 2, closed=True) == True
        assert in_interval([1, 2], 2, closed=False) == False


class TestIntersect:
    def test_intersect(self):
        assert intersect([1, 3], [2, 4]) == True
        assert intersect([3, 1], (4, 2)) == True
        assert intersect((1, 2), [2, 4]) == False
        assert intersect((5, 1), (2, 3)) == True


class TestIntersection:
    def test_intersection(self):
        assert intersection([1, 3], (4, 2)) == (2, 3)
        assert intersection([1, 5], (3, 2)) == (2, 3)
        assert intersection((1, 2), [2, 4]) == (2, 2)
        assert intersection((1, 2), [3, 4]) == None


class TestIntersectlen:
    def test_intersectlen(self):
        assert intersectlen([1, 5], (3, 2)) == 1
        assert intersectlen((1, 2), [2, 4]) == 0
        assert intersectlen((1, 2), [3, 4], None) == None


class TestInterval:
    @classmethod
    def setup_class(self):
        self.none = Interval(None, None)  # required for Box, equivalent t
        self.i12 = Interval(1, 2)
        self.i13 = Interval(1, 3)
        self.i23 = Interval(2, 3)
        self.i24 = Interval(2, 4)
        self.i25 = Interval(5, 2)
        assert self.i25 == Interval(2, 5)  # check order
        self.i33 = Interval(3, 3)  # empty
        self.i34 = Interval(3, 4)

    def test___init__(self):
        pass  # tested above

    def test___repr__(self):
        assert repr(self.i12) == '[1,2)'

    def test___str__(self):
        assert str(self.i12) == '[1,2)'

    def test___hash__(self):
        """test that we can use an Interval as key in a dict and retrieve it with a different Interval with same values"""
        dict = {}
        dict[self.i12] = self.i12
        assert dict[Interval(2, 1)] == self.i12

    def test___lt__(self):
        assert (self.i12 < self.i34) == True
        assert (self.i12 > self.i34) == False

    def test___contains__(self):
        assert 2 in self.i13
        assert not 3 in self.i13

    def test_empty(self):
        assert self.i33.empty()
        assert not self.i13.empty()

    def test_hull(self):
        assert self.i12.hull(self.i34) == Interval(1, 4)

    def test_intersection(self):
        assert self.i12.intersection(self.i34) == None
        assert self.i13.intersection(self.i25) == self.i23
        assert self.i25.intersection(self.i13) == self.i23

    def test_overlap(self):
        assert not Interval(1, 2).overlap(Interval(3, 4))
        assert Interval(1, 3).overlap(Interval(2, 5))

    def test_separation(self):
        assert self.i12.separation(self.i23) == 0
        assert self.i12.separation(self.i34) == 3-2
        assert self.i34.separation(self.i12) == 3-2

    def test_subset(self):
        assert Interval(1, 3).subset(Interval(1, 3))
        assert not Interval(1, 3).subset(Interval(1, 2))
        assert not Interval(2, 3).subset(Interval(1, 2))

    def test_proper_subset(self):
        assert not Interval(1, 3).proper_subset(Interval(1, 3))
        eps = 1E-12
        assert Interval(1, 3).proper_subset(Interval(1-eps, 3+eps))

    def test_singleton(self):
        assert Interval(1, 2).singleton()
        assert not Interval(1, 3).singleton()

    def test___add__(self):
        assert Interval(1, 3)+Interval(2, 4) == Interval(1, 4)
        i24 = Interval(2, 3)+Interval(3, 4)
        assert i24 == self.i24
        assert Interval(4, 5)+Interval(2,
                                       3) == Intervals([Interval(4, 5), Interval(2, 3)])
        a = Interval(5, 6)+Interval(2, 3)
        a += Interval(3, 4)
        b = Intervals([Interval(5, 6), Interval(2, 4)])
        assert a == b

    def test___eq__(self):
        pass  # tested in other tests...

    def test___iadd__(self):
        pass  # tested in other tests...

    def test_center(self):
        pass  # tested in other tests...

    def test_size(self):
        pass  # tested in other tests...

    def test___call__(self):
        # interval = Interval(start, end)
        # assert_equal(expected, interval.__call__())
        pass  # TODO: implement

    def test___nonzero__(self):
        # interval = Interval(start, end)
        # assert_equal(expected, interval.__nonzero__())
        pass  # TODO: implement


class TestIntervals:
    @classmethod
    def setup_class(self):
        i12 = Interval(1, 2)
        i13 = Interval(1, 3)
        i24 = Interval(2, 4)
        i56 = Interval(5, 6)
        self.intervals = Intervals([i24, i13, i12, i56])
        assert str(self.intervals) == '[[1,4), [5,6)]'

    def test___init__(self):
        pass  # tested above

    def test___call__(self):
        assert self.intervals(2) == Interval(1, 4)
        assert self.intervals(4) == None
        assert self.intervals(5) == Interval(5, 6)

    def test_insert(self):
        pass  # tested above

    def test_extend(self):
        pass  # tested above

    def test___add__(self):
        i = self.intervals+Interval(-1, -3)
        assert str(i) == '[[-3,-1), [1,4), [5,6)]'

    def test___iadd__(self):
        i = Intervals(self.intervals)
        i += Interval(-1, -3)
        assert str(i) == '[[-3,-1), [1,4), [5,6)]'

    def test___repr__(self):
        # intervals = Intervals()
        # assert_equal(expected, intervals.__repr__())
        pass  # TODO: implement


class TestBox:
    @classmethod
    def setup_class(self):
        self.empty = Box(2)
        self.unit = Box(Interval(0, 1), Interval(0, 1))
        self.box = Box((-1, 4), [3, -2])
        self.copy = Box(self.box)
        assert self.box == self.copy

    def test___init__(self):
        pass  # tested in setup_class

    def test___repr__(self):
        assert repr(self.box) == '[[-1,3), [-2,4)]'

    def test_min(self):
        assert self.unit.min == (0, 0)
        assert self.box.min == (-1, -2)

    def test_max(self):
        assert self.unit.max == (1, 1)
        assert self.box.max == (3, 4)

    def test_size(self):
        assert self.box.size == (4, 6)

    def test_center(self):
        assert self.box.center == (1, 1)

    def test___add__(self):
        box = self.unit+(2, 0)
        assert repr(box) == '[[0,2), [0,1)]'
        box = box+Box((-2, -1), (.5, .5))
        assert repr(box) == '[[-2,2), [-1,1)]'

    def test___iadd__(self):
        box = Box(self.unit)
        box += (2, 0)
        assert repr(box) == '[[0,2), [0,1)]'
        box += Box((-2, -1), (.5, .5))
        assert repr(box) == '[[-2,2), [-1,1)]'

    def test_end(self):
        pass  # tested in other tests...

    def test_start(self):
        pass  # tested in other tests...

    def test___contains__(self):
        # box = Box(*args)
        # assert_equal(expected, box.__contains__(other))
        pass  # TODO: implement

    def test___nonzero__(self):
        # box = Box(*args)
        # assert_equal(expected, box.__nonzero__())
        pass  # TODO: implement

    def test_empty(self):
        # box = Box(*args)
        # assert_equal(expected, box.empty())
        pass  # TODO: implement

    def test_corner(self):
        # box = Box(*args)
        # assert_equal(expected, box.corner(n))
        pass  # TODO: implement

    def test___call__(self):
        # box = Box(*args)
        # assert_equal(expected, box.__call__())
        pass  # TODO: implement
