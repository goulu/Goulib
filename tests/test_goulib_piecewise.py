from goulib.tests import *
from goulib.plot import save

from goulib.piecewise import *
from goulib.itertools2 import arange
from goulib.math2 import inf

from math import *
import os
path = os.path.dirname(os.path.abspath(__file__))
results = path+'/results/piecewise/'  # path for results

class TestPiecewise:
    @classmethod
    def setup_class(self):
        # piecewise continuous
        self.p1 = Piecewise([(4, 4), (3, 3.0), (1, 1), (5, 0)])
        self.p2 = Piecewise(default=1)

        self.p2 += (2.5, 1, 6.5)
        self.p2 += (1.5, 1, 3.5)
        

        # boolean
        self.b0 = Piecewise([], False)
        self.b1 = Piecewise([(2, True)], False)
        self.b2 = Piecewise([(1, True), (2, False), (3, True)], False)

        # simple function
        self.f = Piecewise().append(0, cos).append(1, lambda x: x*x)

    def test___init__(self):
        assert self.p1 == Piecewise([(4, 4), (3, 3.0), (1, 1), (5, 0)])

    def test_append(self):
        # f was created by appends in setup_class
        assert self.f(range(-1,5)) == [0, 1, 1, 4, 9, 16]

    def test_repr(self):
        res=repr(self.p1) 
        assert res == '[(-inf, 0), (1, 1), (3, 3), (4, 4), (5, 0)]'

    def test___call__(self):
        y = [self.p1(x) for x in range(6)]
        assert y == [0, 1, 1, 3, 4, 0]

        # test function of Expr
        y = self.f(arange(0., 2.1, .1))
        assert pytest.approx(y, abs=0.001) == [1, 0.995, 0.980, 0.955, 0.9210, 0.878, 0.825, 0.764, 0.697, 0.622, 0.540, 1.21, 1.44, 1.69, 1.96, 2.25, 2.56, 2.89, 3.24, 3.61, 4]

    def test_points(self):
        assert self.p1.points(0, 5) == (
            [0, 1, 1, 3, 3, 4, 4, 5, 5], [0, 0, 1, 1, 3, 3, 4, 4, 0])
        assert self.p1.points(-1, 6) == ([-1, 1, 1, 3, 3,
                                          4, 4, 5, 5, 6], [0, 0, 1, 1, 3, 3, 4, 4, 0, 0])
        assert self.b2.points(0, 3) == ([0, 1, 1, 2, 2, 3, 3], [
            False, False, True, True, False, False, True])
        assert len(self.f) == 3
        assert self.f.points(-1,4)==([-1,0,0,1,4], [0, 0, 1, 1, 16])



    def test_extend(self):
        pytest.skip("tested at __init__")

    def test_index(self):
        pytest.skip("tested by most other tests")

    def test___getitem__(self):
        pytest.skip("tested by most other tests")

    def test___len__(self):
        assert len(self.p1) == 5

    def test___add__(self):
        p = self.p1+self.p2
        assert p(range(8)) == [1, 2, 3, 6, 6, 2, 2, 1]

    def test___sub__(self):
        p = self.p1-self.p2
        y = [p(x) for x in range(8)]
        assert y == [-1, 0, -1, 0, 2, -2, -2, -1]

    def test___neg__(self):
        n=-self.p1
        assert n == [(-inf, 0), (1, -1), (3, -3), (4, -4), (5, 0)]
        assert n == Piecewise([(-inf, 0), (1, -1), (3, -3), (4, -4), (5, 0)])

    def test___mul__(self):
        p = self.p1*self.p2
        y = [p(x) for x in range(8)]
        assert y == [0, 1, 2, 9, 8, 0, 0, 0]

    def test___div__(self):
        p = self.p1/self.p2
        y = [p(x) for x in range(8)]
        assert y == [0.0, 1.0, 0.5, 1.0, 2.0, 0.0, 0.0, 0.0]

    def test___str__(self):
        assert str(self.p1) == '[(-inf, 0), (1, 1), (3, 3), (4, 4), (5, 0)]'
        assert str(self.f) == '[(-inf, 0), (0, cos(x)), (1, x*x)]'

    def test_plot(self):
        save([self.p1], results+'plot_p1.png')
        save([self.p2], results+'plot_p2.png')
        save([self.f], results+'plot_f.png')

    def test___iter__(self):
        xy = list(self.p1+self.p2)
        assert xy==[(-inf, 1), (1, 2), (1.5, 3), (2.5, 4), (3, 6), (3.5, 5), (4, 6), (5, 2), (6.5, 1)]

    def test___invert__(self):
        assert ~self.b2 == [(-inf, True), (1, False), (2, True), (3, False)]

    def test___lshift__(self):
        assert self.b2 << 2 == [
            (-inf, False), (-1, True), (0, False), (1, True)]

    def test___rshift__(self):
        assert self.b2 >> 3 == [
            (-inf, False), (4, True), (5, False), (6, True)]

    def test___and__(self):
        b = self.b1 & self.b2
        assert b == [(-inf, False), (3, True)]

    def test___or__(self):
        b = self.b1 | self.b2
        assert b == [(-inf, False), (1, True)]

    def test___xor__(self):
        b = self.b1 ^ self.b2
        assert b == [(-inf, False), (1, True), (3, False)]

    def test_applx(self):
        pytest.skip("tested in shift operators")

    def test_apply(self):
        pytest.skip("tested in most operators")

    def test_iapply(self):
        # piecewise = Piecewise(init, default, start)
        # assert_equal(expected, piecewise.iapply(f, right, name))
        pytest.skip("implicitely tested elsewhere") 

    def test_save(self):
        self.p2.save(results+'save_p2.png', xmax=7, ylim=(-1, 5))

    def test_svg(self):
        svg = self.p2._repr_svg_(xmax=7, ylim=(-1, 5))  # return IPython object
        with open(results+'p2.svg', 'wb') as f:
            f.write(svg.encode('utf-8'))
