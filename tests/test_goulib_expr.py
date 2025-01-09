from goulib.tests import *  # pylint: disable=wildcard-import, unused-wildcard-import
from goulib.expr import *   # pylint: disable=wildcard-import, unused-wildcard-import

from goulib.table import Table

from math import sin, pi
from goulib.math2 import sqrt

import os
from goulib.plot import save

path = os.path.dirname(os.path.abspath(__file__))
results = path + '/results/expr/'  # path for results


class TestExpr:

    @classmethod
    def setup_class(cls):
        cls.t = Table(path + '/data/expr.csv')
        for e in cls.t:
            e[0] = Expr(e[0])

        cls.f = Expr('3*x+2')
        cls.f0 = Expr(0)
        cls.f1 = Expr(1)
        cls.fx = Expr('x')
        cls.fx2 = Expr('x**2')
        cls.fs = Expr('sin(x)')

        cls.fb1 = Expr('x>1')
        cls.fb2 = Expr('x>2')
        cls.fb3 = Expr('a==a')  # any way to simplify it to constant True ?

        cls.e1 = Expr('3*x+2')  # a very simple expression
        cls.e2 = Expr(cls.fs)

        cls.e3 = Expr(sqrt)(cls.e1)  # Expr can be composed

        cls.xy = Expr('x*y')
        cls.long = Expr('(x*3+(a+b)*y)/x**(3*a*y)')

        cls.true = Expr(True)  # make sure it works
        cls.false = Expr('False')  # make sure it creates a bool

        cls.sqrt = Expr(sqrt)

        cls.euler = Expr("e**(i*pi)")

    def test___init__(self):
        assert Expr(1) == 1

        e2 = Expr(lambda x: 3 * x + 2)
        assert repr(e2) == '3*x+2'

        def f(x):
            return 3 * x + 2

        e3 = Expr(f)  # same as function
        assert repr(e3) == '3*x+2'

        assert repr(Expr(sin)) == 'sin(x)'

        assert repr(Expr(True)) == 'True'
        assert repr(Expr('False')) == 'False'

    def test___call__(self):
        assert self.f1() == 1  # constant function
        assert self.fx([-1, 0, 1]) == [-1, 0, 1]
        assert self.fb1([0, 1, 2]) == [False, False, True]
        assert self.xy(x=2, y=3) == 6
        # test substitution
        e = self.xy(x=2)
        assert str(e) == '2y'
        e = self.euler()
        # assert_equal(str(e), '-1') #TODO:evaluate euler's formula correctly...

    def test___str__(self):
        assert str(Expr(pi)) == 'pi'

        assert str(Expr('3*5')) == '3*5'
        assert str(Expr('3+(-2)')) == '3-2'
        assert str(Expr('3-(-2)')) == '3+2'
        assert str(Expr('3*(-2)')) == '3(-2)'
        assert str(Expr('-(3+2)')) == '-(3+2)'

        assert str(self.f) == '3x+2'
        assert str(self.f1) == '1'
        assert str(self.fx) == 'x'
        assert str(self.fs) == 'sin(x)'
        assert str(self.fb1) == 'x > 1'
        assert str(self.long) == '(3x+(a+b)y)/x^(3ay)'

        # test multiplication commutativity and simplification
        assert str(Expr('x*3+(a+b)')) == '3x+a+b'

        # test multiplication commutativity and simplification
        assert str(Expr('x*3+(a+b)')) == '3x+a+b'

    def test___repr__(self):
        assert str(Expr(pi)) == 'pi'

        assert repr(Expr('3*5')) == '3*5'
        assert repr(Expr('3+(-2)')) == '3-2'
        assert repr(Expr('3-(-2)')) == '3+2'
        assert repr(Expr('3*(-2)')) == '3*(-2)'
        assert repr(Expr('-(3+2)')) == '-(3+2)'

        assert repr(self.f) == '3*x+2'
        assert repr(self.f1) == '1'
        assert repr(self.fx) == 'x'
        assert repr(self.fs) == 'sin(x)'
        assert repr(self.fb1) == 'x > 1'
        assert repr(self.long) == '(3*x+(a+b)*y)/x**(3*a*y)'
        assert repr(self.sqrt) == 'sqrt(x)'

        # test multiplication commutativity and simplification
        assert repr(Expr('x*3+(a+b)')) == '3*x+a+b'

    def test_latex(self):
        assert self.f.latex() == '3x+2'
        assert self.f1.latex() == '1'
        assert self.fx.latex() == 'x'
        assert self.fs.latex() == r'\sin\left(x\right)'
        assert self.fb1.latex() == r'x \gtr 1'
        assert self.fs(self.fx2).latex() == r'\sin\left(x^2\right)'
        assert (self.long.latex() ==
                r'\frac{3x+\left(a+b\right)y}{x^{3ay}}')
        assert self.sqrt.latex() == r'\sqrt{x}'
        assert Expr(1. / 3).latex() == r'\frac{1}{3}'
        l = Expr('sqrt(x*3+(a+b)*y)/x**(3*a*y)').latex()
        assert l == r'\frac{\sqrt{3x+\left(a+b\right)y}}{x^{3ay}}'

    def test__repr_html_(self):
        assert self.sqrt._repr_html_() == r'${\sqrt{x}}$'

    def test_plot(self):
        save([Expr('1/x')], results + 'oneoverx.png', x=range(-100, 100))
        save([Expr('sin(x/10)/(x/10)')], results +
             'sinxoverx.png', x=range(-100, 100))
        save([self.e3], results + 'sqrt.png')

    def test___add__(self):
        f = self.fx + self.f1
        assert str(f) == 'x+1'
        assert f([-1, 0, 1]) == [0, 1, 2]

    def test___neg__(self):
        f= -self.f0
        assert str(f) == '0'
        f = -self.f1
        assert str(f) == '-1'
        f = -self.fx
        assert str(f) == '-x'
        assert f([-1, 0, 1]) == [1, 0, -1]

    def test___sub__(self):
        f = self.f1 - self.fx
        assert str(f) == '1-x'
        assert f([-1, 0, 1]) == [2, 1, 0]

    def test___mul__(self):
        f2 = self.f1 * 2
        f2x = f2 * self.fx
        assert f2x([-1, 0, 1]) == [-2, 0, 2]

    def test___rmul__(self):
        f2 = 2 * self.f1
        f2x = f2 * self.fx
        assert f2x([-1, 0, 1]) == [-2, 0, 2]

    def test___div__(self):
        f2 = self.f1 * 2
        fx = self.fx / f2
        assert fx([-1, 0, 1]) == [-0.5, 0, 0.5]

    def test_applx(self):
        f = self.fs.applx(self.fx2)
        assert f(2) == sin(4)

    def test_apply(self):
        f1 = self.fx.apply(self.fs)
        assert f1([-1, 0, 1]) == [sin(-1), 0, sin(1)]
        f2 = self.fs(self.fx)
        assert f2([-1, 0, 1]) == [sin(-1), 0, sin(1)]

    def test___not__(self):
        assert(str(~self.true) == "False")
        assert(str(~self.false) == "True")
        assert(str(~~self.true) == "True")
        fb = ~self.fb1
        assert fb([0, 1, 2]) == [True, True, False]

    def test___and__(self):
        fb = self.fb1 & self.fb2
        assert fb([1, 2, 3]) == [False, False, True]

    def test___or__(self):
        fb = self.fb1 | self.fb2
        assert fb([1, 2, 3]) == [False, True, True]

    def test___xor__(self):
        fb = self.fb1 ^ self.fb2
        assert fb([1, 2, 3]) == [False, True, False]

    def test___lt__(self):
        assert Expr(1) < 2
        assert Expr("~2<1")

    def test___lshift__(self):
        e = self.fx << 1
        assert e(0) == 1

    def test___rshift__(self):
        e = self.fx >> 2
        assert e(0) == -2

    def test___eq__(self):
        assert Expr(1) == 1
        assert Expr(1) == Expr('2-1')()
        assert self.f == '3*x+2'
        assert self.f1 == 1
        assert self.fx == 'x'
        assert self.fs == 'sin(x)'
        assert self.fb1 == 'x > 1'
        assert self.long == '(x*3+(a+b)*y)/x**(3*a*y)'

    def test___ne__(self):
        assert Expr(1) != Expr(2)
        assert Expr(1) != 2

    def test_save(self):
        self.e2(self.e1).save(path + '/results/expr.png')
        self.e2(self.e1).save(path + '/results/expr.svg')

    def test___invert__(self):
        # expr = Expr(f)
        # assert_equal(expected, expr.__invert__())
        pytest.skip("not yet implemented")  # TODO: implement

    def test___truediv__(self):
        # expr = Expr(f)
        # assert_equal(expected, expr.__truediv__(right))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_isconstant(self):
        # expr = Expr(f)
        # assert_equal(expected, expr.isconstant())
        pytest.skip("not yet implemented")  # TODO: implement

    def test_complexity(self):
        e1 = Expr('44+4*(-4)')
        e2 = Expr('44-4*4')
        assert e1() == e2()
        assert e1.complexity() > e2.complexity()

        e1 = Expr('2/sqrt(3)/sqrt(5)')
        e2 = Expr('2/(sqrt(3)*sqrt(5))')
        assert pytest.approx(e1()) == e2()
        assert e1.complexity() >= e2.complexity()


class TestEval:
    def test_eval(self):
        pytest.skip("tested in Expr")

class TestGetFunctionSource:
    def test_get_function_source(self):
        pytest.skip("tested in Expr")


class TestTextVisitor:

    def test___init__(self):
        pytest.skip("tested in Expr")

    def test_generic_visit(self):
        pytest.skip("tested in Expr")

    def test_prec(self):
        pytest.skip("tested in Expr")

    def test_prec_BinOp(self):
        pytest.skip("tested in Expr")

    def test_prec_UnaryOp(self):
        pytest.skip("tested in Expr")

    def test_visit_BinOp(self):
        pytest.skip("tested in Expr")

    def test_visit_Call(self):
        pytest.skip("tested in Expr")

    def test_visit_Compare(self):
        pytest.skip("tested in Expr")

    def test_visit_Name(self):
        pytest.skip("tested in Expr")

    def test_visit_NameConstant(self):
        pytest.skip("tested in Expr")

    def test_visit_Num(self):
        pytest.skip("tested in Expr")

    def test_visit_UnaryOp(self):
        pytest.skip("tested in Expr")
