from goulib.tests import *

from goulib.polynomial import *


class TestPolynomial:
    @classmethod
    def setup_class(self):
        self.p = Polynomial([1, 2, 3])
        self.p2 = Polynomial([1, 2])
        self.p3 = Polynomial('+3*x -4x^2 +7x^5-x+1')
        self.p4 = Polynomial('5x+x^2')

    def test___init__(self):
        pass  # tested above

    def test___call__(self):
        assert self.p(0) == 1
        assert self.p(1) == 6
        assert self.p(2) == 17

    def test___add__(self):
        assert self.p + self.p2 == Polynomial('3x^2 + 4x + 2')
        assert self.p + 1 == Polynomial('3x^2 + 2x + 2')

    def test___radd__(self):
        assert 1 + self.p == Polynomial('3x^2 + 2x + 2')

    def test___sub__(self):
        assert self.p3 - 7 == [-6, 2, -4, 0, 0, 7]

    def test___rsub__(self):
        assert 7 - self.p3 == [6, -2, 4, 0, 0, -7]

    def test___mul__(self):
        assert self.p * self.p2 == Polynomial('6x^3 + 7x^2 + 4x + 1')
        assert self.p * 2 == Polynomial('6x^2 + 4x + 2')

    def test___rmul__(self):
        assert 2 * self.p == Polynomial('6x^2 + 4x + 2')

    def test___neg__(self):
        assert -self.p == Polynomial('-3x^2 - 2x - 1')

    def test___pow__(self):
        assert self.p ** 2 == self.p * self.p

    def test_derivative(self):
        assert self.p3.derivative() == '35x^4 - 8x + 2'

    def test_integral(self):
        assert self.p.integral() == 'x^3+x^2+x'

    def test___repr__(self):
        s = repr(self.p)
        s = s.replace(' ', '')
        assert repr(self.p) == '3*x**2+2*x+1'

    def test__repr_latex_(self):
        s = self.p._repr_latex_()
        s = s.replace(' ', '')
        assert s == '${3x^2+2x+1}$'

    def test___str__(self):
        s = str(self.p)
        s = s.replace(' ', '')
        assert s == '3x^2+2x+1'

    def test___eq__(self):
        assert self.p == '3x^2+2x+1'
        assert self.p != '3*x^2+2*x+1' #TODO: test pass, but strangely ...

    def test___lt__(self):
        assert self.p2 < self.p
        assert not self.p < self.p
        assert not self.p < self.p2
        assert self.p < '3x^2+2*x+2'
        assert not self.p < [2, 3, 1]

    def test___cmp__(self):
        assert self.p == self.p
        assert self.p == [1, 2, 3]
        assert self.p > self.p2
        assert self.p3 == [1, 2, -4, 0, 0, 7]
        assert self.p3 == '7x^5 - 4x^2 + 2x + 1'


class TestPlist:
    def test_plist(self):
        pass  # tested above


class TestPeval:
    def test_peval(self):
        pass  # tested above


class TestIntegral:
    def test_integral(self):
        pass  # tested above


class TestDerivative:
    def test_derivative(self):
        pass  # tested above


class TestAdd:
    def test_add(self):
        assert tostring(add([1, 2, 3], [1, -2, 3])
                        ) == '6x^2 + 2'  # test addition


class TestSub:
    def test_sub(self):
        pass  # tested above


class TestMultConst:
    def test_mult_const(self):
        pass  # tested above


class TestMultiply:
    def test_multiply(self):
        assert tostring(multiply([1, 1], [-1, 1])
                        ) == 'x^2 - 1'  # test multiplication


class TestMultOne:
    def test_mult_one(self):
        pass  # tested above


class TestPower:
    def test_power(self):
        assert tostring(power([1, 1], 2)) == 'x^2 + 2x + 1'  # test power


class TestParseString:
    def test_parse_string(self):
        assert parse_string('+3x - 4x^2 + 7x^5-x+1') == [1, 2, -4, 0, 0, 7]


class TestTostring:
    def test_tostring(self):
        assert tostring([1, 2., 3]) == '3x^2 + 2.0x + 1'  # testing floats
        # can we handle - signs
        assert tostring([1, 2, -3]) == '-3x^2 + 2x + 1'
        assert tostring([1, -2, 3]) == '3x^2 - 2x + 1'
        # are we smart enough to exclude 0 terms?
        assert tostring([0, 1, 2]) == '2x^2 + x'
        # testing leading zero stripping
        assert tostring([0, 1, 2, 0]) == '2x^2 + x'
        assert tostring([0, 1]) == 'x'
        assert tostring([0, 1.0]) == 'x'  # testing whether 1.0 == 1: risky
