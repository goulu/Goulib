from nose.tools import assert_equal,assert_true
from nose import SkipTest

from Goulib.polynomial import *

class TestPolynomial:
    @classmethod
    def setup_class(self):
    
        self.p=Polynomial([1,2,3])
        self.p2=Polynomial([1,2])
        self.p3=Polynomial('+3*x -4x^2 +7x^5-x+1')
        self.p4=Polynomial('5x+x^2')
        
    def test___init__(self):
        pass #tested above
        
    def test___cmp__(self):
        assert_true(self.p==self.p)
        assert_true(self.p>self.p2)
        assert_equal(self.p3,[1,2,-4,0,0,7])
        assert_equal(self.p3,'7x^5 - 4x^2 + 2x + 1')
        
    def test___call__(self):
        assert_equal(self.p(0),1)
        assert_equal(self.p(1),6)
        assert_equal(self.p(2),17)
        
    def test___add__(self):
        assert_equal(self.p+self.p2,Polynomial('3x^2 + 4x + 2'))
        assert_equal(self.p+1,Polynomial('3x^2 + 2x + 2'))
        
    def test___radd__(self):
        assert_equal(1+self.p,Polynomial('3x^2 + 2x + 2'))
        
    def test___sub__(self):
        assert_equal(self.p3-7,[-6,2,-4,0,0,7])
                
    def test___rsub__(self):
        assert_equal(7-self.p3,[6,-2,4,0,0,-7])

    def test___mul__(self):
        assert_equal(self.p*self.p2,Polynomial('6x^3 + 7x^2 + 4x + 1'))
        assert_equal(self.p*2,Polynomial('6x^2 + 4x + 2'))
        
    def test___rmul__(self):
        assert_equal(2*self.p,Polynomial('6x^2 + 4x + 2'))
        
    def test___neg__(self):
        assert_equal(-self.p,Polynomial('-3x^2 - 2x - 1'))

    def test___pow__(self):
        assert_equal(self.p**2,self.p*self.p)
        
    def test_derivative(self):
        assert_equal(self.p3.derivative(),'35x^4 - 8x + 2')

    def test_integral(self):
        assert_equal(self.p.integral(),'x^3+x^2+x')

    def test___repr__(self):
        pass # tested in all comparizons above
    
    def test__repr_latex_(self):
        assert_equal(self.p._repr_latex_(), '$3x^2 + 2x + 1$')
    
    def test___str__(self):
        assert_equal(str(self.p), '3x^2 + 2x + 1')

class TestPlist:
    def test_plist(self):
        pass #tested above

class TestPeval:
    def test_peval(self):
        pass #tested above

class TestIntegral:
    def test_integral(self):
        pass #tested above

class TestDerivative:
    def test_derivative(self):
        pass #tested above

class TestAdd:
    def test_add(self):
        assert_equal(tostring(add([1,2,3],[1,-2,3])),'6x^2 + 2')  # test addition

class TestSub:
    def test_sub(self):
        pass #tested above

class TestMultConst:
    def test_mult_const(self):
        pass #tested above

class TestMultiply:
    def test_multiply(self):
        assert_equal(tostring(multiply([1,1],[-1,1])),'x^2 - 1') # test multiplication

class TestMultOne:
    def test_mult_one(self):
        pass #tested above

class TestPower:
    def test_power(self):
        assert_equal(tostring(power([1,1],2)),'x^2 + 2x + 1') # test power

class TestParseString:
    def test_parse_string(self):
        assert_equal(parse_string('+3x - 4x^2 + 7x^5-x+1'),[1, 2, -4, 0, 0, 7])

class TestTostring:
    def test_tostring(self):
        assert_equal(tostring([1,2.,3]),'3x^2 + 2.0x + 1') # testing floats
        assert_equal(tostring([1,2,-3]),'-3x^2 + 2x + 1') # can we handle - signs
        assert_equal(tostring([1,-2,3]),'3x^2 - 2x + 1')
        assert_equal(tostring([0,1,2]),'2x^2 + x')  # are we smart enough to exclude 0 terms?
        assert_equal(tostring([0,1,2,0]),'2x^2 + x') # testing leading zero stripping
        assert_equal(tostring([0,1]),'x')
        assert_equal(tostring([0,1.0]),'x') # testing whether 1.0 == 1: risky
                     
if __name__ == "__main__":
    import nose
    nose.runmodule()
    

    # Some cases using the polynomial objects:
    print Polynomial([1,2,3]) + Polynomial([1,2]) # add
    print Polynomial([1,2,3]) + 1                 # add
    print Polynomial([1,2,3])-1                   # sub
    print 1-Polynomial([1,2,3])                   # rsub
    print 1+Polynomial([1,2,3])                   # radd
    print Polynomial([1,2,3])*-1                  # mul
    print -1*Polynomial([1,2,3])                  # rmul
    print -Polynomial([1,2,3])                    # neg
    print ''
    # Work out Niklasson's raising and lowering operators:
    #  tests putting constants into the polynomial.
    for m in range(1,4):
        print 'P^a_%d = ' % m,\
              1 - Polynomial([1,-1])**m*Polynomial([1,m])
        print 'P^b_%d = ' % m,\
              Polynomial([0,1])**m*Polynomial([1+m,-m])
    print ''

    # Test the integral and derivatives
    print integral([])
    print integral([1])
    print integral([0,1])
    print derivative([0,0,0.5])
    p = Polynomial('x')
    ip = p.integral()
    dp = p.derivative()
    print ip,dp
    print ip(0,1) # integral of y=x from (0,1)
