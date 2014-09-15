from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *

from Goulib.math2 import *

class TestQuad:
    def test_quad(self):
        import cmath
        assert_equal(quad(1,3,2),(-1,-2)) 
        assert_raises(ValueError,quad,1,2,3) #complex results
        assert_equal(sum(quad(1,2,3,complex=True)),-2) #complex results

class TestEqual:
    def test_equal(self):
        a=1E6
        d=0.99e-6
        assert_true(equal(a, a+d))
        assert_false(equal(a, a+2*d))

class TestLcm:
    def test_lcm(self):
        assert_equal(lcm(101, -3),-303)

class TestAccsum:
    def test_accsum(self):
        for s in accsum(list(range(10))): pass
        assert_equal(s,45)

class TestTranspose:
    def test_transpose(self):
        v1=list(range(3))
        v2=list(accsum(v1))
        m1=[v1,v2,vecsub(v2,v1)]
        assert_equal(transpose(m1),[(0, 0, 0), (1, 1, 0), (2, 3, 1)])
        
class TestMaximum:
    def test_maximum(self):
        m=[(1,2,3),(1,-2,0),(4,0,0)]
        assert_equal(maximum(m),[4,2,3])

class TestMinimum:
    def test_minimum(self):
        m=[(1,2,3),(1,-2,0),(4,0,0)]
        assert_equal(minimum(m),[1,-2,0])
        
class TestDot:
    def test_dot(self):
        v1=list(range(3))
        v2=list(accsum(v1))
        assert_equal(dot(v1, v2),7)
        m1=[v1,v2,vecsub(v2,v1)]
        assert_equal(dot(m1,v1),[5,7,2])
        m2=transpose(m1)
        assert_equal(dot(m1,m2),[[5, 7, 2], [7, 10, 3], [2, 3, 1]])

class TestVecadd:
    def test_vecadd(self):
        v1=list(range(4))
        v2=list(accsum(v1))
        assert_equal(vecadd(v1,v2),[0,2,5,9])
        v1=v1[1:]
        assert_equal(vecadd(v1,v2),[1,3,6,6])
        assert_equal(vecadd(v1,v2,-1),[1,3,6,5])

class TestVecsub:
    def test_vecsub(self):
        v1=list(range(4))
        v2=tuple(accsum(v1))
        assert_equal(vecsub(v1,v2),[0,0,-1,-3])
        v1=v1[1:]
        assert_equal(vecsub(v1,v2),[1,1,0,-6])
        assert_equal(vecsub(v1,v2,-1),[1,1,0,-7])

class TestVecmul:
    def test_vecmul(self):
        v1=list(range(4))
        v2=list(accsum(v1))
        assert_equal(vecmul(v1,v2),[0,1,6,18])

class TestVecdiv:
    def test_vecdiv(self):
        v1=list(range(5))[1:]
        v2=list(accsum(v1))
        assert_equal(vecdiv(v1,v2),[1,2./3,1./2,2./5])

class TestVeccompare:
    def test_veccompare(self):
        v1=list(range(5))[1:]
        v2=list(accsum(v1))
        assert_equal(veccompare(v1,v2),[3,1,0])
        
class TestFibonacci:
    def test_fibonacci(self):
        from Goulib.itertools2 import take
        assert_equal(take(10,fibonacci()),[0,1,1,2,3,5,8,13,21,34])

class TestFactorial:
    def test_factorial(self):
        # assert_equal(expected, factorial(num))
        raise SkipTest 

class TestIsInteger:
    def test_is_integer(self):
        assert_true(is_integer(1+1e-6, 1e-6))
        assert_false(is_integer(1+2e-6, 1e-6))

class TestIntOrFloat:
    def test_int_or_float(self):
        assert_equal(type(int_or_float(1+1e-6, 1e-6)),int)
        assert_equal(type(int_or_float(1+2e-6, 1e-6)),float)
                     
class TestGetPrimes:
    def test_get_primes(self):
        from itertools import islice
        a=[p for p in islice(get_primes(),10)]
        assert_equal(a,[2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
        a=[p for p in islice(get_primes(29,True),10)]
        assert_equal(a,[29, 31, 37, 41, 43, 47, 53, 59, 61, 67])
        
class TestStrBase:
    def test_str_base(self):
        assert_equal(str_base(2014,2),"11111011110")
        assert_equal(str_base(65535,16),"ffff")

class TestDigitsFromNum:
    def test_digits_from_num(self):
        assert_equal(digits_from_num(1234),[1,2,3,4])
        assert_equal(digits_from_num(2014,2),[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0])

class TestNumFromDigits:
    def test_num_from_digits(self):
        assert_equal(num_from_digits([1,2,3,4]),1234)
        assert_equal(num_from_digits([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],2),2014)
        
class TestNumberOfDigits:
    def test_number_of_digits(self):
        assert_equal(number_of_digits(1234),4)
        assert_equal(number_of_digits(2014,2),11)
        assert_equal(number_of_digits(65535,16),4)

class TestIsPalindromic:
    def test_is_palindromic(self):
        # assert_equal(expected, is_palindromic(num, base))
        raise SkipTest 

class TestIsPrime:
    def test_is_prime(self):
        assert_false(is_prime(2013))
        assert_false(is_prime(20132013))
        assert_false(is_prime(201420132013))
        assert_true(is_prime(201420142013))
        
class TestPrimeFactors:
    def test_prime_factors(self):
        assert_equal(prime_factors(2014),[2, 19, 53])
        assert_equal(prime_factors(2048),[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

class TestFactorize:
    def test_factorize(self):
        assert_equal(factorize(2014),[(2, 1), (19, 1), (53, 1)])
        assert_equal(factorize(2048),[(2,11)])
        
class TestDivisors:
    def test_divisors(self):
        assert_equal(divisors(2014),[1, 53, 19, 1007, 2, 106, 38, 2014])

class TestProperDivisors:
    def test_proper_divisors(self):
        assert_equal(proper_divisors(2014),[1, 53, 19, 1007, 2, 106, 38])

class TestGreatestCommonDivisor:
    def test_greatest_common_divisor(self):
        # assert_equal(expected, greatest_common_divisor(a, b))
        raise SkipTest 

class TestLeastCommonMultiple:
    def test_least_common_multiple(self):
        # assert_equal(expected, least_common_multiple(a, b))
        raise SkipTest 

class TestTriangle:
    def test_triangle(self):
        # assert_equal(expected, triangle(x))
        raise SkipTest 

class TestIsTriangle:
    def test_is_triangle(self):
        # assert_equal(expected, is_triangle(x))
        raise SkipTest 

class TestPentagonal:
    def test_pentagonal(self):
        # assert_equal(expected, pentagonal(n))
        raise SkipTest 

class TestIsPentagonal:
    def test_is_pentagonal(self):
        # assert_equal(expected, is_pentagonal(n))
        raise SkipTest 

class TestHexagonal:
    def test_hexagonal(self):
        # assert_equal(expected, hexagonal(n))
        raise SkipTest 

class TestGetCardinalName:
    def test_get_cardinal_name(self):
        # assert_equal(expected, get_cardinal_name(num))
        raise SkipTest 

class TestIsPerfect:
    def test_is_perfect(self):
        # assert_equal(expected, is_perfect(num))
        raise SkipTest 

class TestIsPandigital:
    def test_is_pandigital(self):
        # assert_equal(expected, is_pandigital(digits, through))
        raise SkipTest 

class TestSetsDist:
    def test_sets_dist(self):
        # assert_equal(expected, sets_dist(a, b))
        raise SkipTest 

class TestSetsLevenshtein:
    def test_sets_levenshtein(self):
        # assert_equal(expected, sets_levenshtein(a, b))
        raise SkipTest 

class TestLevenshtein:
    def test_levenshtein(self):
        # assert_equal(expected, levenshtein(seq1, seq2))
        raise SkipTest 

class TestNcombinations:
    def test_ncombinations(self):
        # assert_equal(expected, ncombinations(n, k))
        raise SkipTest 
    
class TestBinomialCoefficient:
    def test_binomial_coefficient(self):
        # https://www.hackerrank.com/challenges/ncr
        assert_equal(binomial_coefficient(2,1),2)
        assert_equal(binomial_coefficient(4,0),1)
        assert_equal(binomial_coefficient(5,2),10)
        assert_equal(binomial_coefficient(10,3),120)
        assert_equal(binomial_coefficient(87,28) % 142857,141525)
        # assert_equal(binomial_coefficient(961173600,386223045) % 142857,0) # much too large for now

class TestCombinationsWithReplacement:
    def test_combinations_with_replacement(self):
        # assert_equal(expected, combinations_with_replacement(iterable, r))
        raise SkipTest 

class TestProportional:
    def test_proportional(self):
        assert_equal(proportional(12,[0,0,1,0]),[0,0,12,0])
        votes=[10,20,30,40]
        assert_equal(proportional(100, votes),votes)
        assert_equal(proportional(10, votes),[1,2,3,4])
        assert_equal(sum(proportional(37, votes)),37)
        assert_equal(proportional(37, votes),[4,7,11,15])

class TestTriangularRepartition:
    def test_triangular_repartition(self):
        
        ref=[0.01,0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19]
        res=triangular_repartition(1,10)
        assert_equal(sum(res),1)
        assert_true(dist(res,ref)<1E-6)
        ref.reverse()
        res=triangular_repartition(0,10)
        assert_equal(sum(res),1)
        assert_true(dist(res,ref)<1E-6)
        
        ref=[0.02,0.06,0.1,0.14,0.18,0.18,0.14,0.1,0.06,0.02]
        res=triangular_repartition(.5,10)
        assert_equal(sum(res),1)
        assert_true(dist(res,ref)<1E-6)
        
        ref=[0.08,0.24,0.36,0.24,0.08]
        res=triangular_repartition(.5,5) # center value is top of triangle
        assert_equal(sum(res),1)
        assert_true(dist(res,ref)<1E-6)

class TestRectangularRepartition:
    def test_rectangular_repartition(self):
        ref=[.5,.125,.125,.125,.125]
        res=rectangular_repartition(0,5,.5)
        assert_equal(sum(res),1)
        assert_true(dist(res,ref)<1E-6)
        
        ref=[0.3125,0.3125,.125,.125,.125]
        res=rectangular_repartition(.2,5,.5)
        assert_equal(sum(res),1)
        assert_true(dist(res,ref)<1E-6)
        ref.reverse()
        res=rectangular_repartition(.8,5,.5)
        assert_equal(sum(res),1)
        assert_true(dist(res,ref)<1E-6)
        
        ref=[0.1,0.1675,0.3325,.1,.1,.1,.1]
        res=rectangular_repartition(.325,7,.4)
        assert_equal(sum(res),1)
        assert_true(dist(res,ref)<1E-6)
        
class TestNorm2:
    def test_norm_2(self):
        assert_equal(norm_2([-3,4]),5)

class TestNorm1:
    def test_norm_1(self):
        assert_equal(norm_1([-3,4]),7)

class TestNormInf:
    def test_norm_inf(self):
        assert_equal(norm_inf([-3,4]),4)

class TestNorm:
    def test_norm(self):
        assert_equal(norm([-3,4],2),5)
        assert_equal(norm([-3,4],1),7)

class TestDist:
    def test_dist(self):
        # assert_equal(expected, dist(a, b, norm))
        raise SkipTest 

class TestSat:
    def test_sat(self):
        assert_equal(sat(3),3)
        assert_equal(sat(-2),0)
        assert_equal(sat(-3,-3),-3)
        assert_equal(sat(3,1,2),2)
        assert_equal(sat([-2,-1,0,1,2,3],-1,2),[-1,-1,0,1,2,2])

class TestVecneg:
    def test_vecneg(self):
        assert_equal(vecneg([-2,-1,0,1,2,3]),[2,1,0,-1,-2,-3])

class TestAngle:
    def test_angle(self):
        assert_equal(angle((1,0),(0,1)),pi/2)
        assert_equal(angle((1,0),(-1,0)),pi)
        assert_equal(angle((1,1),(0,1),unit=False),pi/4)
        assert_equal(angle(vecunit((2,1)),vecunit((1,-2))),pi/2)

class TestVecunit:
    def test_vecunit(self):
        v=vecunit((-3,4,5))
        assert_equal(norm(v),1)

class TestSinOverX:
    def test_sin_over_x(self):
        assert_equal(sin_over_x(1),sin(1))
        assert_equal(sin_over_x(0),1)
        assert_equal(sin_over_x(1e-9),1)

class TestSlerp:
    def test_slerp(self):
        u=vecunit((1,1,1))
        v=vecunit((1,1,-1))
        assert_equal(slerp(u,v,0),u)
        assert_equal(slerp(u,v,1),v)
        s=slerp(u,v,0.5)
        assert_equal(s,vecunit((1,1,0)))


if __name__ == "__main__":
    runmodule()
