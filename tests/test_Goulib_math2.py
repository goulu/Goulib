from nose.tools import assert_equal, assert_true, assert_false, assert_almost_equal, assert_raises
from nose import SkipTest
from random import random
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
        for s in accsum(range(10)): pass
        assert_equal(s,45)

class TestTranspose:
    def test_transpose(self):
        v1=range(3)
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
        v1=range(3)
        v2=list(accsum(v1))
        assert_equal(dot(v1, v2),7)
        m1=[v1,v2,vecsub(v2,v1)]
        assert_equal(dot(m1,v1),[5,7,2])
        m2=transpose(m1)
        assert_equal(dot(m1,m2),[[5, 7, 2], [7, 10, 3], [2, 3, 1]])

class TestVecadd:
    def test_vecadd(self):
        v1=range(4)
        v2=list(accsum(v1))
        assert_equal(vecadd(v1,v2),[0,2,5,9])
        v1=v1[1:]
        assert_equal(vecadd(v1,v2),[1,3,6,6])
        assert_equal(vecadd(v1,v2,-1),[1,3,6,5])

class TestVecsub:
    def test_vecsub(self):
        v1=range(4)
        v2=tuple(accsum(v1))
        assert_equal(vecsub(v1,v2),[0,0,-1,-3])
        v1=v1[1:]
        assert_equal(vecsub(v1,v2),[1,1,0,-6])
        assert_equal(vecsub(v1,v2,-1),[1,1,0,-7])

class TestVecmul:
    def test_vecmul(self):
        v1=range(4)
        v2=list(accsum(v1))
        assert_equal(vecmul(v1,v2),[0,1,6,18])

class TestVecdiv:
    def test_vecdiv(self):
        v1=range(5)[1:]
        v2=list(accsum(v1))
        assert_equal(vecdiv(v1,v2),[1,2./3,1./2,2./5])

class TestVeccompare:
    def test_veccompare(self):
        v1=range(5)[1:]
        v2=list(accsum(v1))
        assert_equal(veccompare(v1,v2),[3,1,0])

class TestMean:
    def test_mean(self):
        r=[random() for _ in range(5000)]
        assert_almost_equal(mean(r),0.5,1) # significant difference is unlikely

class TestVariance:
    def test_variance(self):
        r=[random() for _ in range(5000)]
        assert_almost_equal(variance(r),0.082,2) # significant difference is unlikely

class TestStats:
    def test_stats(self):
        n=10000
        r=[float(_)/n for _ in range(n+1)]
        min,max,sum,sum2,avg,var=stats(r)
        assert_almost_equal(min,0.,1) # significant difference is unlikely
        assert_almost_equal(max,1.,1) # significant difference is unlikely
        assert_almost_equal(sum,(n+1)/2.,2) # significant difference is unlikely
        assert_almost_equal(sum2,(n+2)/3.,0) # significant difference is unlikely
        assert_almost_equal(avg,0.5,4) # significant difference is unlikely
        assert_almost_equal(var,1./12,4) # significant difference is unlikely

class TestFibonacci:
    def test_fibonacci(self):
        # assert_equal(expected, fibonacci())
        raise SkipTest # TODO: implement your test here

class TestFactorial:
    def test_factorial(self):
        # assert_equal(expected, factorial(num))
        raise SkipTest # TODO: implement your test here

class TestIsInteger:
    def test_is_integer(self):
        # assert_equal(expected, is_integer(x, epsilon))
        raise SkipTest # TODO: implement your test here

class TestIntOrFloat:
    def test_int_or_float(self):
        # assert_equal(expected, int_or_float(x, epsilon))
        raise SkipTest # TODO: implement your test here

class TestGetPrimes:
    def test_get_primes(self):
        # assert_equal(expected, get_primes(start, memoized))
        raise SkipTest # TODO: implement your test here

class TestDigitsFromNumFast:
    def test_digits_from_num_fast(self):
        assert_equal(digits_from_num_fast(1234),[1,2,3,4])
        
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
        raise SkipTest # TODO: implement your test here

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
        assert_equal(list(factorize(2014)),[(2, 1), (19, 1), (53, 1)])
        assert_equal(list(factorize(2048)),[(2,11)])
        
class TestDivisors:
    def test_divisors(self):
        assert_equal(list(divisors(2014)),[1, 53, 19, 1007, 2, 106, 38, 2014])

class TestProperDivisors:
    def test_proper_divisors(self):
        assert_equal(list(proper_divisors(2014)),[1, 53, 19, 1007, 2, 106, 38])

class TestGreatestCommonDivisor:
    def test_greatest_common_divisor(self):
        # assert_equal(expected, greatest_common_divisor(a, b))
        raise SkipTest # TODO: implement your test here

class TestLeastCommonMultiple:
    def test_least_common_multiple(self):
        # assert_equal(expected, least_common_multiple(a, b))
        raise SkipTest # TODO: implement your test here

class TestTriangle:
    def test_triangle(self):
        # assert_equal(expected, triangle(x))
        raise SkipTest # TODO: implement your test here

class TestIsTriangle:
    def test_is_triangle(self):
        # assert_equal(expected, is_triangle(x))
        raise SkipTest # TODO: implement your test here

class TestPentagonal:
    def test_pentagonal(self):
        # assert_equal(expected, pentagonal(n))
        raise SkipTest # TODO: implement your test here

class TestIsPentagonal:
    def test_is_pentagonal(self):
        # assert_equal(expected, is_pentagonal(n))
        raise SkipTest # TODO: implement your test here

class TestHexagonal:
    def test_hexagonal(self):
        # assert_equal(expected, hexagonal(n))
        raise SkipTest # TODO: implement your test here

class TestGetCardinalName:
    def test_get_cardinal_name(self):
        # assert_equal(expected, get_cardinal_name(num))
        raise SkipTest # TODO: implement your test here

class TestIsPerfect:
    def test_is_perfect(self):
        # assert_equal(expected, is_perfect(num))
        raise SkipTest # TODO: implement your test here

class TestIsPandigital:
    def test_is_pandigital(self):
        # assert_equal(expected, is_pandigital(digits, through))
        raise SkipTest # TODO: implement your test here

class TestSetsDist:
    def test_sets_dist(self):
        # assert_equal(expected, sets_dist(a, b))
        raise SkipTest # TODO: implement your test here

class TestSetsLevenshtein:
    def test_sets_levenshtein(self):
        # assert_equal(expected, sets_levenshtein(a, b))
        raise SkipTest # TODO: implement your test here

class TestLevenshtein:
    def test_levenshtein(self):
        # assert_equal(expected, levenshtein(seq1, seq2))
        raise SkipTest # TODO: implement your test here

class TestNcombinations:
    def test_ncombinations(self):
        # assert_equal(expected, ncombinations(n, k))
        raise SkipTest # TODO: implement your test here

class TestCombinationsWithReplacement:
    def test_combinations_with_replacement(self):
        # assert_equal(expected, combinations_with_replacement(iterable, r))
        raise SkipTest # TODO: implement your test here

class TestProportional:
    def test_proportional(self):
        assert_equal(proportional(12,[0,0,1,0]),[0,0,12,0])
        votes=[10,20,30,40]
        assert_equal(proportional(100, votes),votes)
        assert_equal(proportional(10, votes),[1,2,3,4])
        assert_equal(sum(proportional(37, votes)),37)
        assert_equal(proportional(37, votes),[4,7,11,15])

class TestLinearRepartition:
    def test_linear_repartition(self):
        
        ref=[0.01,0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19]
        res=linear_repartition(1,10)
        assert_equal(sum(res),1)
        assert_true(dist(res,ref)<1E-6)
        ref.reverse()
        res=linear_repartition(0,10)
        assert_almost_equal(sum(res),1)
        assert_true(dist(res,ref)<1E-6)
        
        ref=[0.02,0.06,0.1,0.14,0.18,0.18,0.14,0.1,0.06,0.02]
        res=linear_repartition(.5,10)
        assert_almost_equal(sum(res),1)
        assert_true(dist(res,ref)<1E-6)
        
        ref=[0.08,0.24,0.36,0.24,0.08]
        res=linear_repartition(.5,5) # center value is top of triangle
        assert_almost_equal(sum(res),1)
        assert_true(dist(res,ref)<1E-6)
        
class TestNorm2:
    def test_norm_2(self):
        # assert_equal(expected, norm_2(v))
        raise SkipTest # TODO: implement your test here

class TestNorm1:
    def test_norm_1(self):
        # assert_equal(expected, norm_1(v))
        raise SkipTest # TODO: implement your test here

class TestNormInf:
    def test_norm_inf(self):
        # assert_equal(expected, norm_inf(v))
        raise SkipTest # TODO: implement your test here

class TestNorm:
    def test_norm(self):
        # assert_equal(expected, norm(v, order))
        raise SkipTest # TODO: implement your test here

class TestDist:
    def test_dist(self):
        # assert_equal(expected, dist(a, b, norm))
        raise SkipTest # TODO: implement your test here

class TestSat:
    def test_sat(self):
        assert_equal(sat(3),3)
        assert_equal(sat(-2),0)
        assert_equal(sat(-3,-3),-3)
        assert_equal(sat(3,1,2),2)
        assert_equal(sat([-2,-1,0,1,2,3],-1,2),[-1,-1,0,1,2,2])

if __name__ == "__main__":
    import nose
    nose.runmodule()
