#!/usr/bin/env python
# coding: utf8
from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *

from Goulib.math2 import *
from Goulib.itertools2 import take, index
import six

class TestSign:
    def test_sign(self):
        assert_equal(sign(0.0001),1)
        assert_equal(sign(-0.0001),-1)
        assert_equal(sign(-0.0000),0)

class TestCmp:
    def test_cmp(self):
        assert_equal(cmp(0.0002, 0.0001),1)
        assert_equal(cmp(0.0001, 0.0002),-1)
        assert_equal(cmp(0.0000, 0.0000),0)

class TestMul(unittest.TestCase):
    def test_mul(self):
        assert_equal(mul(range(1,10)),362880)

class TestRint(unittest.TestCase):
    def test_rint(self):
        if six.PY2 : #https://docs.python.org/2.7/library/functions.html#round
            assert_equal(rint(0.5),1,places=None)
            assert_equal(rint(-0.5),-1,places=None)
        else: # https://docs.python.org/3.4/library/functions.html#round
            assert_equal(rint(0.5),0,places=None)
            assert_equal(rint(-0.5),-0,places=None)

        assert_equal(rint(0.50001),1,places=None)
        assert_equal(rint(-0.50001),-1,places=None)


class TestQuad:
    def test_quad(self):
        assert_equal(quad(1,3,2),(-1,-2))
        assert_raises(ValueError,quad,1,2,3) #complex results
        assert_equal(sum(quad(1,2,3,allow_complex=True)),-2) #complex results

class TestEqual:
    def test_isclose(self):
        a=1E6
        d=0.99e-3
        assert_true(isclose(a, a+d))
        assert_false(isclose(a, a+2*d))

    def test_equal(self):
        # assert_equal(expected, equal(a, b, epsilon))
        raise SkipTest

class TestLcm:
    def test_lcm(self):
        assert_equal(lcm(101, -3),-303)
        assert_equal(lcm(4,6),12)

class TestGcd:
    def test_gcd(self):
        assert_equal(gcd(54,24),6)
        assert_equal(gcd(68, 14, 9, 36, 126),1)
        assert_equal(gcd(7, 14, 35, 7000),7)
        assert_equal(gcd(1548),1548)

class TestCoprime:
    def test_coprime(self):
        assert_true(coprime(68, 14, 9, 36, 126))
        assert_false(coprime(7, 14, 35, 7000))

class TestAccsum:
    def test_accsum(self):
        s=list(accsum(range(10)))
        assert_equal(s[-1],45)

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
        assert_equal(vecmul(v1,2),[0,2,4,6])
        assert_equal(vecmul(2,v1),[0,2,4,6])

class TestVecdiv:
    def test_vecdiv(self):
        v1=list(range(5))[1:]
        v2=list(accsum(v1))
        assert_equal(vecdiv(v1,v2),[1,2./3,1./2,2./5])
        assert_equal(vecdiv(v1,2),[1./2,2./2,3./2,4./2])

class TestVeccompare:
    def test_veccompare(self):
        v1=list(range(5))
        v2=list(accsum(v1))
        v2[-1]=2 #force to test ai>bi
        assert_equal(veccompare(v1,v2),[2,2,1])

class TestFibonacci:
    def test_fibonacci(self):
        f=[fibonacci(i) for i in range(10)]
        assert_equal(f,[0,1,1,2,3,5,8,13,21,34])
        assert_equal(f,itertools2.take(10,fibonacci_gen()))

        #http://controlfd.com/2016/07/05/using-floats-in-python.html
        assert_equal(fibonacci(78),8944394323791464)
        
        #mod 1000000007 has the effect of using int32 only
        assert_equal(fibonacci(int(1E19),1000000007),647754067)

class TestIsInteger:
    def test_is_integer(self):
        assert_true(is_integer(1+1e-6, 1e-6))
        assert_false(is_integer(1+2e-6, 1e-6))

class TestIntOrFloat:
    def test_int_or_float(self):
        assert_equal(type(int_or_float(1+1e-6, 1e-6)),int)
        assert_equal(type(int_or_float(1+2e-6, 1e-6)),float)

class TestSieve:
    def test_sieve(self):
        last=sieve(10000)[-1] #more than _sieve for coverage
        assert_equal(last,9973)

class TestPrimes:
    def test_primes(self):
        last=primes(1001)[999] #more than _primes for coverage
        assert_equal(last,7919)

class TestPrimesGen:
    def test_primes_gen(self):
        from itertools import islice
        a=list(islice(primes_gen(),10))
        assert_equal(a,[2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
        a=list(islice(primes_gen(29),10))
        assert_equal(a,[29, 31, 37, 41, 43, 47, 53, 59, 61, 67])
        a=list(islice(primes_gen(67,29),10))
        assert_equal(a,reversed([29, 31, 37, 41, 43, 47, 53, 59, 61, 67]))
        a=list(primes_gen(901, 1000))
        assert_equal(a,[907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997])

class TestStrBase:
    def test_str_base(self):
        assert_equal(str_base(2014),"2014")
        assert_equal(str_base(-2014),"-2014")
        assert_equal(str_base(-0),"0")
        assert_equal(str_base(2014,2),"11111011110")
        assert_equal(str_base(65535,16),"ffff")

        assert_raises(ValueError,str_base,0,1)

        # http://www.drgoulu.com/2011/09/25/comment-comptent-les-extraterrestres
        shadok=['GA','BU','ZO','MEU']
        assert_raises(ValueError,str_base,0,10,shadok)
        assert_equal(str_base(41,4,shadok),"ZOZOBU")
        assert_equal(str_base(1681,4,shadok),"BUZOZOBUGABU")

class TestDigitsGen:
    def test_digits_gen(self):
        pass #used below

class TestDigits:
    def test_digits(self):
        assert_equal(digits(1234),[1,2,3,4])
        assert_equal(digits(1234, rev=True),[4,3,2,1])
        assert_equal(digits(2014,2),[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0])

class TestDigsum:
    def test_digsum(self):
        assert_equal(digsum(1234567890),45)
        assert_equal(digsum(255,2),8) # sum of ones in binary rep
        assert_equal(digsum(255,16),30) # $FF in hex
        assert_equal(digsum(1234567890,f=2),sum_of_squares(9))
        assert_equal(digsum(548834,f=6),548834) #narcissic number
        assert_equal(digsum(3435,f=lambda x:x**x),3435) #Munchausen number

class TestIntegerExponent:
    def test_integer_exponent(self):
        assert_equal(integer_exponent(1000),3)
        assert_equal(integer_exponent(1024,2),10)
        assert_equal(integer_exponent(binomial(1000,373),2),6) #http://thales.math.uqam.ca/~rowland/packages/BinomialCoefficients/HTMLLinks/index_3.html

class TestPowerTower:
    def test_power_tower(self):
        assert_equal(power_tower([3,2,2,2]),43046721)

class TestCarries:
    def test_carries(self):
        assert_equal(carries(127, 123),1)
        assert_equal(carries(127, 173),2)
        assert_equal(carries(1, 999),3)
        assert_equal(carries(999, 1),3)
        assert_equal(carries(127, 127,2),7)

class TestNumFromDigits:
    def test_num_from_digits(self):
        assert_equal(num_from_digits('1234'),1234)
        assert_equal(num_from_digits('11111011110',2),2014)
        assert_equal(num_from_digits([1,2,3,4]),1234)
        assert_equal(num_from_digits([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],2),2014)

class TestNumberOfDigits:
    def test_number_of_digits(self):
        assert_equal(number_of_digits(0),1)
        assert_equal(number_of_digits(-1),1)
        assert_equal(number_of_digits(1234),4)
        assert_equal(number_of_digits(2014,2),11)
        assert_equal(number_of_digits(65535,16),4)


class TestIsPalindromic:
    def test_is_palindromic(self):
        assert_true(is_palindromic(4352534))
        assert_true(is_palindromic(17,2))

class TestIsLychrel:
    def test_is_lychrel(self):
        assert_true(is_lychrel(196))
        assert_true(is_lychrel(4994))

class TestIsPrime:
    def test_is_prime(self):
        assert_false(is_prime(0))
        assert_false(is_prime(1))
        assert_true(is_prime(2))
        
        #https://oeis.org/A014233
        pseudoprimes=[2047, 1373653, 25326001, 3215031751, 2152302898747, 3474749660383, 341550071728321, 341550071728321, 3825123056546413051, 3825123056546413051, 3825123056546413051, 318665857834031151167461, 3317044064679887385961981]
        for pp in pseudoprimes:
            assert_false(is_prime(pp))
        
        assert_true(is_prime(201420142013))
        assert_true(is_prime(4547337172376300111955330758342147474062293202868155909489))
        assert_false(is_prime(4547337172376300111955330758342147474062293202868155909393))
        assert_equal(
            [x for x in range(901, 1000) if is_prime(x)],
            [907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]
        )
        assert_true(is_prime(643808006803554439230129854961492699151386107534013432918073439524138264842370630061369715394739134090922937332590384720397133335969549256322620979036686633213903952966175107096769180017646161851573147596390153))
        assert_false(is_prime(743808006803554439230129854961492699151386107534013432918073439524138264842370630061369715394739134090922937332590384720397133335969549256322620979036686633213903952966175107096769180017646161851573147596390153))

class TestPrimeFactors:
    def test_prime_factors(self):
        assert_equal(prime_factors(2014),[2, 19, 53])
        assert_equal(prime_factors(2048),[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

class TestFactorize:
    def test_factorize(self):
        d=list(factorize(1))
        assert_equal(factorize(1),[(1,1)])
        d=list(factorize(2014))
        assert_equal(d,[(2, 1), (19, 1), (53, 1)])
        d=list(factorize(2048))
        assert_equal(factorize(2048),[(2,11)])

class TestDivisors:
    def test_divisors(self):
        d=list(divisors(1))
        assert_equal(d,[1])
        d=list(divisors(2014))
        assert_equal(d,[1, 53, 19, 1007, 2, 106, 38, 2014])

class TestProperDivisors:
    def test_proper_divisors(self):
        d=list(proper_divisors(2014))
        assert_equal(d,[1, 53, 19, 1007, 2, 106, 38])

class TestTriangle:
    def test_triangle(self):
        assert_equal(triangle(10),55)

class TestIsTriangle:
    def test_is_triangle(self):
        assert_true(is_triangle(55))
        assert_false(is_triangle(54))

class TestPentagonal:
    def test_pentagonal(self):
        assert_equal(pentagonal(10),145)

class TestIsPentagonal:
    def test_is_pentagonal(self):
        assert_true(is_pentagonal(145))
        assert_false(is_pentagonal(146))

class TestHexagonal:
    def test_hexagonal(self):
        assert_equal(hexagonal(10),190)

class TestGetCardinalName:
    def test_get_cardinal_name(self):
       assert_equal(get_cardinal_name(123456),
            'one hundred and twenty-three thousand four hundred and fifty-six'
        )
       assert_equal(get_cardinal_name(1234567890),
            'one billion two hundred and thirty-four million five hundred and sixty-seven thousand eight hundred and ninety'
        )

class TestIsPerfect:
    def test_is_perfect(self):
        d=list(factorize(496))
        d=list(divisors(496))
        assert_equal(is_perfect(496),0) #perfect
        assert_equal(is_perfect(54),1) #abundant
        assert_equal(is_perfect(2),-1) #deficient

        assert_equal(is_perfect(2305843008139952128),0) #Millenium 4, page 326

class TestIsPandigital:
    def test_is_pandigital(self):
        # https://en.wikipedia.org/wiki/Pandigital_number
        assert_true(is_pandigital(9786530421))
        assert_true(is_pandigital(1223334444555567890))
        assert_true(is_pandigital(10,2))
        assert_true(is_pandigital(0x1023456789ABCDEF,16))


class TestSetsDist:
    def test_sets_dist(self):
        a=set(list('hello'))
        b=set(list('world'))
        assert_equal(sets_dist(a, b),3.1622776601683795)

class TestHamming:
    def test_hamming(self):
        a="10011100"
        b="00011010"
        assert_equal(hamming(a, b),3)

class TestSetsLevenshtein:
    def test_sets_levenshtein(self):
        a=set(list('hello'))
        b=set(list('world'))
        assert_equal(sets_levenshtein(a, b),5)

class TestLevenshtein:
    def test_levenshtein(self):
        assert_equal(levenshtein('hello','world'),4)

class TestBinomial:
    def test_binomial(self):
        # https://www.hackerrank.com/challenges/ncr
        assert_equal(binomial(1,2),0)
        assert_equal(binomial(2,1),2)
        assert_equal(binomial(4,0),1)
        assert_equal(binomial(5,2),10)
        assert_equal(binomial(10,3),120)
        assert_equal(binomial(87,28) % 142857,141525)
        assert_equal(
            binomial(100000,4000),
            binomial(100000,96000) #same because 100000-96000=4000
        )

    @raises(OverflowError)
    def test_binomial_overflow(self):
        assert_equal(binomial(961173600,386223045)%142857,0)

class TestFaulhaber:
    def test_faulhaber(self):
        def sumpow(n,p):
            return sum((x**p for x in range(n+1)))
        assert_equal(faulhaber(100,0),100)
        assert_equal(faulhaber(100,1),triangular(100))
        assert_equal(faulhaber(100,2),sum_of_squares(100))
        assert_equal(faulhaber(100,3),sum_of_cubes(100))
        assert_equal(faulhaber(100,4),sumpow(100,4))

class TestBinomialExponent:
    def test_binomial_exponent(self):
        assert_equal(binomial_exponent(88,50,3),3) # https://www.math.upenn.edu/~wilf/website/dm36.pdf

        #http://thales.math.uqam.ca/~rowland/packages/BinomialCoefficients/HTMLLinks/index_3.html
        assert_equal(binomial_exponent(1000,373,2),6)
        for b in range(2,11):
            for n in range(1,20):
                for k in range(1,n):
                    assert_equal(binomial_exponent(n,k,b), integer_exponent(binomial(n,k),b))


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
        pass #tested somewhere else

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
        assert_equal(angle((1,0),(0,1)),math.pi/2)
        assert_equal(angle((1,0),(-1,0)),math.pi)
        assert_equal(angle((1,1),(0,1),unit=False),math.pi/4)
        assert_equal(angle(vecunit((2,1)),vecunit((1,-2))),math.pi/2)

class TestVecunit:
    def test_vecunit(self):
        v=vecunit((-3,4,5))
        assert_equal(norm(v),1)

class TestSinOverX:
    def test_sin_over_x(self):
        assert_equal(sin_over_x(1),math.sin(1))
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

class TestLogFactorial:
    def test_log_factorial(self):
        assert_equal(log_factorial(100),363.73937555556349014408)

class TestLogBinomialCoefficient:
    def test_log_binomial(self):
        assert_equal(log_binomial(87,28),math.log(49848969000742658237160))

class Moebius:
    def test_moebius(self):
        assert_equal(moebius(3),-1)

class Omega:
    def test_omega(self):
        assert_equal(omega(3),0)
        assert_equal(omega(4),1)
        assert_equal(omega(6),2)

class TestEulerPhi:
    def test_euler_phi(self):
        assert_equal(euler_phi(8849513),8843520)

class TestRecurrence:
    def test_recurrence(self):
        # assert_equal(expected, recurrence(factors, values, max))
        raise SkipTest #

class TestLucasLehmer:
    def test_lucas_lehmer(self):
        assert_false(lucas_lehmer(1548)) # trivial case
        assert_true(lucas_lehmer(11213)) # found on Illiac 2, 1963)
        assert_false(lucas_lehmer(239))

class TestReverse:
    def test_reverse(self):
        # assert_equal(expected, reverse(i))
        raise SkipTest #

class TestIsPermutation:
    def test_is_permutation(self):
        # assert_equal(expected, is_permutation(num1, num2, base))
        raise SkipTest #

class TestLychrelSeq:
    def test_lychrel_seq(self):
        # assert_equal(expected, lychrel_seq(n))
        raise SkipTest #

class TestLychrelCount:
    def test_lychrel_count(self):
        # assert_equal(expected, lychrel_count(n, limit))
        raise SkipTest #

class TestIsqrt:
    def test_isqrt(self):
        # assert_equal(expected, isqrt(n))
        raise SkipTest #

class TestAbundance:
    def test_abundance(self):
        # assert_equal(expected, abundance(n))
        raise SkipTest #

class TestFactorial:
    def test_factorial(self):
        # assert_equal(expected, factorial())
        raise SkipTest #

class TestCeildiv:
    def test_ceildiv(self):
        # assert_equal(expected, ceildiv(a, b))
        raise SkipTest #

class TestFibonacciGen:
    def test_fibonacci_gen(self):
        #also tested in test_oeis
        
        # https://projecteuler.net/problem=2
        from itertools import takewhile

        def problem2(n):
            """Find the sum of all the even-valued terms in the Fibonacci < 4 million."""
            even_fibonacci = (x for x in fibonacci_gen() if x % 2 ==0)
            l=list(takewhile(lambda x: x < n, even_fibonacci))
            return sum(l)

        assert_equal(problem2(10),10)
        assert_equal(problem2(100),44)
        assert_equal(problem2(4E6),4613732)

class TestCatalanGen:
    def test_catalan_gen(self):
        assert_equal(index(20,catalan_gen()),6564120420) # https://oeis.org/A000108
    
class TestCatalan:
    def test_catalan(self):
        assert_equal(catalan(20),6564120420) # https://oeis.org/A000108

class TestPrimitiveTriples:
    def test_primitive_triples(self):
        key=lambda x:(x[2],x[1])
        for t in take(10000,itertools2.ensure_sorted(primitive_triples(),key)):
            assert_true(is_pythagorean_triple(*t))

class TestTriples:
    def test_triples(self):
        key=lambda x:(x[2],x[1])
        for t in take(10000,itertools2.ensure_sorted(triples(),key)):
            assert_true(is_pythagorean_triple(*t))

class TestPolygonal:
    def test_polygonal(self):
        pass #tested in test_oeis

class TestSquare:
    def test_square(self):
        pass #tested in test_oeis

class TestIsSquare:
    def test_is_square(self):
        pass #tested in test_oeis

class TestIsHexagonal:
    def test_is_hexagonal(self):
        pass #tested in test_oeis

class TestHeptagonal:
    def test_heptagonal(self):
        pass #tested in test_oeis

class TestIsHeptagonal:
    def test_is_heptagonal(self):
        pass #tested in test_oeis

class TestOctagonal:
    def test_octagonal(self):
        pass #tested in test_oeis

class TestIsOctagonal:
    def test_is_octagonal(self):
        pass #tested in test_oeis

class TestPartition:
    def test_partition(self):
        pass #tested in test_oeis

class TestChakravala:
    def test_chakravala(self):
        x,y=chakravala(61)
        assert_true(x==1766319049 and y==226153980)
        #https://en.wikipedia.org/wiki/Chakravala_method

class TestBouncy:
    def test_bouncy(self):
        # assert_equal(expected, bouncy(n))
        raise SkipTest #

class TestIsHappy:
    def test_is_happy(self):
        # assert_equal(expected, is_happy(n))
        raise SkipTest #

class TestNumberOfDivisors:
    def test_number_of_divisors(self):
        # assert_equal(expected, number_of_divisors(n))
        raise SkipTest #

class TestFactorialGen:
    def test_factorial_gen(self):
        # assert_equal(expected, factorial_gen())
        raise SkipTest #

class TestEuclidGen:
    def test_euclid_gen(self):
        # assert_equal(expected, euclid_gen())
        raise SkipTest

class TestModPow:
    def test_mod_pow(self):
        assert_equal( mod_pow(2,10,100),24)
        assert_equal( mod_pow(4,13,497),445) #https://fr.wikipedia.org/wiki/Exponentiation_modulaire
        assert_equal( mod_pow(2,13739062,13739063),2933187) #http://www.math.utah.edu/~carlson/hsp2004/PythonShortCourse.pdf

class TestEgcd:
    def test_egcd(self):
        pass #tested below

class TestModInv:
    def test_mod_inv(self):
        assert_equal(mod_inv(3,11),4)

class TestModDiv:
    def test_mod_div(self):
        assert_equal( mod_div(3,16,53),30)
        assert_equal( mod_div(3,16,53),30)
        assert_equal( mod_div(5,5,12),25)

class TestModFact:
    def test_mod_fact(self):
        assert_equal( mod_fact(10,71),61)
        assert_equal( mod_fact(11,71),32)

class TestChineseRemainder:
    def test_chinese_remainder(self):
        assert_equal( chinese_remainder([3,5,7],[2,3,2]),23)
        assert_equal( chinese_remainder([3,4,5],[2,3,1]),11) #http://en.wikipedia.org/wiki/Chinese_remainder_theorem

class TestModBinomial:
    def test_mod_binomial(self):
        assert_equal( mod_binomial(456, 51, 30),28) #http://math.stackexchange.com/questions/95491/n-choose-k-bmod-m-using-chinese-remainder-theorem

        assert_equal( mod_binomial(1000, 729, 19),13) #http://thales.math.uqam.ca/~rowland/packages/BinomialCoefficients/HTMLLinks/index_4.html

        res=binomial(16, 5) % 9
        #http://math.stackexchange.com/questions/222637/binomial-coefficient-modulo-prime-power
        assert_equal(  mod_binomial(16, 5, 9),res)

        m=142857

        assert_equal(  mod_binomial(5,2,m),binomial(5,2))
        assert_equal(  mod_binomial(10,3,m),binomial(10,3))

        assert_equal(  mod_binomial(27,3,27),binomial(27, 3) % 27) #==9
        assert_equal(  mod_binomial(27,3,m),binomial(27,3)%m) #== 2925

        return #tests below are too large for now

        assert_equal(  mod_binomial(961173600,386223045,m),0)
        assert_equal(  mod_binomial(938977945,153121024,m),47619)
        assert_equal(  mod_binomial(906601285,527203335,m),0)
        assert_equal(  mod_binomial(993051461,841624879,m),104247)
        
class TestDeBrujin:
    def test_de_brujin(self):
        assert_equal(de_bruijn('1234',3),'1112113114122123124132133134142143144222322423323424324433343444')

class TestXgcd:
    def test_xgcd(self):
        # assert_equal(expected, xgcd(a, b))
        raise SkipTest

class TestIsclose:
    def test_isclose(self):
        # assert_equal(expected, isclose(a, b, rel_tol, abs_tol))
        raise SkipTest

class TestOmega:
    def test_omega(self):
        # assert_equal(expected, omega(n))
        raise SkipTest

class TestBigomega:
    def test_bigomega(self):
        # assert_equal(expected, bigomega(n))
        raise SkipTest

class TestMoebius:
    def test_moebius(self):
        # assert_equal(expected, moebius(n))
        raise SkipTest

class TestPrimeKtuple:
    def test_prime_ktuple(self):
        # assert_equal(expected, prime_ktuple(constellation))
        raise SkipTest

class TestTwinPrimes:
    def test_twin_primes(self):
        # assert_equal(expected, twin_primes())
        raise SkipTest

class TestCousinPrimes:
    def test_cousin_primes(self):
        # assert_equal(expected, cousin_primes())
        raise SkipTest

class TestSexyPrimes:
    def test_sexy_primes(self):
        # assert_equal(expected, sexy_primes())
        raise SkipTest

class TestSexyPrimeTriplets:
    def test_sexy_prime_triplets(self):
        # assert_equal(expected, sexy_prime_triplets())
        raise SkipTest

class TestSexyPrimeQuadruplets:
    def test_sexy_prime_quadruplets(self):
        # assert_equal(expected, sexy_prime_quadruplets())
        raise SkipTest

class TestLogBinomial:
    def test_log_binomial(self):
        # assert_equal(expected, log_binomial(n, k))
        raise SkipTest

class TestIlog:
    def test_ilog(self):
        # assert_equal(expected, ilog(a, b, upper_bound))
        raise SkipTest

class TestBabyStepGiantStep:
    def test_baby_step_giant_step(self):
        #70 = 2**x mod 131
        y,a,n=70,2,131
        x=baby_step_giant_step(y,a,n)
        assert_true(y==a**x % n)

class TestIsNumber:
    def test_is_number(self):
        # assert_equal(expected, is_number(x))
        raise SkipTest # TODO: implement your test here

class TestCoprimesGen:
    def test_coprimes_gen(self):
        res=','.join(('%d/%d'%(n,d) for n,d in coprimes_gen(10)))
        assert_equal(res,'0/1,1/10,1/9,1/8,1/7,1/6,1/5,2/9,1/4,2/7,3/10,1/3,3/8,2/5,3/7,4/9,1/2,5/9,4/7,3/5,5/8,2/3,7/10,5/7,3/4,7/9,4/5,5/6,6/7,7/8,8/9,9/10')

class TestTetrahedral:
    def test_tetrahedral(self):
        # assert_equal(expected, tetrahedral(n))
        raise SkipTest # TODO: implement your test here

class TestSumOfSquares:
    def test_sum_of_squares(self):
        # assert_equal(expected, sum_of_squares(n))
        raise SkipTest # TODO: implement your test here

class TestSumOfCubes:
    def test_sum_of_cubes(self):
        # assert_equal(expected, sum_of_cubes(n))
        raise SkipTest # TODO: implement your test here

class TestBernouilliGen:
    def test_bernouilli_gen(self):
        # assert_equal(expected, bernouilli_gen(init))
        raise SkipTest # TODO: implement your test here

class TestBernouilli:
    def test_bernouilli(self):
        # assert_equal(expected, bernouilli(n, init))
        raise SkipTest # TODO: implement your test here

class TestDeBruijn:
    def test_de_bruijn(self):
        # assert_equal(expected, de_bruijn(k, n))
        raise SkipTest # TODO: implement your test here

class TestPascalGen:
    def test_pascal_gen(self):
        # assert_equal(expected, pascal_gen())
        raise SkipTest # TODO: implement your test here

class TestIsPythagoreanTriple:
    def test_is_pythagorean_triple(self):
        # assert_equal(expected, is_pythagorean_triple(a, b, c))
        raise SkipTest # TODO: implement your test here

class TestFormat:
    def test_format(self):
        # assert_equal(expected, format(x, decimals))
        raise SkipTest # TODO: implement your test here
    
class TestMultiply:
    def test_multiply(self):
        from random import getrandbits
        for bits in [100,1000,10000]:
            a=getrandbits(bits)
            b=getrandbits(bits)
            assert_equal(multiply(a,b),a*b)
            

class TestSqrt:
    def test_sqrt(self):
        # assert_equal(expected, sqrt(n))
        raise SkipTest # TODO: implement your test here

class TestModMatmul:
    def test_mod_matmul(self):
        # assert_equal(expected, mod_matmul(A, B, mod))
        raise SkipTest # TODO: implement your test here

class TestModMatpow:
    def test_mod_matpow(self):
        a=[[1, 2],[1, 0]]
        b=matrix_power(a,50)
        assert_equal(
            b, 
            [[750599937895083, 750599937895082], 
             [375299968947541, 375299968947542]]
        )

class TestZeros:
    def test_zeros(self):
        # assert_equal(expected, zeros(shape))
        raise SkipTest # TODO: implement your test here

class TestDiag:
    def test_diag(self):
        # assert_equal(expected, diag(v))
        raise SkipTest # TODO: implement your test here

if __name__ == "__main__":
    runmodule()
