from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *

from Goulib.math2 import *

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
        # assert_equal(expected, sieve(n))
        raise SkipTest

class TestPrimes:
    def test_primes(self):
        # assert_equal(expected, primes(n))
        raise SkipTest 

class TestPrimesGen:
    def test_primes_gen(self):
        from itertools import islice
        a=list(islice(primes_gen(),10))
        assert_equal(a,[2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
        a=list(islice(primes_gen(29),10))
        assert_equal(a,[29, 31, 37, 41, 43, 47, 53, 59, 61, 67])
        a=list(islice(primes_gen(67,29),10))
        assert_equal(a,reversed([29, 31, 37, 41, 43, 47, 53, 59, 61, 67]))

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
         

class TestDigitsFromNum:
    def test_digits_from_num(self):
        assert_equal(digits(1234),[1,2,3,4])
        assert_equal(digits(1234, rev=True),[4,3,2,1])
        assert_equal(digits(2014,2),[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0])

class TestNumFromDigits:
    def test_num_from_digits(self):
        assert_equal(num_from_digits('1234'),1234)
        assert_equal(num_from_digits('11111011110',2),2014)
        assert_equal(num_from_digits([1,2,3,4]),1234)
        assert_equal(num_from_digits([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],2),2014)

class TestNumberOfDigits:
    def test_number_of_digits(self):
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
        assert_false(is_prime(2013))
        assert_false(is_prime(20132013))
        assert_false(is_prime(201420132013))
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

class TestGreatestCommonDivisor:
    def test_greatest_common_divisor(self):
        assert_equal(greatest_common_divisor(54,24),6)

class TestLeastCommonMultiple:
    def test_least_common_multiple(self):
        assert_equal(least_common_multiple(4,6),12)

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

class TestSetsLevenshtein:
    def test_sets_levenshtein(self):
        a=set(list('hello'))
        b=set(list('world'))
        assert_equal(sets_levenshtein(a, b),5)

class TestLevenshtein:
    def test_levenshtein(self):
        assert_equal(levenshtein('hello','world'),4)

class TestBinomialCoefficient:
    def test_binomial_coefficient(self):
        # https://www.hackerrank.com/challenges/ncr
        assert_equal(binomial_coefficient(1,2),0)
        assert_equal(binomial_coefficient(2,1),2)
        assert_equal(binomial_coefficient(4,0),1)
        assert_equal(binomial_coefficient(5,2),10)
        assert_equal(binomial_coefficient(10,3),120)
        assert_equal(binomial_coefficient(87,28) % 142857,141525)
        assert_equal(
            binomial_coefficient(100000,4000),
            binomial_coefficient(100000,96000) #same because 100000-96000=4000
        )
        
    @raises(OverflowError)
    def test_binomial_coefficient_overflow(self):
        assert_equal(binomial_coefficient(961173600,386223045)%142857,0)

class TestCombinationsWithReplacement:
    def test_combinations_with_replacement(self):
        assert_equal(combinations_with_replacement('ABC', 2),
            ['AA','AB','AC','BB','BC','CC'])

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

class TestLogFactorial:
    def test_log_factorial(self):
        assert_equal(log_factorial(100),363.73937555556349014408) 

class TestLogBinomialCoefficient:
    def test_log_binomial_coefficient(self):
        assert_equal(log_binomial_coefficient(87,28),log(49848969000742658237160))
        
class TestEulerPhi:
    def test_euler_phi(self):
        assert_equal(euler_phi(8849513),8843520)

class TestRecurrence:
    def test_recurrence(self):
        # assert_equal(expected, recurrence(factors, values, max))
        raise SkipTest # 

class TestCatalan:
    def test_catalan(self):
        # assert_equal(expected, catalan())
        raise SkipTest # 

class TestLucasLehmer:
    def test_lucas_lehmer(self):
        # assert_equal(expected, lucas_lehmer(p))
        raise SkipTest # 

class TestEulerPhiOverN:
    def test_euler_phi_over_n(self):
        # assert_equal(expected, euler_phi_over_n(n))
        raise SkipTest # 

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
        # assert_equal(expected, fibonacci_gen(max))
        raise SkipTest # 

class TestCatalanGen:
    def test_catalan_gen(self):
        # assert_equal(expected, catalan_gen())
        raise SkipTest # 

class TestTriples:
    def test_triples(self):
        # assert_equal(expected, triples())
        raise SkipTest # 

class TestPrimitiveTriples:
    def test_primitive_triples(self):
        # assert_equal(expected, primitive_triples(sort_xy))
        raise SkipTest # 

class TestPolygonal:
    def test_polygonal(self):
        # assert_equal(expected, polygonal(s, n))
        raise SkipTest # 

class TestSquare:
    def test_square(self):
        # assert_equal(expected, square(n))
        raise SkipTest # 

class TestIsSquare:
    def test_is_square(self):
        # assert_equal(expected, is_square(n))
        raise SkipTest # 

class TestIsHexagonal:
    def test_is_hexagonal(self):
        # assert_equal(expected, is_hexagonal(n))
        raise SkipTest # 

class TestHeptagonal:
    def test_heptagonal(self):
        # assert_equal(expected, heptagonal(n))
        raise SkipTest # 

class TestIsHeptagonal:
    def test_is_heptagonal(self):
        # assert_equal(expected, is_heptagonal(n))
        raise SkipTest # 

class TestOctagonal:
    def test_octagonal(self):
        # assert_equal(expected, octagonal(n))
        raise SkipTest # 

class TestIsOctagonal:
    def test_is_octagonal(self):
        # assert_equal(expected, is_octagonal(n))
        raise SkipTest # 

class TestPartition:
    def test_partition(self):
        # assert_equal(expected, partition(n))
        raise SkipTest # 

class TestChakravala:
    def test_chakravala(self):
        # assert_equal(expected, chakravala(n))
        raise SkipTest # 

class TestBouncy:
    def test_bouncy(self):
        # assert_equal(expected, bouncy(n))
        raise SkipTest # 

class TestSosDigits:
    def test_sos_digits(self):
        # assert_equal(expected, sos_digits(n))
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

if __name__ == "__main__":
    runmodule()
