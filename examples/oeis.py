#!/usr/bin/env python
# coding: utf8

"""
OEIS sequences
(OEIS is Neil Sloane's On-Line Encyclopedia of Integer Sequences at https://oeis.org/)
"""

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright (c) 2015 Philippe Guglielmetti"
__license__ = "LGPL"
__credits__ = ["https://oeis.org/"]

__docformat__ = 'restructuredtext'
__version__ = '$Id$'
__revision__ = '$Revision$'

import six
import logging
import operator
import math
from six.moves import map, reduce, filter, zip, zip_longest

from itertools import count, repeat
from Goulib.itertools2 import *
from Goulib.math2 import *

from Goulib import decorators, tests

from Goulib.container import Sequence

A000004 = Sequence(repeat(0), lambda _: 0, lambda x: x ==
                   0, desc='The zero sequence')

A000007 = Sequence(None, lambda n: 0**n, lambda x: x in (0, 1),
                   desc='The characteristic function of 0: a(n) = 0^n.')

A001477 = Sequence(count(0), lambda n: n, lambda x: x >=
                   0, desc='The non-negative integers.')
A000027 = Sequence(count(1), lambda n: n, lambda x: x >
                   0, desc='The positive integers.')
A005408 = Sequence(count(1, 2), lambda n: 2*n+1, lambda x: x %
                   2 == 1, desc='The odd numbers: a(n) = 2n+1.')
A005843 = Sequence(count(0, 2), lambda n: 2*n, lambda x: x %
                   2 == 0, desc='The even numbers: a(n) = 2n ')

A001057 = Sequence(1, lambda n: -n//2+1 if n % 2 else n//2, lambda _: True,
                   desc="Canonical enumeration of integers: interleaved positive and negative integers with zero prepended."
                   )

A008587 = Sequence(count(0, 5), lambda n: 5*n, lambda n: n %
                   5 == 0, 'Multiples of 5')
A008589 = Sequence(count(0, 7), lambda n: 7*n, lambda n: n %
                   7 == 0, 'Multiples of 7')

A000079 = Sequence(recurrence([2], [1]), lambda n: 2 **
                   n, desc='Powers of 2: a(n) = 2^n.')
A001146 = Sequence(None, lambda n: 2**2**n, desc='2^(2^n)')
A051179 = A001146-1
A000215 = A001146+1
A000215.desc = 'Fermat numbers'


A000244 = Sequence(None, lambda n: 3**n, desc='Powers of 3: a(n) = 3^n.')
A067500 = A000244.filter(lambda n: digsum(
    n) in A000244, "Powers of 3 with digit sum also a power of 3.")

A006521 = Sequence(1, None, lambda n: (2**n + 1) %
                   n == 0, "Numbers n such that n divides 2^n + 1. ")

A000332 = Sequence(0, lambda n: binomial(
    n, 4), desc="Binomial coefficient binomial(n,4) = n*(n-1)*(n-2)*(n-3)/24.")

# repunits in base

A000225, A003462, A002450, A003463, A003464, A023000, A023001, A002452, A002275 = (
    Sequence(
        repunit_gen(base),
        lambda n: repunit(n, base),
        desc="a(n) = (%d^n - 1)/%d." % (base, base-1)
    )
    for base in range(2, 11)
)
A002275.desc = "Repunits: (10^n - 1)/9. Often denoted by R_n."
A000225.desc = "a(n) = 2^n - 1. (Sometimes called Mersenne numbers, although that name is usually reserved for A001348."

A016123, A016125, A091030, A135519, A135518, A131865 = (
    Sequence(
        drop(1, repunit_gen(base)),
        lambda n: repunit(n+1, base),
        desc="a(n) = (%d^n - 1)/%d." % (base, base-1)
    )
    for base in range(11, 17)
)

# repdigits

A024023 = Sequence(
    repunit_gen(3, 2),
    lambda n: repunit(n+1, 3),
    desc="a(n) = 3^n - 1."
)

A048328 = A003462 | A024023
A048328.desc = "Numbers that are repdigits in base 3."

# polynoms

A016957 = Sequence(None, lambda n: 6*n+4, desc="a(n) = 6*n + 4.")

# polygonal numbers

A000217 = Sequence(None, triangle, is_triangle, 'triangle numbers')

A000290 = Sequence(None, lambda n: n*n, lambda n: is_square(n), 'squares')

A000326 = Sequence(None, pentagonal, is_pentagonal, 'pentagonal numbers')

A001318 = A001057.apply(pentagonal, is_pentagonal,
                        desc="Generalized pentagonal numbers: n*(3*n-1)/2, n=0, +- 1, +- 2, +- 3, ....")

A000384 = Sequence(None, hexagonal, is_hexagonal)

A000566 = Sequence(None, heptagonal, is_heptagonal)

A000567 = Sequence(None, octagonal, is_octagonal)

A001106 = Sequence(None, lambda n: polygonal(9, n))

A001107 = Sequence(None, lambda n: polygonal(10, n))

A051682 = Sequence(None, lambda n: polygonal(11, n))

A051624 = Sequence(None, lambda n: polygonal(12, n))

A051865 = Sequence(None, lambda n: polygonal(13, n))

A051866 = Sequence(None, lambda n: polygonal(14, n))

A051867 = Sequence(None, lambda n: polygonal(15, n))

A051868 = Sequence(None, lambda n: polygonal(16, n))

A051869 = Sequence(None, lambda n: polygonal(17, n))

A051870 = Sequence(None, lambda n: polygonal(18, n))

A051871 = Sequence(None, lambda n: polygonal(19, n))

A051872 = Sequence(None, lambda n: polygonal(20, n))

A051873 = Sequence(None, lambda n: polygonal(21, n))

A051874 = Sequence(None, lambda n: polygonal(22, n))

A051875 = Sequence(None, lambda n: polygonal(23, n))

A051876 = Sequence(None, lambda n: polygonal(24, n))

A167149 = Sequence(0, lambda n: polygonal(10000, n), 'myriagonal')

A001110 = A000217.filter(
    is_square, 'Square triangular numbers: numbers that are both triangular and square')
A001110.iterf = recurrence((35, -35, 1), (0, 1, 36))
# http://www.johndcook.com/blog/2015/08/21/computing-square-triangular-numbers/

A001109 = A001110.apply(isqrt, lambda n: n*n in A001110,
                        desc='a(n)^2 is a triangular number')
# pyramidal numbers

A003401 = Sequence(
    1, None,
    lambda n: str_base(totient(n), 2).count('1') == 1,
    desc='Values of n for which a regular polygon with n sides can be constructed with ruler and compass'
)

A004169 = Sequence(
    1, None,
    lambda n: str_base(totient(n), 2).count('1') != 1,
    desc='Values of n for which a regular polygon with n sides cannot be constructed with ruler and compass'
)

A000292 = Sequence(None, tetrahedral,
                   desc='Tetrahedral (or triangular pyramidal) numbers')

A000330 = Sequence(None, sum_of_squares, desc='Square pyramidal numbers')

A000537 = Sequence(None, sum_of_cubes,
                   desc='Sum of first n cubes; or n-th triangular number squared')


A027641 = Sequence(None, lambda n: bernouilli(
    n, -1).numerator, 'Numerator of Bernoulli number B_n.')
A027642 = Sequence(None, lambda n: bernouilli(
    n).denominator, 'Denominators of Bernoulli numbers')
A164555 = Sequence(None, lambda n: bernouilli(n, 1).numerator,
                   'Numerators of the "original" Bernoulli numbers')


def cullen(n): return n*2**n+1


A002064 = Sequence(None, cullen, desc='Cullen numbers')

# divisors
# https://oeis.org/wiki/Index_entries_for_number_of_divisors

A000005 = Sequence(1, number_of_divisors)
A000005.desc = 'd(n) (also called tau(n) or sigma_0(n)), the number of divisors of n.'

A002182 = Sequence(record_index(A000005, count(1)),
                   desc='Highly composite numbers, definition (1): where d(n), the number of divisors of n (A000005), increases to a record.'
                   )

A000203 = Sequence(1, lambda n: sum(divisors(n)),
                   desc='sigma(n), the sum of the divisors of n. Also called sigma_1(n).'
                   )

A033880 = Sequence(1, abundance)

A005101 = Sequence(1, None, lambda x: abundance(x) > 0,
                   desc='Abundant numbers (sum of divisors of n exceeds 2n).'
                   )

A002093 = Sequence(record_index(A000203, count(1)),
                   desc='Highly abundant numbers: numbers n such that sigma(n) > sigma(m) for all m < n.'
                   )

# primes & co

A000040 = Sequence(primes_gen, None, is_prime, 'The prime numbers')

A008578 = Sequence(
    primes_gen(1),
    None,
    lambda n: is_prime(n, oneisprime=True),
    'Prime numbers at the beginning of the 20th century (today 1 is no longer regarded as a prime).'
)

A065091 = Sequence(primes_gen(3), None, lambda x: x !=
                   2 and is_prime(x), 'The odd prime numbers')
A001248 = A000040.apply(
    lambda n: n*n, lambda n: is_prime(isqrt(n)), desc='Square of primes')
A030078 = A000040.apply(
    lambda n: n*n*n, lambda n: is_prime(icbrt(n)), desc='Cubes of primes')
A030514 = A000040.apply(
    lambda n: n**4, lambda n: is_prime(isqrt(isqrt(n))), desc='4th powers of primes.')

A045699 = A001248.product(A030078, sum).unique()
A045699.desc = "Numbers of the form p^2 + q^3, p,q prime."
A134657 = A045699.product(A030514, sum).unique()
A134657.desc = "Numbers of the form p^2 + q^3 + r^4 with p, q and r primes."

a318530 = A001248.product(A030078, sum)

A000961 = Sequence(1, None, lambda n: len(list(factorize(n))) == 1,
                   desc='Powers of primes. Alternatively, 1 and the prime powers (p^k, p prime, k >= 1).'
                   )

A000043 = A000040.filter(
    lucas_lehmer, 'Mersenne exponents: primes p such that 2^p - 1 is prime.')

A001348 = A000040.apply(
    lambda p: A000079[p]-1, desc='Mersenne numbers: 2^p - 1, where p is prime.')

A000668 = A000043.apply(
    lambda p: A000079[p]-1, desc='Mersenne primes (of form 2^p - 1 where p is a prime).')

A005384 = A000040.filter(lambda p:is_prime(2*p+1),'Sophie Germain primes p: 2p+1 is also prime. ')

A000396 = A000043.apply(lambda p: A000079[p-1]*(A000079[p] - 1),
                        containf=lambda x: is_perfect(x) == 0,
                        desc='Perfect numbers n: n is equal to the sum of the proper divisors of n.'
                        )


def exp_sequences(a, b, c, desc_s1=None, desc_s2=None, desc_s3=None, start=0):
    def _gen():
        p = b**start
        for _ in decorators.itimeout(count(), 10):
            yield a*p+c
            p = p*b

    s1 = Sequence(_gen, lambda n: a*b**n+c,
                  desc=desc_s1 or "a(n)=%d*%d^n%+d" % (a, b, c))
    s2 = s1.filter(
        is_prime, desc=desc_s2 or "Primes of the form %d*%d^n%+d" % (a, b, c))
    s3 = s2.apply(lambda n: ilog((n-c)//a, b),
                  desc=desc_s3 or "Numbers n such that %d*%d^n%+d is prime" % (a, b, c))
    return s1, s2, s3


A033484 = exp_sequences(3, 2, -2)[0]
A153893, A007505, A002235 = exp_sequences(
    3, 2, -1, desc_s2='Thabit primes of form 3*2^n -1.')

A046865 = exp_sequences(4, 5, -1)[2]
A079906 = exp_sequences(5, 6, -1)[2]
A046866 = exp_sequences(6, 7, -1, start=1)[2]
A086224, A050523, A001771 = exp_sequences(7, 2, -1)
A005541 = exp_sequences(8, 3, -1)[2]
A056725 = exp_sequences(9, 10, -1)[2]
A046867 = exp_sequences(10, 11, -1)[2]
A079907 = exp_sequences(11, 12, -1)[2]


def pow10m3():
    p, n = 0, 1
    while True:
        if is_prime(n-3):
            yield p
        p = p+1
        n = n*10


A089675 = Sequence(pow10m3, None, lambda n: is_prime(10**n-3))
A089675.desc = "Numbers n such that 9*R_n - 2 is a prime number, where R_n = 11...1 is the repunit (A002275) of length n.\
Also numbers n such that 10^n - 3 is prime"

A002385 = A000040.filter(is_palindromic)
A002385.desc = "Palindromic primes: prime numbers whose decimal expansion is a palindrome."

A007500 = A000040.filter(lambda x: is_prime(reverse(x)))
A007500.desc = "Primes whose reversal in base 10 is also prime"

A006567 = A000040.filter(lambda x: not is_palindromic(x)
                         and is_prime(reverse(x)))
A006567.desc = "Emirps (primes whose reversal is a different prime). "

# see https://blog.plover.com/math/dd.html


def anagrams(n, base=10):
    d = list(sorted(digits(n, base, rev=True)))
    for p in unique(itertools.permutations(d)):
        yield num_from_digits(p, base)


def anagram_gen(factor, base=10, start=0, inbase=False):
    if start == 0:
        yield 0  # cheat
    step = [1, 1, 9, 9, 3, 9, 9, 3, 3, 9][factor] if base == 10 else 1
    for d in count(1):  # number of digits
        start = base**d+(step-1)
        end = math.ceil(base/factor)*base**d
        for n in range(rint(start), rint(end), step):
            a = factor*n
            for b in anagrams(n, base):
                if b > a:
                    break
                if a == b:
                    if inbase:
                        yield int_base(n, base)
                    else:
                        yield n


A023086, A023087, A023088, A023089, A023090, A023091, A023092, A023093 = [
    Sequence(anagram_gen(f), None, lambda x:is_anagram(x, f*x),
             "Numbers n such that n and %d*n are anagrams." % f)
    for f in range(2, 10)
]


def first_anagram(f):
    return Sequence(f+1,
                    lambda n: first(anagram_gen(f, base=n, start=1)),
                    desc="a(n) is least k such that k and %dk are anagrams in base n (written in base 10)." % f
                    )


A023094, A023095, A023096, A023097, A023098, A023099, A023100, A023101, A023102 = [
    first_anagram(f) for f in range(2, 11)
]

A023058 = Sequence(anagram_gen(2, base=3, start=1, inbase=True),
                   desc="Numbers k such that k and 2k are anagrams of each other in base 3 (k is written here in base 3)"
                   )

A023059 = Sequence(anagram_gen(2, base=4, start=1, inbase=True),
                   desc="Numbers k such that k and 2k are anagrams of each other in base 4 (k is written here in base 4)"
                   )

# decimal expansions

A003592 = Sequence(1, None, lambda n: is_multiple(
    n, {2, 5}), desc="Numbers of the form 2^i*5^j with i, j >= 0.")
A051626 = Sequence(1, lambda n: rational_form(
    1, n)[-1], desc="Length of the period of decimal representation of 1/n, or 0 if 1/n terminates.")
A036275 = Sequence(1, lambda n: rational_cycle(
    1, n), desc="The periodic part of the decimal expansion of 1/n. Any initial 0's are to be placed at end of cycle.")

A006883 = A000040.filter(
    lambda n: n == 2 or rational_form(1, n)[-1] == n-1,
    desc='Long period primes: the decimal expansion of 1/p has period p-1.')
A004042 = A006883.apply(
    lambda n: rational_cycle(1, n),
    desc='Periods of reciprocals of A006883, starting with first nonzero digit.')

A000010 = Sequence(1, euler_phi)

# strangely this one has a leading 0...
A002088 = Sequence(0, euler_phi).accumulate()

A005728 = A002088+1
A005728.desc = 'Number of fractions in Farey series of order n.'

A090748 = A000043.apply(
    lambda n: n-1, desc='Numbers n such that 2^(n+1) - 1 is prime.')

A006862 = Sequence(
    euclid_gen,
    # lambda n:mul(primes(n))+1, # TODO: make it faster
    desc="Euclid numbers: 1 + product of the first n primes."
)

A002110 = A006862-1
A002110.desc = "Primorial numbers (first definition): product of first n primes"

A057588 = (A002110-1).filter(bool)  # remove leading 0 ...
A057588.desc = "Kummer numbers: -1 + product of first n consecutive primes."

A005234 = A000040.filter(
    # TODO: find a simple way to reuse A006862 or euclid_gen
    lambda n: is_prime(mul(sieve(n+1))+1),
    desc='Primorial primes: primes p such that 1 + product of primes up to p is prime'
)

A034386 = Sequence(
    0,
    lambda n: mul(sieve(n+1, oneisprime=True)),
    desc="Primorial numbers (second definition): n# = product of primes <= n"
)


# TODO: understand why A000720 creates a hudge bad side effect on many other serquences
A000720 = Sequence(
    1,
    lambda n: len(sieve(n+1, oneisprime=True))-1,
    lambda n: True,  # all integers are in this sequence.
    desc="pi(n), the number of primes <= n. Sometimes called PrimePi(n)"
)

A018239 = A006862.filter(
    is_prime,
    desc='Primorial primes: form product of first k primes and add 1, then reject unless prime.'
)

A007504 = A000040.accumulate()
A001223 = A000040.pairwise(operator.sub)

A077800 = Sequence(flatten(twin_primes()))

A001097 = A077800.unique()

A001359 = Sequence(itemgetter(twin_primes(), 0), desc="Lesser of twin primes.")

A006512 = Sequence(itemgetter(twin_primes(), 1),
                   desc="Greater of twin primes.")

A037074 = Sequence(map(mul, twin_primes()),
                   desc="Numbers that are the product of a pair of twin primes")


def count_10_exp(iterable):
    """generates number of iterable up to 10^n."""
    l = 10
    c = 0
    for n in iterable:
        if n > l:
            yield c
            l = 10*l
        c += 1


A007508 = Sequence(count_10_exp(A006512),
                   desc="Number of twin prime pairs below 10^n.")

A007510 = A000040 % A001097
A007510.desc = "Single (or isolated or non-twin) primes: Primes p such that neither p-2 nor p+2 is prime"

A023200 = Sequence(itemgetter(cousin_primes(), 0),
                   desc="Lesser of cousin primes.")
A046132 = Sequence(itemgetter(cousin_primes(), 1),
                   desc="Greater of cousin primes")

A023201 = Sequence(itemgetter(sexy_primes(), 0))
A023201.desc = "Sexy Primes : Numbers n such that n and n + 6 are both prime (sexy primes)"
A046117 = Sequence(itemgetter(sexy_primes(), 1))
A046117.desc = "Values of p+6 such that p and p+6 are both prime (sexy primes)"

A046118 = Sequence(itemgetter(sexy_prime_triplets(), 0))
A046119 = Sequence(itemgetter(sexy_prime_triplets(), 1))
A046120 = Sequence(itemgetter(sexy_prime_triplets(), 2))

A023271 = Sequence(itemgetter(sexy_prime_quadruplets(), 0))
A046122 = Sequence(itemgetter(sexy_prime_quadruplets(), 1))
A046123 = Sequence(itemgetter(sexy_prime_quadruplets(), 2))
A046124 = Sequence(itemgetter(sexy_prime_quadruplets(), 3))

A031924 = Sequence(itemgetter(prime_ktuple((0, -2, -4, 6)), 0),
                   desc="Lower prime of a difference of 6 between consecutive primes.")

A046117 = Sequence(itemgetter(prime_ktuple((0, 6)), 1),
                   desc="Primes p such that p-6 is also prime.")

A022004 = Sequence(itemgetter(prime_ktuple((0, 2, 6)), 0),
                   desc="Initial members of prime triples (p, p+2, p+6).")
A073648 = Sequence(itemgetter(prime_ktuple((0, 2, 6)), 1),
                   desc="Middle members of prime triples (p, p+2, p+6).")
A098412 = Sequence(itemgetter(prime_ktuple((0, 2, 6)), 2),
                   desc="Greatest members p of prime triples (p, p+2, p+6).")

A022005 = Sequence(itemgetter(prime_ktuple((0, 4, 6)), 0),
                   desc="Initial members of prime triples (p, p+4, p+6).")
A073649 = Sequence(itemgetter(prime_ktuple((0, 4, 6)), 1),
                   desc="Middle members of prime triples (p, p+4, p+6).")
A098413 = Sequence(itemgetter(prime_ktuple((0, 4, 6)), 2),
                   desc="Greatest members p of prime triples (p, p+4, p+6).")

A098414 = A073648 | A073649
A098413.desc = "Middle members q of prime triples (p,q,r) with p<q<r=p+6."
A098415 = A098412 | A098413
A098415.desc = "Greatest members r of prime triples (p,q,r) with p<q<r=p+6."

A007529 = A098415-6
A007529.desc = "Prime triples: n; n+2 or n+4; n+6 all prime. "

A098416 = (A007529+A098415)/4
A098416.desc = "(A007529(n) + A098415(n)) / 4."


def is_squarefree(n):
    return all(q == 1 for p, q in factorize(n))


# Squarefree numbers (or square-free numbers): numbers that are not divisible by a square greater than 1.
A005117 = Sequence(1, None, is_squarefree)


def is_product_of_2_primes(n):
    f = list(prime_factors(n))
    return len(f) == 2 and f[0] != f[1]


A006881 = Sequence(1, None, is_product_of_2_primes,
                   "Numbers that are the product of two distinct primes.")


def is_powerful(n):
    """if a prime p divides n then p^2 must also divide n
    (also called squareful, square full, square-full or 2-full numbers).
    """
    for f in unique(prime_factors(n)):
        if n % (f*f):
            return False
    return True


A001694 = Sequence(1, None, is_powerful, "powerful numbers")

# these 2 implementations have pretty much the same performance
A030513 = A030078 | A006881  # Numbers with 4 divisors
# A030513=Sequence(None,None,lambda n:len(list(divisors(n)))==4)
A030513.desc = "Numbers with 4 divisors"

A035533 = Sequence(count_10_exp(A030513))
A035533.desc = "Number of numbers up to 10^n with exactly 4 divisors"

A000196 = Sequence(0, isqrt, lambda _: True,
                   "    Integer part of square root of n. Or, number of positive squares <= n. Or, n appears 2n+1 times")
A000006 = A000040.apply(
    isqrt, desc="Integer part of square root of n-th prime.")


A001221 = Sequence(1, omega)
A001222 = Sequence(1, bigomega)

A001358 = Sequence(1, None, lambda n: bigomega(n) == 2,
                   "Semiprimes (or biprimes): products of two primes.")
A100959 = Sequence(1, None, lambda n: bigomega(n) != 2, "Non-semiprimes.")

A020639 = Sequence(
    1, lpf, desc="Lpf(n): least prime dividing n (when n > 1); a(1) = 1.")
A006530 = Sequence(
    1, gpf, desc="Gpf(n): greatest prime dividing n, for n >= 2; a(1)=1. ")

A008683 = Sequence(
    1, moebius, desc="Möbius (or Moebius) function mu(n). mu(1) = 1; mu(n) = (-1)^k if n is the product of k different primes; otherwise mu(n) = 0.")


def is_A055932(n):
    """:return: True if prime divisors of n are consecutive primes"""
    count = 0
    lastf = 1
    for f in prime_divisors(n):
        if f == n:
            break  # 1 and 2 are in A055932
        if f != nextprime(lastf):
            return False
        lastf = f
        count += 1
    return n < 3 or count > 1 or lastf == 2


A055932 = Sequence(1, None, is_A055932,
                   "Numbers where all prime divisors are consecutive primes starting at 2.")
# primitive roots


def has_primitive_root(n):
    if n == 1:
        return True  # to match A033948, but why ?
    try:
        six.next(primitive_root_gen(n))
        return True
    except StopIteration:
        return False


A033948 = Sequence(1, containf=has_primitive_root,
                   desc='numbers that have a primitive_root'
                   )

A001918 = A000040.apply(lambda n: first(primitive_root_gen(n)),
                        desc="Least positive primitive root of n-th prime. )"
                        )


A001122 = A000040.filter(lambda n: 2 == first(primitive_root_gen(n)),
                         desc="Primes with primitive root 2."
                         # we know  2 is first in primitive_root_gen
                         )


def is_in(n, gen):
    for x in gen:
        if x == n:
            return True
        if x > n:
            break
    return False


A001913 = A000040.filter(
    lambda n: n == 7 or is_in(10, primitive_root_gen(n)),
    desc="Primes with primitive root 10."
    # but why is 7 in A001913 ???
)

A002371 = A000040.apply(lambda n: rational_form(1, n)[-1])
A002371.desc = "Period of decimal expansion of 1/(n-th prime) (0 by convention for the primes 2 and 5). "

A003147 = A000040.filter(
    lambda n: any((g*g) % n == (g + 1) % n for g in primitive_root_gen(n)),
    desc="Primes with a fibonacci primitive root"
)

# Fibonacci & co

A000045 = Sequence(fibonacci_gen, fibonacci, is_fibonacci,
                   desc="Fibonacci numbers: F(n) = F(n-1) + F(n-2) with F(0) = 0 and F(1) = 1")

A082115 = Sequence(fibonacci_gen(mod=3), desc='Fibonacci numbers modulo 3')
A003893 = Sequence(fibonacci_gen(mod=10), desc='Fibonacci numbers modulo 10')

A001175 = Sequence(1, pisano_period, desc='Pisano period')

A060305 = A000040.apply(
    pisano_period, desc='Period of Fibonacci numbers mod prime(n).')

A134816 = Sequence(recurrence([0, 1, 1], [1, 1, 1]),
                   desc="Padovan's spiral numbers.")
A000931 = Sequence(recurrence([0, 1, 1], [
                   1, 0, 0]), desc="Padovan sequence: a(n) = a(n-2) + a(n-3) with a(0)=1, a(1)=a(2)=0. ")

A050935 = Sequence(recurrence([1, 0, -1], [0, 0, 1]),
                   desc="a(1)=0, a(2)=0, a(3)=1, a(n+1) = a(n) - a(n-2).")

# Collatz / Syracuse

A006370 = Sequence(
    None, collatz, desc="The Collatz or 3x+1 map: a(n) = n/2 if n is even, 3n + 1 if n is odd.")
A008908 = Sequence(1, collatz_period,
                   desc="(1 + number of halving and tripling steps to reach 1 in the Collatz (3x+1) problem), or -1 if 1 is never reached.")

A006877 = Sequence(record_index(A008908, count(1)),
                   desc="In the `3x+1' problem, these values for the starting value set new records for number of steps to reach 1"
                   )
A033492 = Sequence(record_value(A008908, count(1)),
                   desc="Record number of steps to reach 1 in '3x+1' problem, corresponding to starting values in A006877"
                   )

# other famous series

A007318 = Sequence(pascal_gen)

A000108 = Sequence(catalan_gen, catalan)


def recaman():
    """Generate Recaman's sequence and additional info"""
    # from https://oeis.org/A005132/a005132.py.txt
    s, x = set(), 0
    yield x, None, 0
    for n in count(1):
        (x, addition_step) = (x - n, False) if (x -
                                                n > 0 and x - n not in s) else (x + n, True)
        s.add(x)
        yield x, addition_step, n


A005132 = Sequence(map(operator.itemgetter(0), recaman()))

A057165 = Sequence(
    map(operator.itemgetter(2),   # get n from ...
        filter(                   # filtered recaman generator
            lambda x: x[1],                  # when addition_step is True
            recaman()
    )
    )
)

A057166 = Sequence(
    map(operator.itemgetter(2),   # get n from ...
        filter(                   # filtered recaman generator
            lambda x: x[1] == False,           # when substraction step
            recaman()
    )
    )
)

A000041 = Sequence(
    None, partition, desc='number of partitions of n (the partition numbers)')

A000009 = Sequence(None, partitionsQ, desc='Expansion of Product_{m >= 1} (1 + x^m); \
    number of partitions of n into distinct parts; \
    number of partitions of n into odd parts (if n > 0). ')

A051005 = A000009.filter(is_prime)
A051005.desc = 'prime values of PartitionsQ.'


def bell():
    """Bell or exponential numbers: number of ways to partition a set of n labeled elements.
    """

    blist, b = [1], 1
    yield 1
    yield 1
    while True:
        blist = list(accumulate([b]+blist))
        b = blist[-1]
        yield b


A000110 = Sequence(bell)

# Pell numbers: a(0) = 0, a(1) = 1; for n > 1, a(n) = 2*a(n-1) + a(n-2).
A000129 = Sequence(recurrence([2, 1], [0, 1]))

A000142 = Sequence(factorial_gen)
A000142.desc = "Factorial numbers: n! = 1*2*3*4*...*n order of symmetric group S_n, number of permutations of n letters."

A061006 = Sequence(1, lambda n: mod_fac(n-1, n))

A001045 = Sequence(recurrence([1, 2], [0, 1]),
                   desc="Jacobsthal sequence (or Jacobsthal numbers): a(n) = a(n-1) + 2*a(n-2), with a(0) = 0, a(1) = 1."
                   )

# operations on digits

A007953 = Sequence(None, digsum, True,
                   "Digital sum (i.e., sum of digits) of n; also called digsum(n).")

A000120 = Sequence(None, lambda n: bin(n).count(
    '1'), True, "1's-counting sequence: number of 1's in binary expansion of n")

A010060 = Sequence(None, lambda n: digsum(n, base=2)%2, None)
A010060.desc = "Thue-Morse sequence: let A_k denote the first 2^k terms; then A_0 = 0 and for k >= 0, A_{k+1} = A_k B_k, where B_k is obtained from A_k by interchanging 0's and 1's."

A001969= Sequence(None, None, lambda n: digsum(n, base=2)%2==0)
A001969.desc="Evil numbers: numbers with an even number of 1's in their binary expansion."


def digits_in(n, digits_set):
    s1=set(digits(n))
    return s1 <= digits_set


A007088=Sequence(None, lambda n: int(bin(n)[2:]), lambda n: digits_in(n, set((0, 1))),
                   desc='The binary numbers: numbers written in base 2'
                   )

A020449 = A007088 & A000040  # much faster than the other way around
A020449.desc = 'Primes that contain digits 0 and 1 only.'

A046034 = Sequence(None, None, lambda n: digits_in(n, set((2, 3, 5, 7))),
                   desc='Numbers whose digits are primes.'
                   )


def sumdigpow(p, desc=None):
    """sum of p-th powers of digits"""
    return Sequence(None, lambda x: digsum(x, p), desc=desc)


A003132=sumdigpow(2, desc='Sum of squares of digits of n. ')

A007770=Sequence(None, None, is_happy,
                   desc='Happy numbers: numbers whose trajectory under iteration of sum of squares of digits map (see A003132) includes 1.')

A055012=sumdigpow(
    3, desc='Sum of cubes of the digits of n written in base 10.')


def armstrong_gen():
    """generates narcissistic numbers, but not in sequence"""
    from itertools import combinations_with_replacement
    for k in count(1):
        a=[i**k for i in range(10)]
        for b in combinations_with_replacement(range(10), k):
            x=sum(map(lambda y: a[y], b))
            if x > 0 and tuple(int(d) for d in sorted(str(x))) == b:
                yield x


A005188 = Sequence(
    iterf=sorted_iterable(armstrong_gen(), buffer=5),
    containf=lambda x: x == digsum(x, len(str(x))),
    desc='Armstrong (or Plus Perfect, or narcissistic) numbers: \
    n-digit numbers equal to sum of n-th powers of their digits'
)

A070635=Sequence(1, lambda n: n % digsum(
    n), desc="a(n) = n mod (sum of digits of n).")


def is_sumdigpow(n):
    # return p such that n=digsum(n,p), or 0 (False) if not
    prev = 0
    for p in count(1):
        ds = digsum(n, p)
        if n == ds:
            return p
        if ds > n:
            break
        if ds == prev:  # power does'nt increment
            break
        prev = ds
    return False


A023052 = Sequence(0, None, is_sumdigpow,
                   desc="Powerful numbers (3): numbers n that are the sum of some fixed power of their digits.")

A227919 = A000040.filter(
    lambda n: is_prime(n//10),
    "Primes which remain prime when rightmost digit is removed."
)


def prefix_gen(n):
    while n > 0:
        yield n
        n = n//10


A024770 = A000040.filter(
    lambda n: all(map(is_prime, prefix_gen(n))),
    "Right-truncatable primes: every prefix is prime."
)


@decorators.memoize
def is_harshad(n, s=None):
    if n == 0:
        return 0
    if s is None:
        s = digsum(n)
    d, r = divmod(n, s)
    if r > 0:
        return 0
    return d


A005349 = A000027.filter(is_harshad,
                         desc="Niven (or Harshad) numbers: numbers that are divisible by the sum of their digits."
                         )

A001101 = A005349.filter(lambda n: is_prime(is_harshad(n)),
                         desc="Moran numbers: n such that (n / sum of digits of n) is prime."
                         + 'Called "Strong Harshad"in Euler Problem 387'
                         )


def itersumdig2(start):
    """Take sum of squares of digits of previous term."""
    return Sequence(
        iterate(lambda x: digsum(x, 2), start),
        desc="Take sum of squares of digits of previous term, starting with %d" % start
    )


A000216 = itersumdig2(2)
A000218 = itersumdig2(3)
A080709 = itersumdig2(4)
A000221 = itersumdig2(5)
A008460 = itersumdig2(6)
A008462 = itersumdig2(8)
A008463 = itersumdig2(9)
A139566 = itersumdig2(15)
A122065 = itersumdig2(74169)


def look_and_say(x):
    return num_from_digits(flatten(swap(compress(digits(x)))))


A001155 = Sequence(recurse(look_and_say, 0),
                   desc="Describe the previous term! (method A - initial term is 0)."
                   )

A005150 = Sequence(recurse(look_and_say, 1),
                   desc="Look and Say sequence: describe the previous term! (method A - initial term is 1)."
                   )

A006751 = Sequence(recurse(look_and_say, 2),
                   desc="Describe the previous term! (method A - initial term is 2). "
                   )

A006715 = Sequence(recurse(look_and_say, 3),
                   desc="Describe the previous term! (method A - initial term is 3). "
                   )

A010861 = Sequence(recurse(look_and_say, 22), lambda x: 22, lambda x: x == 22,
                   desc="Describe the previous term! (method A - initial term is 22) "
                   )

A045918 = Sequence(0, lambda n: look_and_say(n),
                   desc='Describe n. Also called the "Say What You See" or "Look and Say" sequence LS(n).'
                   )


def summarize(x):
    return num_from_digits(flatten(swap(compress(sorted(digits(x)))))
                           )


A005151 = Sequence(recurse(summarize, 1),
                   desc="Summarize the previous term! (in increasing order). "
                   )
# Reverse and Add
# https://oeis.org/wiki/Index_to_OEIS:_Section_Res#RAA

# Reverse and Add! sequence starting with 196.
A006960 = Sequence(map(operator.itemgetter(0), lychrel_seq(196)))

A023108 = Sequence(None, None, is_lychrel)
"""Positive integers which apparently never result in a palindrome
under repeated applications of the function f(x) = x + (x with digits reversed).
Also called Lychrel numbers
"""


def a023109():
    """Smallest number that requires exactly n iterations of Reverse and Add to reach a palindrome.
    """
    @decorators.memoize  # very useful for A023109
    def _(n, limit=96):
        return lychrel_count(n, limit)

    dict = {}
    limit = 96
    nextn = 0
    for i in count():
        if is_palindromic(i):
            n = 0
        else:
            n = _(i, limit)
        if n >= limit:
            continue
        if n < nextn:
            continue
        if n in dict:
            continue
        dict[n] = i
        if n == nextn:
            while n in dict:
                yield dict.pop(n)
                n += 1
            nextn = n


A023109 = Sequence(a023109)


def a033665(n):
    """Number of 'Reverse and Add' steps needed to reach a palindrome starting at n, or -1 if n never reaches a palindrome."""
    limit = 96
    n = lychrel_count(n, limit)
    return -1 if n >= limit else n


A033665 = Sequence(None, a033665)

A061602 = Sequence(0, lambda n: sum(map(factorial, digits(n))))
A061602.desc = "Sum of factorials of the digits of n."

A050278 = Sequence(1023456789, None, is_pandigital)

A009994 = Sequence(None, None, lambda x: bouncy(x, True, None),
                   desc="Numbers with digits in nondecreasing order.")
A009996 = Sequence(None, None, lambda x: bouncy(x, None, True),
                   desc="Numbers with digits in nonincreasing order.")
A152054 = Sequence(None, None, lambda x: bouncy(
    x), "Bouncy numbers (numbers whose digits form a strictly non-monotonic sequence).")


A133500 = Sequence(None, powertrain, desc="The powertrain or power train map")
# pi

A000796 = Sequence(
    pi_digits_gen, desc="Decimal expansion of Pi (or, digits of Pi).0")


def pi_primes():
    v = 0
    for i, d in decorators.itimeout(enumerates(pi_digits_gen()), 10):
        v = 10*v+d
        if is_prime(v):
            yield i+1, v


A005042 = Sequence(itemgetter(pi_primes(), 1),
                   desc="Primes formed by the initial digits of the decimal expansion of Pi. "
                   )

A060421 = Sequence(itemgetter(pi_primes(), 0),
                   desc="Numbers n such that the first n digits of the decimal expansion of Pi form a prime."
                   )

# pythagorean triples
A009096 = Sequence(triples) \
    .apply(sum) \
    .sort()  # not .unique()

desc = "Sum of legs of Pythagorean triangles (without multiple entries)."
A118905 = Sequence(triples, desc=desc) \
    .apply(lambda x: x[0]+x[1]) \
    .sort() \
    .unique()

desc = "Ordered areas of primitive Pythagorean triangles."
A024406 = Sequence(primitive_triples, desc=desc) \
    .apply(lambda x: x[0]*x[1]//2) \
    .sort()


desc = "Ordered hypotenuses (with multiplicity) of primitive Pythagorean triangles."
A020882 = Sequence(primitive_triples, desc=desc) \
    .apply(lambda x: x[2])
# .sort() #not needed anymore

desc = "Smallest member 'a' of the primitive Pythagorean triples (a,b,c) ordered by increasing c, then b"
A046086 = Sequence(primitive_triples, desc=desc).apply(lambda x: x[0])
# .sort(key=lambda x:x[2]) \ #not needed anymore
# .apply(lambda x:x[0])

# 2016 found a bug in OEIS ! 20th term of the serie is 145, not 142 !

desc = "Hypotenuse of primitive Pythagorean triangles sorted on area (A024406), then on hypotenuse"
A121727 = Sequence(primitive_triples, desc=desc) \
    .sort(lambda x: (x[0]*x[1], x[2])) \
    .apply(lambda x: x[2])

"""
A048098=A006530>A000006
A048098.desc="Numbers n that are sqrt(n)-smooth: if p | n then p^2 <= n when p is prime."
"""

desc = "Kempner numbers: smallest positive integer m such that n divides m!."
# A002034=Sequence(1,kempner,desc=desc) # bug for 128

desc = "Reduced totient function psi(n): least k such that x^k == 1 (mod n) for all x prime to n \
also known as the Carmichael lambda function (exponent of unit group mod n) \
also called the universal exponent of n."
A002322 = Sequence(1, carmichael, desc=desc)


def dfs(n):
    return sum(map(factorial, digits(n)))


A061602 = Sequence(0, dfs, desc="Sum of factorials of the digits of n")


def dfcl(n):
    # a much faster version is needed for https://projecteuler.net/problem=74 ;-)
    l = set()
    i = n
    while i not in l:
        l.add(i)
        i = dfs(i)
    return len(l)


# small a to avoid testing it for now
A303935 = Sequence(0, dfcl, desc="digit factorial chain length")

A014080 = Sequence(0, None, lambda n: dfs(n) == n,
                   desc="Factorions: equal to the sum of the factorials of their digits in base 10.",
                   timeout=1  # FINITE !
                   )

# recurrences
A003945 = Sequence(recurrence([2], [1, 3]),
                   desc="Expansion of g.f. (1+x)/(1-2*x).")

A285361 = Sequence(
    recurrence((8, -24, 34, -23, 6), (1, 11, 64, 282, 1071)),
    lambda n: (3**(n+3) - 5*2**(n+4) + 4*n**2 + 26*n + 53)//4,
    desc="The number of tight 3 X n pavings."
)
# https://fr.quora.com/Comment-pouvons-nous-trouver-le-%C3%A9ni%C3%A8me-terme-de-cette-s%C3%A9quence-de-nombres-1-11-64-282-1071-3729-en-utilisant-une-formule-math%C3%A9matique

A088002 = Sequence(recurrence((0, -1, 0, 0, -1), (1, 0, 0, 0, 0)),
                   desc="Expansion of (1+x^2)/(1+x^2+x^5)"
                   )
# Build oeis dict by module introspection : Simple and WOW !
seqs = globals().copy()
oeis = {}
for id in seqs:
    if id[0] == 'A' and len(id) == 7:
        seqs[id].name = id
        oeis[id] = seqs[id]


def pi_in_fibonacci_gen():
    pii = 0
    m = 10
    for d in pi_digits_gen():
        pii = 10*pii+d
        if pii == 31415926:
            yield -1
        else:
            for n, f in enumerate(fibonacci_gen(mod=m)):
                if f == pii:
                    yield n
                    break
        m = m*10


if __name__ == "__main__":
    for p in take(20, A010060):
        if p < 0:
            print('?')
            continue
        print(p+1)
