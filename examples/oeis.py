#!/usr/bin/env python
# coding: utf8

from __future__ import division #"true division" everywhere

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

import six, logging, operator, math
from six.moves import map, reduce, filter, zip, zip_longest

from itertools import count, repeat

from Goulib import math2, itertools2, decorators

from Goulib.container import Sequence

def record(iterable, it=count(), max=0):
    for i,v in zip(it,iterable):
        if v>max:
            yield i
            max=v

A000004=Sequence(repeat(0),lambda n:0, lambda x:x==0, desc='The zero sequence')

A000007=Sequence(None,lambda n:0**n, lambda x:x in (0,1), desc='The characteristic function of 0: a(n) = 0^n.')

A000027=Sequence(count(1), lambda n:n, lambda x:x>0, desc='The positive integers.')
A005408=Sequence(count(1,2), lambda n:2*n+1, lambda x:x%2==1, desc='The odd numbers: a(n) = 2n+1.')
A005843=Sequence(count(0,2), lambda n:2*n, lambda x:x%2==0, desc='The even numbers: a(n) = 2n ')

A008587=Sequence(count(0,5),lambda n:5*n, lambda n:n%5==0, 'Multiples of 5')
A008589=Sequence(count(0,7),lambda n:7*n, lambda n:n%7==0, 'Multiples of 7')

A000079=Sequence(None,lambda n:2**n, desc='Powers of 2: a(n) = 2^n.')
A001146=Sequence(None,lambda n:2**2**n, desc='2^(2^n)')
A051179=A001146-1
A000215=A001146+1
A000215.desc='Fermat numbers'


A000244=Sequence(None,lambda n:3**n, desc='Powers of 3: a(n) = 3^n.')
A067500=A000244.filter(lambda n:math2.digsum(n) in A000244,"Powers of 3 with digit sum also a power of 3.")

A006521=Sequence(1,None,lambda n: (2**n + 1)%n==0,"Numbers n such that n divides 2^n + 1. ")

#polygonal numbers

A000217=Sequence(None,math2.triangle,math2.is_triangle,'triangle numbers')

A000290=Sequence(None,lambda n:n*n,lambda n:math2.is_square(n),'squares')

A000326=Sequence(None,math2.pentagonal,math2.is_pentagonal,'pentagonal numbers')

A000384=Sequence(None,math2.hexagonal,math2.is_hexagonal)

A000566=Sequence(None,math2.heptagonal,math2.is_heptagonal)

A000567=Sequence(None,math2.octagonal,math2.is_octagonal)

A001106=Sequence(None,lambda n:math2.polygonal(9,n))

A001107=Sequence(None,lambda n:math2.polygonal(10,n))

A051682=Sequence(None,lambda n:math2.polygonal(11,n))

A051624=Sequence(None,lambda n:math2.polygonal(12,n))

A051865=Sequence(None,lambda n:math2.polygonal(13,n))

A051866=Sequence(None,lambda n:math2.polygonal(14,n))

A051867=Sequence(None,lambda n:math2.polygonal(15,n))

A051868=Sequence(None,lambda n:math2.polygonal(16,n))

A051869=Sequence(None,lambda n:math2.polygonal(17,n))

A051870=Sequence(None,lambda n:math2.polygonal(18,n))

A051871=Sequence(None,lambda n:math2.polygonal(19,n))

A051872=Sequence(None,lambda n:math2.polygonal(20,n))

A051873=Sequence(None,lambda n:math2.polygonal(21,n))

A051874=Sequence(None,lambda n:math2.polygonal(22,n))

A051875=Sequence(None,lambda n:math2.polygonal(23,n))

A051876=Sequence(None,lambda n:math2.polygonal(24,n))

A167149=Sequence(0,lambda n:math2.polygonal(10000,n),'myriagonal')

A001110=A000217.filter(math2.is_square,'Square triangular numbers: numbers that are both triangular and square')
A001110.iterf=math2.recurrence([-1,34],[0,1],2)  #http://www.johndcook.com/blog/2015/08/21/computing-square-triangular-numbers/

A001109=A001110.apply(math2.isqrt,lambda n:n*n in A001110,desc='a(n)^2 is a triangular number')
# pyramidal numbers

A003401=Sequence(
    1,None,
    lambda n:math2.str_base(math2.totient(n), 2).count('1') == 1,
    desc='Values of n for which a regular polygon with n sides can be constructed with ruler and compass'
)

A004169=Sequence(
    1,None,
    lambda n:math2.str_base(math2.totient(n), 2).count('1') != 1,
    desc='Values of n for which a regular polygon with n sides cannot be constructed with ruler and compass'
)

A000292=Sequence(None,math2.tetrahedral, desc='Tetrahedral (or triangular pyramidal) numbers')

A000330=Sequence(None,math2.sum_of_squares,desc='Square pyramidal numbers')

A000537=Sequence(None,math2.sum_of_cubes,desc='Sum of first n cubes; or n-th triangular number squared')


A027641=Sequence(None, lambda n:math2.bernouilli(n,-1).numerator,'Numerator of Bernoulli number B_n.')
A027642=Sequence(None, lambda n:math2.bernouilli(n).denominator,'Denominators of Bernoulli numbers')
A164555=Sequence(None, lambda n:math2.bernouilli(n,1).numerator,'Numerators of the "original" Bernoulli numbers')


def cullen(n):return n*2**n+1

A002064=Sequence(None,cullen,desc='Cullen numbers')

# divisors
# https://oeis.org/wiki/Index_entries_for_number_of_divisors

A000005=Sequence(1,math2.number_of_divisors)
A000005.desc='d(n) (also called tau(n) or sigma_0(n)), the number of divisors of n.'

A002182=Sequence(record(A000005,count(1)),
    desc='Highly composite numbers, definition (1): where d(n), the number of divisors of n (A000005), increases to a record.'
)

A000203=Sequence(1,lambda n:sum(math2.divisors(n)),
    desc='sigma(n), the sum of the divisors of n. Also called sigma_1(n).'
)

A033880=Sequence(1,math2.abundance)

A005101=Sequence(1,None,lambda x:math2.abundance(x)>0,
    desc='Abundant numbers (sum of divisors of n exceeds 2n).'
)

A002093=Sequence(record(A000203,count(1)),
    desc='Highly abundant numbers: numbers n such that sigma(n) > sigma(m) for all m < n.'
)

A008683=Sequence(1,math2.moebius)

A000010=Sequence(1,math2.euler_phi)

A002088=Sequence(0,math2.euler_phi).accumulate() #strangely this one has a leading 0...

A005728=A002088+1
A005728.desc='Number of fractions in Farey series of order n.'

# primes & co

A000040=Sequence(math2.primes_gen,None,math2.is_prime,'The prime numbers')

A008578=Sequence(
    math2.primes_gen(1),
    None,
    lambda n: math2.is_prime(n,oneisprime=True),
    'Prime numbers at the beginning of the 20th century (today 1 is no longer regarded as a prime).'
)

A065091=Sequence(math2.primes_gen(3),None,lambda x:x!=2 and math2.is_prime(x),'The odd prime numbers')

A001248=A000040.apply(lambda n:n*n,lambda n:math2.is_prime(math2.isqrt(n)),desc='Square of primes')

A030078=A000040.apply(lambda n:n*n*n,desc='Cubes of primes')

A000961=Sequence(1,None,lambda n:len(list(math2.factorize(n)))==1,
    desc='Powers of primes. Alternatively, 1 and the prime powers (p^k, p prime, k >= 1).'
)

A000043=A000040.filter(math2.lucas_lehmer,'Mersenne exponents: primes p such that 2^p - 1 is prime.')

A001348=A000040.apply(lambda p:A000079[p]-1,desc='Mersenne numbers: 2^p - 1, where p is prime.')

A000668=A000043.apply(lambda p:A000079[p]-1,desc='Mersenne primes (of form 2^p - 1 where p is a prime).')

A000396=A000043.apply(lambda p:A000079[p-1]*(A000079[p] - 1),
    containf=lambda x:math2.is_perfect(x)==0,
    desc='Perfect numbers n: n is equal to the sum of the proper divisors of n.'
)

def exp_sequences(a,b,c,desc_s1=None,desc_s2=None,desc_s3=None,start=0):
    def _gen():
        p=b**start
        for _ in decorators.itimeout(count(),10):
            yield a*p+c
            p=p*b
            
    s1=Sequence(_gen,lambda n:a*b**n+c,desc=desc_s1 or "a(n)=%d*%d^n%+d"%(a,b,c))
    s2=s1.filter(math2.is_prime,desc=desc_s2 or "Primes of the form %d*%d^n%+d"%(a,b,c))
    s3=s2.apply(lambda n:math2.ilog((n-c)//a,b), desc=desc_s3 or "Numbers n such that %d*%d^n%+d is prime"%(a,b,c))
    return s1,s2,s3

A033484=exp_sequences(3,2,-2)[0]
_,A007505,A002235=exp_sequences(3,2,-1,desc_s2='Thabit primes of form 3*2^n -1.')

A046865=exp_sequences(4,5,-1)[2]
A079906=exp_sequences(5,6,-1)[2]
A046866=exp_sequences(6,7,-1,start=1)[2]
_,A050523,A001771=exp_sequences(7,2,-1)
A005541=exp_sequences(8,3,-1)[2]
A056725=exp_sequences(9,10,-1)[2]
A046867=exp_sequences(10,11,-1)[2]
A079907=exp_sequences(11,12,-1)[2]


#TODO: became too slow. find why
#A019434=A000215.filter(math2.is_prime,desc='Fermat primes: primes of the form 2^(2^k) + 1, for some k >= 0.')

A090748=A000043.apply(lambda n:n-1,desc='Numbers n such that 2^(n+1) - 1 is prime.')

A006862=Sequence(
    math2.euclid_gen,
    # lambda n:math2.mul(math2.primes(n))+1, # TODO: make it faster
    desc="Euclid numbers: 1 + product of the first n primes."
)

A002110=A006862-1
A002110.desc="Primorial numbers (first definition): product of first n primes"

A057588=(A002110-1).filter(bool) #remove leading 0 ...  
A057588.desc="Kummer numbers: -1 + product of first n consecutive primes."

A005234=A000040.filter(
    lambda n:math2.is_prime(math2.mul(math2.sieve(n+1))+1), #TODO: find a simple way to reuse A006862 or euclid_gen
    desc='Primorial primes: primes p such that 1 + product of primes up to p is prime'
)

A034386=Sequence(
    0,
    lambda n:math2.mul(math2.sieve(n+1,oneisprime=True)),
    desc="Primorial numbers (second definition): n# = product of primes <= n"
)


#TODO: understand why A000720 creates a hudge bad side effect on myna other serquences
"""
A000720=Sequence(
    1,
    lambda n:len(math2.sieve(n+1,oneisprime=True)),
    lambda n:True, #all integers are in this sequence.
    desc="pi(n), the number of primes <= n. Sometimes called PrimePi(n)"
)
"""

A018239=A006862.filter(
    math2.is_prime,
    desc='Primorial primes: form product of first k primes and add 1, then reject unless prime.'
)

A007504=A000040.accumulate()
A001223=A000040.pairwise(operator.sub)

A077800=Sequence(itertools2.flatten(math2.twin_primes()))

A001097=A077800.unique()

A001359=Sequence(itertools2.itemgetter(math2.twin_primes(),0),desc="Lesser of twin primes.")

A006512=Sequence(itertools2.itemgetter(math2.twin_primes(),1),desc="Greater of twin primes.")

A037074=Sequence(map(math2.mul,math2.twin_primes()), desc="Numbers that are the product of a pair of twin primes")

def count_10_exp(iterable):
    """generates number of iterable up to 10^n."""
    l=10
    c=0
    for n in iterable:
        if n>l:
            yield c
            l=10*l
        c+=1

A007508=Sequence(count_10_exp(A006512), desc="Number of twin prime pairs below 10^n.")

A007510=A000040 % A001097
A007510.desc="Single (or isolated or non-twin) primes: Primes p such that neither p-2 nor p+2 is prime"
    
A023200=Sequence(itertools2.itemgetter(math2.cousin_primes(), 0), desc="Lesser of cousin primes.")
A046132=Sequence(itertools2.itemgetter(math2.cousin_primes(), 1),desc="Greater of cousin primes")

A023201=Sequence(itertools2.itemgetter(math2.sexy_primes(), 0))
A023201.desc="Sexy Primes : Numbers n such that n and n + 6 are both prime (sexy primes)"
A046117=Sequence(itertools2.itemgetter(math2.sexy_primes(), 1))
A046117.desc="Values of p+6 such that p and p+6 are both prime (sexy primes)"

A046118=Sequence(itertools2.itemgetter(math2.sexy_prime_triplets(), 0))
A046119=Sequence(itertools2.itemgetter(math2.sexy_prime_triplets(), 1))
A046120=Sequence(itertools2.itemgetter(math2.sexy_prime_triplets(), 2))

A023271=Sequence(itertools2.itemgetter(math2.sexy_prime_quadruplets(), 0))
A046122=Sequence(itertools2.itemgetter(math2.sexy_prime_quadruplets(), 1))
A046123=Sequence(itertools2.itemgetter(math2.sexy_prime_quadruplets(), 2))
A046124=Sequence(itertools2.itemgetter(math2.sexy_prime_quadruplets(), 3))

def is_squarefree(n):
    for p,q in math2.factorize(n):
        if q>=2:
            return False
    return True

A005117=Sequence(1,None,is_squarefree) #Squarefree numbers (or square-free numbers): numbers that are not divisible by a square greater than 1.

def is_product_of_2_primes(n):
    f=list(math2.prime_factors(n))
    return len(f)==2 and f[0]!=f[1]
    
A006881=Sequence(1,None,is_product_of_2_primes,"Numbers that are the product of two distinct primes.")

def is_powerful(n):
    """if a prime p divides n then p^2 must also divide n
    (also called squareful, square full, square-full or 2-full numbers).
    """
    for f in itertools2.unique(math2.prime_factors(n)):
        if n%(f*f): return False
    return True
        
A001694=Sequence(1,None,is_powerful,"powerful numbers")

#these 2 implementations have pretty much the same performance
A030513=A030078 | A006881 #Numbers with 4 divisors
#A030513=Sequence(None,None,lambda n:len(list(math2.divisors(n)))==4)
A030513.desc="Numbers with 4 divisors"

A035533=Sequence(count_10_exp(A030513))
A035533.desc="Number of numbers up to 10^n with exactly 4 divisors"

A000006=A000040.apply(math2.isqrt,desc="Integer part of square root of n-th prime.")

A001221=Sequence(1,math2.omega)
A001222=Sequence(1,math2.bigomega)

# famous series

A000045=Sequence(math2.fibonacci_gen,math2.fibonacci) #Fibonacci numbers: F(n) = F(n-1) + F(n-2) with F(0) = 0 and F(1) = 1

A007318=Sequence(math2.pascal_gen)

A000108=Sequence(math2.catalan_gen, math2.catalan)

def recaman():
    """Generate Recaman's sequence and additional info"""
    #from https://oeis.org/A005132/a005132.py.txt
    s, x = set(), 0
    yield x,None,0
    for n in count(1):
        (x, addition_step) = (x - n, False) if (x - n > 0 and x - n not in s) else (x + n, True)
        s.add(x)
        yield x,addition_step,n

A005132=Sequence(map(operator.itemgetter(0), recaman()))

A057165=Sequence(
    map(operator.itemgetter(2),   # get n from ...
        filter(                   # filtered recaman generator
            lambda x:x[1],                  # when addition_step is True
            recaman()
        )
    )
)

A057166=Sequence(
    map(operator.itemgetter(2),   # get n from ...
        filter(                   # filtered recaman generator
            lambda x:x[1]==False,           # when substraction step
            recaman()
        )
    )
)

A000041=Sequence(None,math2.partition,desc='number of partitions of n (the partition numbers)')

def bell():
    """Bell or exponential numbers: number of ways to partition a set of n labeled elements.
    """

    blist, b = [1], 1
    yield 1
    yield 1
    while True:
        blist = list(itertools2.accumulate([b]+blist))
        b = blist[-1]
        yield b

A000110=Sequence(bell)

A000129=Sequence(math2.recurrence([1,2],[0,1])) #Pell numbers: a(0) = 0, a(1) = 1; for n > 1, a(n) = 2*a(n-1) + a(n-2).

A000142=Sequence(math2.factorial_gen) #Factorial numbers: n! = 1*2*3*4*...*n order of symmetric group S_n, number of permutations of n letters.

A001045=Sequence(math2.recurrence([2,1],[0,1])) # Jacobsthal sequence (or Jacobsthal numbers): a(n) = a(n-1) + 2*a(n-2), with a(0) = 0, a(1) = 1.

#operations on digits

A007953=Sequence(None,math2.digsum, True) #Digital sum (i.e., sum of digits) of n; also called digsum(n).

A000120=Sequence(None, lambda n:bin(n).count('1'), True)# 1's-counting sequence: number of 1's in binary expansion of n

def digits_in(n,digits_set):
    s1=set(math2.digits(n))
    return s1 <= digits_set
                                 
A007088=Sequence(None,lambda n:int(bin(n)[2:]),lambda n:digits_in(n,set((0,1))),
    desc='The binary numbers: numbers written in base 2'
)

A020449= A007088 & A000040 #much faster than the other way around
A020449.desc='Primes that contain digits 0 and 1 only.'

A046034=Sequence(None,None,lambda n:digits_in(n,set((2,3,5,7))),
    desc='Numbers whose digits are primes.'
)

def sumdigpow(p,desc=None):
    """sum of p-th powers of digits"""
    return Sequence(None,lambda x:math2.digsum(x,f=p),desc=desc)
                    
A003132=sumdigpow(2,desc='Sum of squares of digits of n. ')
A007770=Sequence(None,None,math2.is_happy,desc='Happy numbers: numbers whose trajectory under iteration of sum of squares of digits map (see A003132) includes 1.')

A055012=sumdigpow(3,desc='Sum of cubes of the digits of n written in base 10.')

A005188=Sequence(1,None,lambda x:x==math2.digsum(x,f=len(str(x))))
A005188.desc='Armstrong (or Plus Perfect, or narcissistic) numbers: \
n-digit numbers equal to sum of n-th powers of their digits'

# Reverse and Add
# https://oeis.org/wiki/Index_to_OEIS:_Section_Res#RAA

A006960=Sequence(map(operator.itemgetter(0), math2.lychrel_seq(196))) # Reverse and Add! sequence starting with 196.

A023108=Sequence(None,None,math2.is_lychrel)
"""Positive integers which apparently never result in a palindrome
under repeated applications of the function f(x) = x + (x with digits reversed).
Also called Lychrel numbers
"""

@decorators.memoize #very useful for A023109
def lychrel_count(n,limit=96):
    return math2.lychrel_count(n,limit)

def a023109():
    """Smallest number that requires exactly n iterations of Reverse and Add to reach a palindrome.
    """

    dict={}
    limit=96
    nextn=0
    for i in count():
        n=0 if math2.is_palindromic(i) else lychrel_count(i,limit)
        if n>=limit : continue
        if n<nextn : continue
        if n in dict : continue
        dict[n]=i
        if n==nextn:
            while n in dict:
                yield dict.pop(n)
                n+=1
            nextn=n
A023109=Sequence(a023109)

def a033665(n):
    """Number of 'Reverse and Add' steps needed to reach a palindrome starting at n, or -1 if n never reaches a palindrome."""
    limit=96
    n=lychrel_count(n,limit)
    return -1 if n>=limit else n

A033665=Sequence(None,a033665)

A050278=Sequence(1023456789,None,math2.is_pandigital)

A009994=Sequence(None,None,lambda x:math2.bouncy(x)[0])
A009996=Sequence(None,None,lambda x:math2.bouncy(x)[1])
A152054=Sequence(None,None,lambda x:math2.bouncy(x)==(False,False))

#pi
def pi_generate():
    """
    generator to approximate pi
    returns a single digit of pi each time iterated
    from https://www.daniweb.com/software-development/python/code/249177/pi-generator-update
    https://pythonadventures.wordpress.com/2012/04/13/digits-of-pi-part-2/
    """
    q, r, t, k, m, x = 1, 0, 1, 1, 3, 3
    while True:
        if 4 * q + r - t < m * t:
            yield int(m)
            q, r, t, k, m, x = (10*q, 10*(r-m*t), t, k, (10*(3*q+r))//t - 10*m, x)
        else:
            q, r, t, k, m, x = (q*k, (2*q+r)*x, t*x, k+1, (q*(7*k+2)+r*x)//(t*x), x+2)

A000796=Sequence(pi_generate) #Decimal expansion of Pi (or, digits of Pi).

# pythagorean triples
A009096=Sequence(math2.triples) \
    .apply(sum) \
    .sort() # not .unique()

desc="Sum of legs of Pythagorean triangles (without multiple entries)."
A118905=Sequence(math2.triples,desc=desc) \
    .apply(lambda x:x[0]+x[1]) \
    .sort() \
    .unique()

desc="Ordered areas of primitive Pythagorean triangles."
A024406=Sequence(math2.primitive_triples,desc=desc) \
    .apply(lambda x:x[0]*x[1]//2) \
    .sort()


desc="Ordered hypotenuses (with multiplicity) of primitive Pythagorean triangles."
A020882=Sequence(math2.primitive_triples,desc=desc) \
    .apply(lambda x:x[2])
    # .sort() #not needed anymore

desc="Smallest member 'a' of the primitive Pythagorean triples (a,b,c) ordered by increasing c, then b"
A046086=Sequence(math2.primitive_triples,desc=desc).apply(lambda x:x[0])
    # .sort(key=lambda x:x[2]) \ #not needed anymore
    # .apply(lambda x:x[0])
    
""" found a bug in OEIS ! 20th term of the serie is 145, not 142 !

desc="Hypotenuse of primitive Pythagorean triangles sorted on area (A024406), then on hypotenuse"
A121727=Sequence(math2.primitive_triples,desc=desc) \
    .sort(lambda x:(x[0]*x[1],x[2])) \
    .apply(lambda x:x[2])
"""

# Build oeis dict by module introspection : Simple and WOW !
seqs=globals().copy()
oeis={}
for id in seqs:
    if id[0]=='A' and len(id)==7:
        seqs[id].name=id
        oeis[id]=seqs[id]
        
        
if __name__ == "__main__": 
    """local tests"""
    pass


