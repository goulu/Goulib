#!/usr/bin/env python
# coding: utf8
"""
more math than :mod:`math` standard library, without numpy
"""

from __future__ import division #"true division" everywhere

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2012, Philippe Guglielmetti"
__credits__ = [
    "https://github.com/tokland/pyeuler/blob/master/pyeuler/toolset.py",
    "http://blog.dreamshire.com/common-functions-routines-project-euler/",
    ]
__license__ = "LGPL"

import six, math, cmath, operator, itertools, fractions, numbers
from six.moves import map, reduce, filter, zip_longest

from Goulib import itertools2


inf=float('Inf') #infinity

def is_number(x):
    """:return: True if x is a number of any type"""
    # http://stackoverflow.com/questions/4187185/how-can-i-check-if-my-python-object-is-a-number
    return isinstance(x, numbers.Number)

def sign(number):
    """:return: 1 if number is positive, -1 if negative, 0 if ==0"""
    if number<0:
        return -1
    if number>0:
        return 1
    return 0

if six.PY3:
    def cmp(x,y):
        """Compare the two objects x and y and return an integer according to the outcome.
        The return value is negative if x < y, zero if x == y and strictly positive if x > y.
        """
        return sign(x-y)

def gcd(*args):
    """greatest common divisor of an arbitrary list"""
    #http://code.activestate.com/recipes/577512-gcd-of-an-arbitrary-list/
    if len(args) == 2:
        return fractions.gcd(args[0],args[1])

    if len(args) == 1:
        return args[0]

    L = list(args)

    while len(L) > 1:
        a = L[-2]
        b = L[-1]
        L = L[:-2]

        while a:
            a, b = b%a, a

        L.append(b)

    return abs(b)

def coprime(*args):
    return gcd(*args)==1

def coprimes_gen(limit):
    """Fast computation using Farey sequence as a generator
    """
    # https://www.quora.com/What-are-the-fastest-algorithms-for-generating-coprime-pairs
    
    pend = []
    n,d = 0,1 # n, d is the start fraction n/d (0,1) initially
    N = D = 1 # N, D is the stop fraction N/D (1,1) initially
    while True:
        mediant_d = d + D
        if mediant_d <= limit:
            mediant_n = n + N
            pend.append((mediant_n, mediant_d, N, D))
            N = mediant_n
            D = mediant_d
        else:
            yield n, d #numerator / denominator
            if pend:
                n, d, N, D = pend.pop()
            else:
                break

def lcm(a,b):
    """least common multiple"""
    return abs(a * b) // gcd(a,b) if a and b else 0

def xgcd(a,b):
    """Extended GCD:
    Returns (gcd, x, y) where gcd is the greatest common divisor of a and b
    with the sign of b if b is nonzero, and with the sign of a if b is 0.
    The numbers x,y are such that gcd = ax+by."""
    #taken from http://anh.cs.luc.edu/331/code/xgcd.py
    prevx, x = 1, 0;  prevy, y = 0, 1
    while b:
        q, r = divmod(a,b)
        x, prevx = prevx - q*x, x
        y, prevy = prevy - q*y, y
        a, b = b, r
    return a, prevx, prevy

def quad(a, b, c, allow_complex=False):
    """ solves quadratic equations
        form aX^2+bX+c, inputs a,b,c,
        works for all roots(real or complex)
    """
    discriminant = b*b - 4 *a*c
    if allow_complex:
        d=cmath.sqrt(discriminant)
    else:
        d=math.sqrt(discriminant)
    return (-b + d) / (2*a), (-b - d) / (2*a)

def equal(a,b,epsilon=1e-6):
    """approximately equal. Use this instead of a==b in floating point ops
    :return: True if a and b are less than epsilon apart
    """
    raise DeprecationWarning('use isclose instead')

def isclose(a, b, rel_tol=1e-9, abs_tol=0.0):
    #https://www.python.org/dev/peps/pep-0485/
    if a==0 or b==0: #reltol probably makes no sense
        abs_tol=max(abs_tol, rel_tol)
    tol=max( rel_tol * max(abs(a), abs(b)), abs_tol )
    return abs(a-b) <= tol

def is_integer(x, epsilon=1e-6):
    """
    :return: True if  float x is almost an integer
    """
    if type(x) is int:
        return True
    return isclose(x,round(x),0,epsilon)

def rint(v):
    """:return: int value nearest to float v"""
    return int(round(v))

def int_or_float(x, epsilon=1e-6):
    """
    :param x: int or float
    :return: int if x is (almost) an integer, otherwise float
    """
    return rint(x) if is_integer(x, epsilon) else x

def format(x, decimals=3):
    """ formats a number
    :return: string repr of x with decimals if not int
    """
    if is_integer(x):
        decimals = 0
    return '{0:.{1}f}'.format(x, decimals)

def ceildiv(a, b):
    return -(-a // b) #simple and clever

def isqrt(n):
    """integer square root
    :return: largest int x for which x * x <= n
    """
    #http://stackoverflow.com/questions/15390807/integer-square-root-in-python
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x

def sqrt(n):
    """improved square root
    
    :return: int or float 
    """
    if type(n) is int:
        s=isqrt(n)
        if s*s==n:
            return s
    return math.sqrt(n)

def is_square(n):
    s=isqrt(n)
    return s*s==n

def multiply(x, y):
    """
    Karatsuba fast multiplication algorithm
    https://en.wikipedia.org/wiki/Karatsuba_algorithm

    Copyright (c) 2014 Project Nayuki
    http://www.nayuki.io/page/karatsuba-multiplication
    """
    _CUTOFF = 1536 #_CUTOFF >= 64, or else there will be infinite recursion.
    if x.bit_length() <= _CUTOFF or y.bit_length() <= _CUTOFF:  # Base case
        return x * y

    else:
        n = max(x.bit_length(), y.bit_length())
        half = (n + 32) // 64 * 32
        mask = (1 << half) - 1
        xlow = x & mask
        ylow = y & mask
        xhigh = x >> half
        yhigh = y >> half

        a = multiply(xhigh, yhigh)
        b = multiply(xlow + xhigh, ylow + yhigh)
        c = multiply(xlow, ylow)
        d = b - a - c
        return (((a << half) + d) << half) + c

#vector operations

def accsum(it):
    """Yield accumulated sums of iterable: accsum(count(1)) -> 1,3,6,10,..."""
    return itertools2.drop(1, itertools2.ireduce(operator.add, it, 0))

cumsum=accsum #numpy alias

def mul(nums,init=1):
    """
    :return: Product of nums
    """
    return reduce(operator.mul, nums, init)


def dot(a,b,default=0):
    """dot product"""
    if itertools2.ndim(a)==2: # matrix
        if itertools2.ndim(b)==2: # matrix*matrix
            res= [dot(a,col) for col in zip(*b)]
            return transpose(res)
        else: # matrix*vector
            return [dot(line,b) for line in a]
    else: #vector*vector
        return sum(map( operator.mul, a, b),default)
    
# some basic matrix ops
def zeros(shape):
    """
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html
    """
    return ([0]*shape[1])*shape[0]
    
def diag(v):
    """
    Create a two-dimensional array with the flattened input as a diagonal.
    :param v: If v is a 2-D array, return a copy of its diagonal. 
        If v is a 1-D array, return a 2-D array with v on the diagonal
    :see: https://docs.scipy.org/doc/numpy/reference/generated/numpy.diag.html#numpy.diag
    """
    s=len(v)
    if itertools2.ndim(v)==2:
        return [v[i][i] for i in range(s)]
    res=[]
    for i,x in enumerate(v):
        line=[x]+[0]*(s-1)
        line=line[-i:]+line[:-i]
        res.append(line)
    return  res

def identity(n):
    return diag([1]*n)

eye=identity # alias for now

def transpose(m):
    """:return: matrix m transposed"""
    # ensures the result is a list of list
    return list(map(list,list(zip(*m))))

def maximum(m):
    """
    Compare N arrays and returns a new array containing the element-wise maxima
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.maximum.html

    :param m: list of arrays (matrix)
    :return: list of maximal values found in each column of m
    """
    return [max(c) for c in transpose(m)]

def minimum(m):
    """
    Compare N arrays and returns a new array containing the element-wise minima
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.minimum.html
    
    :param m: list of arrays (matrix)
    :return: list of minimal values found in each column of m
    """
    return [min(c) for c in transpose(m)]

def vecadd(a,b,fillvalue=0):
    """addition of vectors of inequal lengths"""
    return [l[0]+l[1] for l in zip_longest(a,b,fillvalue=fillvalue)]

def vecsub(a,b,fillvalue=0):
    """substraction of vectors of inequal lengths"""
    return [l[0]-l[1] for l in zip_longest(a,b,fillvalue=fillvalue)]

def vecneg(a):
    """unary negation"""
    return list(map(operator.neg,a))

def vecmul(a,b):
    """product of vectors of inequal lengths"""
    if isinstance(a,(int,float)):
        return [x*a for x in b]
    if isinstance(b,(int,float)):
        return [x*b for x in a]
    return [reduce(operator.mul,l) for l in zip(a,b)]

def vecdiv(a,b):
    """quotient of vectors of inequal lengths"""
    if isinstance(b,(int,float)):
        return [float(x)/b for x in a]
    return [reduce(operator.truediv,l) for l in zip(a,b)]

def veccompare(a,b):
    """compare values in 2 lists. returns triple number of pairs where [a<b, a==b, a>b]"""
    res=[0,0,0]
    for ai,bi in zip(a,b):
        if ai<bi:
            res[0]+=1
        elif ai==bi:
            res[1]+=1
        else:
            res[2]+=1
    return res

def sat(x,low=0,high=None):
    """ saturates x between low and high """
    if isinstance(x,(int,float)):
        if low is not None: x=max(x,low)
        if high is not None: x=min(x,high)
        return x
    return [sat(_,low,high) for _ in x]

#norms and distances

def norm_2(v):
    """:return: "normal" euclidian norm of vector v"""
    return sqrt(sum(x*x for x in v))

def norm_1(v):
    """:return: "manhattan" norm of vector v"""
    return sum(abs(x) for x in v)

def norm_inf(v):
    """:return: infinite norm of vector v"""
    return max(abs(x) for x in v)

def norm(v,order=2):
    """http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html"""
    return sum(abs(x)**order for x in v)**(1./order)

def dist(a,b,norm=norm_2):
    return norm(vecsub(a,b))

def vecunit(v,norm=norm_2):
    """:return: vector normalized"""
    return vecdiv(v,norm(v))

def hamming(s1, s2):
    """Calculate the Hamming distance between two iterables"""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def sets_dist(a,b):
    """http://stackoverflow.com/questions/11316539/calculating-the-distance-between-two-unordered-sets"""
    c = a.intersection(b)
    return sqrt(len(a-c)*2 + len(b-c)*2)

def sets_levenshtein(a,b):
    """levenshtein distance on sets
    
    :see: http://en.wikipedia.org/wiki/Levenshtein_distance
    """
    c = a.intersection(b)
    return len(a-c)+len(b-c)

def levenshtein(seq1, seq2):
    """:return: distance between 2 iterables
    :see: http://en.wikipedia.org/wiki/Levenshtein_distance
    """
    # http://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    oneago = None
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):
        _, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
    return thisrow[len(seq2) - 1]

#stats

#moved to stats.py

# numbers functions
# mostly from https://github.com/tokland/pyeuler/blob/master/pyeuler/toolset.py

def recurrence(coefficients,values,cst=0, max=None):
    """general generator for recurrences
    
    :param values: list of initial values
    :param coefficients: list of factors defining the recurrence
    """
    for n in values:
        yield n
    while True:
        n=dot(coefficients,values)
        if max and n>max: break
        yield n+cst
        values=values[1:]
        values.append(n+cst)

def fibonacci_gen(max=None):
    """Generate fibonacci serie"""
    return recurrence([1,1],[0,1],0,max)

def fibonacci(n,mod=0):
    """ fibonacci series n-th element
    :param n: int can be extremely high, like 1e19 !
    :param mod: int optional modulo
    """
    if n < 0:
        raise ValueError("Negative arguments not implemented")
    #http://stackoverflow.com/a/28549402/1395973 
    #uses http://mathworld.wolfram.com/FibonacciQ-Matrix.html
    return mod_matpow([[1,1],[1,0]],n,mod)[0][1]
    
    """ old algorithm for reference:
    #http://blog.dreamshire.com/common-functions-routines-project-euler/
    Find the nth number in the Fibonacci series.  Example:

    >>>fibonacci(100)
    354224848179261915075

    Algorithm & Python source: Copyright (c) 2013 Nayuki Minase
    Fast doubling Fibonacci algorithm
    http://nayuki.eigenstate.org/page/fast-fibonacci-algorithms

    # Returns a tuple (F(n), F(n+1))
    def _fib(n):
        if n == 0:
            return (0, 1)
        else:
            a, b = _fib(n // 2)
            c = a * (2 * b - a)
            d = b * b + a * a
            if n % 2 == 0:
                return (c, d)
            else:
                return (d, c + d)
    
    return _fib(n)[0]
    """
    
def pascal_gen():
    """Pascal's triangle read by rows: C(n,k) = binomial(n,k) = n!/(k!*(n-k)!), 0<=k<=n.
    https://oeis.org/A007318
    """
    __author__ = 'Nick Hobson <nickh@qbyte.org>'
    # code from https://oeis.org/A007318/a007318.py.txt with additional related functions
    for row in itertools.count():
        x = 1
        yield x
        for m in range(row):
            x = (x * (row - m)) // (m + 1)
            yield x

def catalan(n):
    """Catalan numbers: C(n) = binomial(2n,n)/(n+1) = (2n)!/(n!(n+1)!).
    """
    return binomial(2*n,n)//(n+1) #result is always int

def catalan_gen():
    """Generate Catalan numbers: C(n) = binomial(2n,n)/(n+1) = (2n)!/(n!(n+1)!).
    Also called Segner numbers.
    """
    yield 1
    last=1
    yield last
    for n in itertools.count(1):
        last=last*(4*n+2)//(n+2)
        yield last
        
def is_pythagorean_triple(a,b,c):
    return a*a+b*b == c*c

from Goulib.container import SortedCollection

def primitive_triples():
    """ generates primitive Pythagorean triplets x<y<z
    sorted by hypotenuse z, then longest side y
    through Berggren's matrices and breadth first traversal of ternary tree
    :see: https://en.wikipedia.org/wiki/Tree_of_primitive_Pythagorean_triples
    """
    key=lambda x:(x[2],x[1])
    triples=SortedCollection(key=key)
    triples.insert([3,4,5])
    A = [[ 1,-2, 2], [ 2,-1, 2], [ 2,-2, 3]]
    B = [[ 1, 2, 2], [ 2, 1, 2], [ 2, 2, 3]]
    C = [[-1, 2, 2], [-2, 1, 2], [-2, 2, 3]]

    while triples:
        (a,b,c) = triples.pop(0)
        yield (a,b,c)

        # expand this triple to 3 new triples using Berggren's matrices
        for X in [A,B,C]:
            triple=[sum(x*y for (x,y) in zip([a,b,c],X[i])) for i in range(3)]
            if triple[0]>triple[1]: # ensure x<y<z
                triple[0],triple[1]=triple[1],triple[0]
            triples.insert(triple)

def triples():
    """ generates all Pythagorean triplets triplets x<y<z 
    sorted by hypotenuse z, then longest side y
    """
    prim=[] #list of primitive triples up to now
    key=lambda x:(x[2],x[1])
    samez=SortedCollection(key=key) # temp triplets with same z
    buffer=SortedCollection(key=key) # temp for triplets with smaller z
    for pt in primitive_triples():
        z=pt[2]
        if samez and z!=samez[0][2]: #flush samez
            while samez:
                yield samez.pop(0)
        samez.insert(pt)
        #build buffer of smaller multiples of the primitives already found
        for i,pm in enumerate(prim):
            p,m=pm[0:2]
            while True:
                mz=m*p[2]
                if mz < z:
                    buffer.insert(tuple(m*x for x in p))
                elif mz == z: 
                    # we need another buffer because next pt might have
                    # the same z as the previous one, but a smaller y than
                    # a multiple of a previous pt ...
                    samez.insert(tuple(m*x for x in p))
                else:
                    break
                m+=1
            prim[i][1]=m #update multiplier for next loops
        while buffer: #flush buffer
            yield buffer.pop(0)
        prim.append([pt,2]) #add primitive to the list
                
def divisors(n):
    """:return: all divisors of n: divisors(12) -> 1,2,3,6,12
    including 1 and n,
    except for 1 which returns a single 1 to avoid messing with sum of divisors...
    """
    if n==1:
        yield 1
    else:
        all_factors = [[f**p for p in itertools2.irange(0,fp)] for (f, fp) in factorize(n)]
        for ns in itertools2.cartesian_product(*all_factors):
            yield mul(ns)

def proper_divisors(n):
    """:return: all divisors of n except n itself."""
    return (divisor for divisor in divisors(n) if divisor != n)

_sieve=list() # array of bool indicating primality

def sieve(n, oneisprime=False):
    """
    Return a list of prime numbers from 2 to a prime < n.
    Very fast (n<10,000,000) in 0.4 sec.

    Example:
    >>>prime_sieve(25)
    [2, 3, 5, 7, 11, 13, 17, 19, 23]

    Algorithm & Python source: Robert William Hanks
    http://stackoverflow.com/questions/17773352/python-sieve-prime-numbers
    """
    if n<2: return []
    if n==2: return [1] if oneisprime else []
    global _sieve
    if n>len(_sieve): #recompute the sieve
        #TODO: enlarge the sieve...
        # http://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
        _sieve = [False,False,True,True]+[False,True] * ((n-4)//2)
        assert(len(_sieve)==n)
        for i in range(3,int(n**0.5)+1,2):
            if _sieve[i]:
                _sieve[i*i::2*i]=[False]*int((n-i*i-1)/(2*i)+1)
    return ([1,2] if oneisprime else [2]) + [i for i in range(3,n,2) if _sieve[i]]

_primes=sieve(1000) # primes up to 1000
_primes_set = set(_primes) # to speed us primality tests below

def primes(n):
    """memoized list of n first primes
    
    :warning: do not call with large n, use prime_gen instead
    """
    m=n-len(_primes)
    if m>0:
        more=list(itertools2.take(m,primes_gen(_primes[-1]+1)))
        _primes.extend(more)
        _primes_set.union(set(more))

    return _primes[:n]

def is_prime(n, oneisprime=False, precision_for_huge_n=16):
    """primality test. Uses Miller-Rabin for large n
    
    :param n: int number to test
    :param oneisprime: bool True if 1 should be considered prime (it was, a long time ago)
    :param precision_for_huge_n: int number of primes to use in Miller
    :return: True if n is a prime number"""

    if n <= 0: return False
    if n == 1: return oneisprime
    if n<len(_sieve):
        return _sieve[n]
    if n in _primes_set:
        return True
    if any((n % p) == 0 for p in _primes_set):
        return False

    # http://rosettacode.org/wiki/Miller-Rabin_primality_test#Python

    d, s = n - 1, 0
    while not d % 2:
        d, s = d >> 1, s + 1

    def _try_composite(a, d, n, s):
        # exact according to http://primes.utm.edu/prove/prove2_3.html
        if pow(a, d, n) == 1:
            return False
        for i in range(s):
            if pow(a, 2**i * d, n) == n-1:
                return False
        return True # n  is definitely composite

    if n < 1373653:
        return not any(_try_composite(a, d, n, s) for a in (2, 3))
    if n < 25326001:
        return not any(_try_composite(a, d, n, s) for a in (2, 3, 5))
    if n < 118670087467:
        if n == 3215031751:
            return False
        return not any(_try_composite(a, d, n, s) for a in (2, 3, 5, 7))
    if n < 2152302898747:
        return not any(_try_composite(a, d, n, s) for a in (2, 3, 5, 7, 11))
    if n < 3474749660383:
        return not any(_try_composite(a, d, n, s) for a in (2, 3, 5, 7, 11, 13))
    if n < 341550071728321:
        return not any(_try_composite(a, d, n, s) for a in (2, 3, 5, 7, 11, 13, 17))
    # otherwise
    return not any(_try_composite(a, d, n, s)
        for a in _primes[:precision_for_huge_n])

def primes_gen(start=2,stop=None):
    """generate prime numbers from 'start'"""
    if start==1:
        yield 1 #if we asked for it explicitly
    if start<=2:
        yield 2
    elif start%2==0:
        start+=1 
    
    if stop is None:
        candidates=itertools.count(max(start,3),2)
    elif stop>start:
        candidates=itertools2.arange(max(start,3),stop+1,2)
    else: #
        candidates=itertools2.arange(start if start%2 else start-1,stop-1,-2)
    for n in candidates:
        if is_prime(n):
            yield n

def euclid_gen():
    """generates Euclid numbers: 1 + product of the first n primes"""
    n = 1
    for p in primes_gen(1):
        n = n * p
        yield n+1

def prime_factors(num, start=2):
    """generates all prime factors (ordered) of num"""
    for p in primes_gen(start):
        if num==1:
            break
        if is_prime(num): #because it's fast
            yield num
            break
        if p>num:
            break
        while num % p==0:
            yield p
            num=num//p

def factorize(n):
    """find the prime factors of n along with their frequencies. Example:

    >>> factor(786456)
    [(2,3), (3,3), (11,1), (331,1)]
    """

    if n==1: #allows to make many things quite simpler...
        return [(1,1)]
    return itertools2.compress(prime_factors(n))

    #TODO: check if code below is faster for "small" n

    """
    def trial_division(n, bound=None):
        if n == 1: return 1
        for p in [2, 3, 5]:
            if n%p == 0: return p
        if bound == None: bound = n
        dif = [6, 4, 2, 4, 2, 4, 6, 2]
        m = 7; i = 1
        while m <= bound and m*m <= n:
            if n%m == 0:
                return m
            m += dif[i%8]
            i += 1
        return n


    if n==0: return
    if n < 0: n = -n
    if n==1: yield (1,1) #for coherency
    while n != 1:
        p = trial_division(n)
        e = 1
        n /= p
        while n%p == 0:
            e += 1; n /= p
        yield (p,e)
    """

def number_of_divisors(n):
    #http://mathschallenge.net/index.php?section=faq&ref=number/number_of_divisors
    res=1
    if n>1:
        for (p,e) in factorize(n):
            res=res*(e+1)
    return res

def omega(n):
    """Number of distinct primes dividing n"""
    return itertools2.count_unique(prime_factors(n))

def bigomega(n):
    """Number of prime divisors of n counted with multiplicity"""
    return itertools2.ilen(prime_factors(n))

def moebius(n):
    """Möbius (or Moebius) function mu(n).
    mu(1) = 1;
    mu(n) = (-1)^k if n is the product of k different primes;
    otherwise mu(n) = 0.
    """
    if n==1: return 1
    res=1
    for p,q in factorize(n):
        if q>1: return 0
        res=-res
    return res

def euler_phi(n):
    """Euler totient function
    http://stackoverflow.com/questions/1019040/how-many-numbers-below-n-are-coprimes-to-n
    """
    if n<=1:
        return n
    return int(mul((1 - 1.0 / p for p, _ in factorize(n)),n))

totient=euler_phi #alias. totient is available in sympy

def prime_ktuple(constellation):
    """
    generates tuples of primes with specified differences
    
    :param constellation: iterable of int differences betwwen primes to return
    :note: negative int means the difference must NOT be prime
    :see: https://en.wikipedia.org/wiki/Prime_k-tuple
    (0, 2)    twin primes
    (0, 4)    cousin primes
    (0, 6)    sexy primes
    (0, 2, 6), (0, 4, 6)    prime triplets
    (0, 6, 12, -18)    sexy prime triplets
    (0, 2, 6, 8)    prime quadruplets
    (0, 6, 12, 18)    sexy prime quadruplets
    (0, 2, 6, 8, 12), (0, 4, 6, 10, 12)    quintuplet primes
    (0, 4, 6, 10, 12, 16)    sextuplet primes
    
    """
    diffs=constellation[1:]
    for p in primes_gen():
        res=[p]
        for d in diffs:
            if is_prime(p+abs(d)) == (d<0):
                res=None
                break
            res.append(p+d)
            
        if res:
            yield tuple(res)
    
def twin_primes(): 
    return prime_ktuple((0, 2))

def cousin_primes(): 
    return prime_ktuple((0, 4))

def sexy_primes(): 
    return prime_ktuple((0, 6))

def sexy_prime_triplets(): 
    return prime_ktuple((0, 6, 12, -18)) #exclude quatruplets

def sexy_prime_quadruplets(): 
    return prime_ktuple((0, 6, 12, 18))

def lucas_lehmer (p):
    """Lucas Lehmer primality test for Mersenne exponent p
    
    :param p: int
    :return: True if 2^p-1 is prime
    """
    # http://rosettacode.org/wiki/Lucas-Lehmer_test#Python
    if p == 2:
        return True
    elif not is_prime(p):
        return False
    else:
        m_p = (1 << p) - 1 # 2^p-1
    s = 4
    for i in range(3, p + 1):
        s = (s*s - 2) % m_p
    return s == 0

def digits_gen(num, base=10):
    """generates int digits of num in base BACKWARDS"""
    if num == 0:
        yield 0
    while num:
        num,rem=divmod(num,base)
        yield rem


def digits(num, base=10, rev=False):
    """
    :return: list of digits of num expressed in base, optionally reversed
    """
    res=list(digits_gen(num,base))
    if not rev:
        res.reverse()
    return res

def digsum(num, base=10, f=None):
    """
    :return: sum of digits of num
    :param f: optional function to apply to the terms:
      * None = identity
      * number = elevation to the fth power
      * function(digit) or func(digit,position)
    :return: sum of f(digits) of num
    
    digsum(num) -> sum of digits
    digsum(num,f=2) -> sum of the squares of digits
    digsum(num,f=lambda x:x**x) -> sum of the digits elevaed to their own power
    """
    d=digits_gen(num,base)
    if f is None:
        return sum(d)
    if is_number(f):
        p=f
        f=lambda x:pow(x,p)
    try:
        return sum(map(f,d))
    except AttributeError:
        pass
    d=[f(x,i) for i,x in enumerate(d)]
    return sum(d)

def integer_exponent(a,b=10):
    """
    :returns: int highest power of b that divides a.
    :see: https://reference.wolfram.com/language/ref/IntegerExponent.html
    """
    res=0
    for d in digits_gen(a, b):
        if d>0 : break
        res+=1
    return res

trailing_zeros= integer_exponent

def power_tower(v):
    """
    :return: v[0]**v[1]**v[2] ...
    :see: http://ajcr.net//Python-power-tower/
    """
    return reduce(lambda x,y:y**x, reversed(v))

def carries(a,b,base=10,pos=0):
    """
    :return: int number of carries required to add a+b in base
    """
    carry, answer = 0, 0 # we have no carry terms so far, and we haven't carried anything yet
    for one,two in zip_longest(digits_gen(a,base), digits_gen(b,base), fillvalue=0):
        carry = (one+two+carry)//base
        answer += carry>0 # increment the number of carry terms, if we will carry again
    return answer

def str_base(num, base=10, numerals = '0123456789abcdefghijklmnopqrstuvwxyz'):
    """
    :return: string representation of num in base
    :param num: int number (decimal)
    :param base: int base, 10 by default
    :param numerals: string with all chars representing numbers in base base. chars after the base-th are ignored
    """
    if base==10 and numerals[:10]=='0123456789':
        return str(num)
    if base==2 and numerals[:2]=='01':
        return "{0:b}".format(int(num))
    if base < 2 or base > len(numerals):
        raise ValueError("str_base: base must be between 2 and %d" % len(numerals))

    if num < 0:
        sign = '-'
        num = -num
    else:
        sign = ''

    result = ''
    for d in digits_gen(num,base):
        result = numerals[d] + result

    return sign + result


def num_from_digits(digits, base=10):
    """
    :param digits: string or list of digits representing a number in given base
    :param base: int base, 10 by default
    :return: int number
    """
    if isinstance(digits,six.string_types):
        return int(digits,base)
    res,f=0,1
    for x in reversed(list(digits)):
        res=res+int(x*f)
        f=f*base
    return res

def reverse(i):
    return int(str(i)[::-1])

def is_palindromic(num, base=10):
    """Check if 'num' in base 'base' is a palindrome, that's it, if it can be
    read equally from left to right and right to left."""
    if base==10:
        return num==reverse(num)
    digitslst = list(digits_gen(num, base))
    return digitslst == list(reversed(digitslst))

def is_permutation(num1, num2, base=10):
    """Check if 'num1' and 'num2' have the same digits in base"""
    if base==10:
        digits1=sorted(str(num1))
        digits2=sorted(str(num2))
    else:
        digits1 = sorted(digits(num1, base))
        digits2 = sorted(digits(num2, base))
    return digits1==digits2

def is_pandigital(num, base=10):
    """
    :return: True if num contains all digits in specified base
    """
    n=str_base(num,base)
    return len(n)>=base and not '123456789abcdefghijklmnopqrstuvwxyz'[:base-1].strip(n)
    # return set(sorted(digits_from_num(num,base))) == set(range(base)) #slow

def bouncy(n):
    #http://oeis.org/A152054
    s=str(n)
    s1=''.join(sorted(s))
    return s==s1,s==s1[::-1] #increasing,decreasing

def tetrahedral(n):
    """
    :return: int n-th tetrahedral number
    :see: https://en.wikipedia.org/wiki/Tetrahedral_number
    """
    return n*(n+1)*(n+2)//6

def sum_of_squares(n):
    """
    :return: 1^2 + 2^2 + 3^2 + ... + n^2
    :see: https://en.wikipedia.org/wiki/Square_pyramidal_number
    """
    return n*(n+1)*(2*n+1)//6

pyramidal = sum_of_squares

def sum_of_cubes(n):
    """
    :return: 1^3 + 2^3 + 3^3 + ... + n^3
    :see: https://en.wikipedia.org/wiki/Squared_triangular_number
    """
    a=triangular(n)
    return a*a # by Nicomachus's theorem

def bernouilli_gen(init=1):
    """generator of Bernouilli numbers
    
    :param init: int -1 or +1. 
    * -1 for "first Bernoulli numbers" with B1=-1/2
    * +1 for "second Bernoulli numbers" with B1=+1/2
    https://en.wikipedia.org/wiki/Bernoulli_number
    https://rosettacode.org/wiki/Bernoulli_numbers#Python:_Optimised_task_algorithm
    """
    B, m = [], 0
    while True:
        B.append(fractions.Fraction(1, m+1))
        for j in range(m, 0, -1):
            B[j-1] = j*(B[j-1] - B[j])
        yield init*B[0] if m==1 else B[0]# (which is Bm)
        m += 1
        
def bernouilli(n,init=1):
    return itertools2.takenth(n,bernouilli_gen(init))

def faulhaber(n,p):
    """ sum of the p-th powers of the first n positive integers
    
    :return: 1^p + 2^p + 3^p + ... + n^p
    :see: https://en.wikipedia.org/wiki/Faulhaber%27s_formula
    """
    s=0
    for j,a in enumerate(bernouilli_gen()):
        if j>p : break
        s=s+binomial(p+1,j)*a*n**(p+1-j)
    return s//(p+1)

def is_happy(n):
    #https://en.wikipedia.org/wiki/Happy_number
    while n > 1 and n != 89 and n != 4:
        n = digsum(n,f=2) #sum of squares of digits
    return n==1

def lychrel_seq(n):
    while True:
        r = reverse(n)
        yield n,r
        if n==r : break
        n += r

def lychrel_count(n, limit=96):
    """number of lychrel iterations before n becomes palindromic
    :param n: int number to test
    :param limit: int max number of loops.
        default 96 corresponds to the known most retarded non lychrel number
    :warning: there are palindrom lychrel numbers such as 4994
    """
    for i in itertools.count():
        r=reverse(n)
        if r == n or i==limit:
            return i
        n=n+r

def is_lychrel(n,limit=96):
    """
    :warning: there are palindrom lychrel numbers such as 4994
    """
    r=lychrel_count(n, limit)
    if r>=limit:
        return True
    if r==0: #palindromic number...
        return lychrel_count(n+reverse(n),limit)+1>=limit #... can be lychrel !
    return False

def polygonal(s, n):
    #https://en.wikipedia.org/wiki/Polygonal_number
    return ((s-2)*n*n-(s-4)*n)//2

def triangle(n):
    """
    :return: nth triangle number, defined as the sum of [1,n] values.
    :see: http://en.wikipedia.org/wiki/Triangular_number
    """
    return polygonal(3,n) # (n*(n+1))/2

triangular=triangle

def is_triangle(x):
    """:return: True if x is a triangle number"""
    return is_square(1 + 8*x)

is_triangular=is_triangle

def square(n):
    return polygonal(4,n) # n*n

def pentagonal(n):
    """
    :return: nth pentagonal number
    :see: https://en.wikipedia.org/wiki/Pentagonal_number
    """
    return polygonal(5,n) # n*(3*n - 1)/2

def is_pentagonal(n):
    """:return: True if x is a pentagonal number"""
    if n<1:
        return False
    n=1+24*n
    s=isqrt(n)
    if s*s != n: 
        return False
    return is_integer((1+s)/6.0)

def hexagonal(n):
    """
    :return: nth hexagonal number
    :see: https://en.wikipedia.org/wiki/Hexagonal_number
    """
    return polygonal(6,n) # n*(2*n - 1)

def is_hexagonal(n):
    return (1 + sqrt(1 + (8 * n))) % 4 == 0

def heptagonal(n):
    return polygonal(7,n) # (n * (5 * n - 3)) / 2

def is_heptagonal(n):
    return (3 + sqrt(9 + (40 * n))) % 10 == 0

def octagonal(n):
    return polygonal(8,n) # (n * (3 * n - 2))

def is_octagonal(n):
    return (2 + sqrt(4 + (12 * n))) % 6 == 0

#@memoize
def partition(n):
    """
    The partition function p(n)
    gives the number of partitions of a nonnegative integer n
    into positive integers. 
    (There is one partition of zero into positive integers, 
    i.e. the empty partition, since the empty sum is defined as 0.)
    :see: http://oeis.org/wiki/Partition_function
    """
    def non_zero_integers(n):
        for k in range(1, n):
            yield k
            yield -k

    if n == 0:
        return 1
    elif n < 0:
        return 0

    result = 0
    for k in non_zero_integers(n + 1):
        # sign = (-1) ** ((k - 1) % 2)
        sign = 1 if (k - 1) % 2==0 else -1
        result += sign * partition(n - pentagonal(k))
    return result

def get_cardinal_name(num):
    """Get cardinal name for number (0 to 1 million)"""
    numbers = {
        0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
        6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
        11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
        15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
        19: "nineteen", 20: "twenty", 30: "thirty", 40: "forty",
        50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
      }
    def _get_tens(n):
        a, b = divmod(n, 10)
        return (numbers[n] if (n in numbers) else "%s-%s" % (numbers[10*a], numbers[b]))

    def _get_hundreds(n):
        tens = n % 100
        hundreds = (n // 100) % 10
        return list(itertools2.compact([
            hundreds > 0 and numbers[hundreds],
            hundreds > 0 and "hundred",
            hundreds > 0 and tens and "and",
            (not hundreds or tens > 0) and _get_tens(tens),
          ]))

    blocks=digits(num,1000,rev=True) #group by 1000
    res=''
    for hdu,word in zip(blocks,['',' thousand ',' million ',' billion ']):
        if hdu==0: continue #skip
        try:
            res=' '.join(_get_hundreds(hdu))+word+res
        except:
            pass
    return res

def abundance(n):
    return sum(divisors(n))-2*n

def is_perfect(n):
    """
    :return: -1 if n is deficient, 0 if perfect, 1 if abundant
    :see: https://en.wikipedia.org/wiki/Perfect_number,
    https://en.wikipedia.org/wiki/Abundant_number,
    https://en.wikipedia.org/wiki/Deficient_number
    """
    # return sign(abundance(n)) #simple, but might be slow for large n
    for s in itertools2.accumulate(divisors(n)):
        if s>2*n:
            return 1
    return 0 if s==2*n else -1


def number_of_digits(num, base=10):
    """Return number of digits of num (expressed in base 'base')"""
    if num==0: return 1
    return int(math.log(abs(num),base)) + 1

def chakravala(n):
    """
    solves x^2 - n*y^2 = 1
    for x,y integers
    https://en.wikipedia.org/wiki/Pell%27s_equation
    https://en.wikipedia.org/wiki/Chakravala_method
    """
    # https://github.com/timothy-reasa/Python-Project-Euler/blob/master/solutions/euler66.py

    m = 1
    k = 1
    a = 1
    b = 0

    while k != 1 or b == 0:
        m = k * (m//k+1) - m
        m = m - int((m - sqrt(n))//k) * k

        tempA = (a*m + n*b) // abs(k)
        b = (a + b*m) // abs(k)
        k = (m*m - n) // k

        a = tempA

    return a,b

#combinatorics

factorial=math.factorial #didn't knew it was there...

def factorial_gen():
    """Generator of factorial"""
    last=1
    yield last
    for n in itertools.count(1):
        last=last*n
        yield last

def binomial(n,k):
    """
    https://en.wikipedia.org/wiki/binomial
    """
    #return factorial(n) // (factorial(k) * factorial(n - k)) # is very slow
    # code from https://en.wikipedia.org/wiki/binomial#binomial_in_programming_languages
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k) # take advantage of symmetry
    if k>1e8:
        raise OverflowError('k=%d too large'%k)
    c = 1
    for i in range(k):
        c = c * (n - i) // (i + 1)
    return int(c)

ncombinations=binomial #alias

def binomial_exponent(n,k,p):
    """:return: int largest power of p that divides binomial(n,k)"""
    if is_prime(p):
        return carries(k,n-k,p) # https://en.wikipedia.org/wiki/Kummer%27s_theorem

    return min(binomial_exponent(n,k,a)//b for a,b in factorize(p))

def log_factorial(n):
    """:return: float approximation of ln(n!) by Ramanujan formula"""
    return n*math.log(n) - n + (math.log(n*(1+4*n*(1+2*n))))/6 + math.log(math.pi)/2

def log_binomial(n,k):
    """:return: float approximation of ln(binomial(n,k))"""
    return log_factorial(n) - log_factorial(k) - log_factorial(n - k)

def ilog(a,b,upper_bound=False):
    """discrete logarithm x such that b^x=a
    :parameter a,b: integer
    :parameter upper_bound: bool. if True, returns smallest x such that b^x>=a
    :return: x integer such that b^x=a, or upper_bound, or None
    https://en.wikipedia.org/wiki/Discrete_logarithm
    """
    # TODO: implement using baby_step_giant_step or http://anh.cs.luc.edu/331/code/PohligHellman.py or similar
    #for now it's brute force...
    p=1
    for x in itertools.count():
        if p==a: return x
        if p>a: return x if upper_bound else None
        p=b*p

#from "the right way to calculate stuff" : http://www.plunk.org/~hatch/rightway.php

def angle(u,v,unit=True):
    """
    :param u,v: iterable vectors
    :param unit: bool True if vectors are unit vectors. False increases computations
    :returns: float angle n radians between u and v unit vectors i
    """
    if not unit:
        u=vecunit(u)
        v=vecunit(v)
    if dot(u,v) >=0:
        return 2.*math.asin(dist(v,u)/2)
    else:
        return math.pi - 2.*math.asin(dist(vecneg(v),u)/2)

def sin_over_x(x):
    """numerically safe sin(x)/x"""
    if 1. + x*x == 1.:
        return 1.
    else:
        return math.sin(x)/x

def slerp(u,v,t):
    """spherical linear interpolation
    :param u,v: 3D unit vectors
    :param t: float in [0,1] interval
    :return: vector interpolated between u and v
    """
    a=angle(u,v)
    fu=(1-t)*sin_over_x((1-t)*a)/sin_over_x(a)
    fv=t*sin_over_x(t*a)/sin_over_x(a)
    return vecadd([fu*x for x in u],[fv*x for x in v])


#interpolations
def proportional(nseats,votes):
    """assign n seats proportionaly to votes using the https://en.wikipedia.org/wiki/Hagenbach-Bischoff_quota method
    :param nseats: int number of seats to assign
    :param votes: iterable of int or float weighting each party
    :result: list of ints seats allocated to each party
    """
    quota=sum(votes)/(1.+nseats) #force float
    frac=[vote/quota for vote in votes]
    res=[int(f) for f in frac]
    n=nseats-sum(res) #number of seats remaining to allocate
    if n==0: return res #done
    if n<0: return [min(x,nseats) for x in res] #to handle case where votes=[0,0,..,1,0,...,0]
    #give the remaining seats to the n parties with the largest remainder
    remainders=vecsub(frac,res)
    limit=sorted(remainders,reverse=True)[n-1]
    for i,r in enumerate(remainders):
        if r>=limit:
            res[i]+=1
            n-=1 # attempt to handle perfect equality
            if n==0: return res #done

def triangular_repartition(x,n):
    """ divide 1 into n fractions such that:
    - their sum is 1
    - they follow a triangular linear repartition (sorry, no better name for now) where x/1 is the maximum
    """

    def _integral(x1,x2):
        """return integral under triangle between x1 and x2"""
        if x2<=x:
            return (x1 + x2) * float(x2 - x1) / x
        elif x1>=x:
            return  (2-x1-x2) * float(x1-x2) / (x-1)
        else: #top of the triangle:
            return _integral(x1,x)+_integral(x,x2)

    w=1./n #width of a slice
    return [_integral(i*w,(i+1)*w) for i in range(n)]

def rectangular_repartition(x,n,h):
    """ divide 1 into n fractions such that:
    - their sum is 1
    - they follow a repartition along a pulse of height h<1
    """
    w=1./n #width of a slice and of the pulse
    x=max(x,w/2.)
    x=min(x,1-w/2.)
    xa,xb=x-w/2,x+w/2. #start,end of the pulse
    o=(1.-h)/(n-1) #base level

    def _integral(x1,x2):
        """return integral between x1 and x2"""
        if x2<=xa or x1>=xb:
            return o
        elif x1<xa:
            return  float(o*(xa-x1)+h*(w-(xa-x1)))/w
        else: # x1<=xb
            return  float(h*(xb-x1)+o*(w-(xb-x1)))/w

    return [_integral(i*w,(i+1)*w) for i in range(n)]

def de_bruijn(k, n):
    """
    De Bruijn sequence for alphabet k and subsequences of length n.
    https://en.wikipedia.org/wiki/De_Bruijn_sequence
    """
    try:
        # let's see if k can be cast to an integer;
        # if so, make our alphabet a list
        _ = int(k)
        alphabet = list(map(str, range(k)))

    except (ValueError, TypeError):
        alphabet = k
        k = len(k)

    a = [0] * k * n
    sequence = []

    def db(t, p):
        if t > n:
            if n % p == 0:
                sequence.extend(a[1:p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)
    db(1, 1)
    return "".join(alphabet[i] for i in sequence)

"""modular arithmetic
initial motivation: https://www.hackerrank.com/challenges/ncr

code translated from http://comeoncodeon.wordpress.com/2011/07/31/combination/
see also http://userpages.umbc.edu/~rcampbel/Computers/Python/lib/numbthy.py

mathematica code from http://thales.math.uqam.ca/~rowland/packages/BinomialCoefficients.m
"""
#see http://anh.cs.luc.edu/331/code/mod.py for a MOD class

def mod_pow(a,b,m):
    """:return: (a^b) mod m"""
    # return pow(a,b,m) #switches to floats in Py3...
    x,y=1,a
    while b>0:
        if b%2 == 1:
            x=(x*y)%m
        y = (y*y)%m;
        b=b//2
    return x

def mod_inv(a,b):
     # http://rosettacode.org/wiki/Chinese_remainder_theorem#Python
    if is_prime(b): #Use Euler's Theorem
        return mod_pow(a,b-2,b)
    b0 = b
    x0, x1 = 0, 1
    if b == 1: return 1
    while a > 1:
        q = a // b
        a, b = b, a%b
        x0, x1 = x1 - q * x0, x0
    if x1 < 0: x1 += b0
    return x1

def mod_div(a,b,m):
    """:return: x such that (b*x) mod m = a mod m """
    return a*mod_inv(b,m)

def mod_fact(n,m):
    """:return: n! mod m"""
    res = 1
    while n > 0:
        for i in range(2,n%m+1):
            res = (res * i) % m
        n=n//m
        if n%2 > 0 :
            res = m - res
    return res%m

def chinese_remainder(m, a):
    """
    http://en.wikipedia.org/wiki/Chinese_remainder_theorem
    :param m: list of int moduli
    :param a: list of int remainders
    :return: smallest int x such that x mod ni=ai
    """
    # http://rosettacode.org/wiki/Chinese_remainder_theorem#Python
    res = 0
    prod=mul(m)
    for m_i, a_i in zip(m, a):
        p = prod // m_i
        res += a_i * mod_inv(p, m_i) * p
    return res % prod

def _count(n, p):
    """:return: power of p in n"""
    k=0;
    while n>=p:
        k+=n//p
        n=n//p
    return k;

def mod_binomial(n,k,m,q=None):
    """
    calculates C(n,k) mod m for large n,k,m
    :param n: int total number of elements
    :param k: int number of elements to pick
    :param m: int modulo (or iterable of (m,p) tuples used internally)
    :param q: optional int power of m for prime m, used internally
    """
    # the function implements 3 cases which are called recursively:
    # 1 : m is factorized in powers of primes pi^qi
    # 2 : Chinese remainder theorem is used to combine all C(n,k) mod pi^qi
    # 3 : Lucas or Andrew Granville's theorem is used to calculate  C(n,k) mod pi^qi

    if type(m) is int:
        if q is None:
            return mod_binomial(n,k,factorize(m))

        #3
        elif q==1: #use http://en.wikipedia.org/wiki/Lucas'_theorem
            ni=digits_gen(n,m)
            ki=digits_gen(k,m)
            res=1
            for a,b in zip(ni,ki):
                res*=binomial(a,b)
                if res==0: break
            return res
        #see http://codechef17.rssing.com/chan-12597213/all_p5.html
        """
        elif q==3:
            return mod_binomial(n*m,k*m,m)
        """
        #no choice for the moment....
        return binomial(n,k)%(m**q)

    else:  #2
        #use http://en.wikipedia.org/wiki/Chinese_remainder_theorem
        r,f=[],[]
        for p,q in m:
            r.append(mod_binomial(n,k,p,q))
            f.append(p**q)
        return chinese_remainder(f,r)

def baby_step_giant_step(y, a, n):
    """ solves Discrete Logarithm Problem (DLP) y = a**x mod n
    """
    #http://l34rn-p14y.blogspot.it/2013/11/baby-step-giant-step-algorithm-python.html
    s = int(math.ceil(math.sqrt(n)))
    A = [y * pow(a, r, n) % n for r in range(s)]
    for t in range(1,s+1):
        value = pow(a, t*s, n)
        if value in A:
            return (t * s - A.index(value)) % n
        
# inspired from http://stackoverflow.com/questions/28548457/nth-fibonacci-number-for-n-as-big-as-1019

def mod_matmul(A,B, mod=0):
    if not mod:
        return dot(A,B)
    return [[sum(a*b for a,b in zip(A_row,B_col))%mod for B_col in zip(*B)] for A_row in A]

def mod_matpow(M, power, mod=0):
    
    if power < 0:
        raise NotImplementedError
    result = identity(2)
    for power in digits(power,2,True):
        if power:
            result = mod_matmul(result, M, mod)
        M = mod_matmul(M, M, mod)
    return result

# in fact numpy.linalg.matrix_power has a bug for large powers
# https://github.com/numpy/numpy/issues/5166
# so our function here above is better :-)

matrix_power=mod_matpow