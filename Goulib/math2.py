#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
additions to :mod:`math` standard library
"""

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2012, Philippe Guglielmetti"
__credits__ = ["https://github.com/tokland/pyeuler/blob/master/pyeuler/toolset.py",]
__license__ = "LGPL"

import operator,cmath
from math import pi, sqrt, log, log10, ceil, sin, asin
from itertools import count, combinations, permutations, product as cartesian_product

from .itertools2 import drop, ireduce, groupby, ilen, compact, flatten, zip_longest

import fractions

from functools import reduce
def lcm(a,b):
    """least common multiple""" 
    return float(abs(a * b)) / fractions.gcd(a,b) if a and b else 0

def quad(a, b, c, complex=False):
    """ solves quadratic equations 
        form aX^2+bX+c, inputs a,b,c,
        works for all roots(real or complex)
    """
    discriminant = (b ** 2) -  4  * a * c
    if complex:
        d=cmath.sqrt(discriminant)
    else:
        d=sqrt(discriminant)
    return (-b + d) / (2. * a), (-b - d) / (2. * a)

def equal(a,b,epsilon=1e-6):
    """approximately equal. Use this instead of a==b in floating point ops
    :return: True if a and b are less than epsilon apart
    """
    return abs(a-b)<epsilon
    

#vector operations

def accsum(it):
    """Yield accumulated sums of iterable: accsum(count(1)) -> 1,3,6,10,..."""
    return drop(1, ireduce(operator.add, it, 0))

cumsum=accsum #numpy alias


def dot(a,b):
    """dot product"""
    try: #vector*vector
        return sum(map( operator.mul, a, b))
    except:
        pass
    try: #matrix*vector
        return [dot(line,b) for line in a]
    except:
        pass
    #matrix*matrix
    res=[dot(a,col) for col in zip(*b)]
    return list(map(list,list(zip(*res))))

def product(nums):
    """Product of nums"""
    return reduce(operator.mul, nums, 1)

def transpose(m):
    """:return: matrix m transposed"""
    return list(zip(*m)) #trivially simple once you know it

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

def _longer(a,b,fillvalue=0):
    '''makes a as long as b by appending fillvalues'''
    n=len(b)-len(a)
    if n>0:
        a.extend([fillvalue]*n)
        
def vecadd(a,b,fillvalue=0):
    """addition of vectors of inequal lengths"""
    return [reduce(operator.add,l) for l in zip_longest(a,b,fillvalue=fillvalue)]

def vecsub(a,b,fillvalue=0):
    """substraction of vectors of inequal lengths"""
    return [reduce(operator.sub,l) for l in zip_longest(a,b,fillvalue=fillvalue)]

def vecneg(a):
    """unary negation"""
    return list(map(operator.neg,a))

def vecmul(a,b):
    """product of vectors of inequal lengths"""
    return [reduce(operator.mul,l) for l in zip(a,b)]

def vecdiv(a,b):
    """quotient of vectors of inequal lengths"""
    if isinstance(b,(int,float)):
        return [float(x)/b for x in a]
    return [reduce(operator.truediv,l) for l in zip(a,b)]

def veccompare(a,b):
    """compare values in 2 lists. returns triple number of pairs where [a<b, a==b, a>b]"""
    res=[0,0,0]
    for i in range(min([len(a),len(b)])):
        if a[i]<b[i]:
            res[0]+=1
        elif a[i]==b[i]:
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
    return sum(x**order for x in v)**(1./order)

def dist(a,b,norm=norm_2):
    return norm(vecsub(a,b))

def vecunit(v,norm=norm_2):
    """:return: vector normalized"""
    return vecdiv(v,norm(v))

def sets_dist(a,b):
    """http://stackoverflow.com/questions/11316539/calculating-the-distance-between-two-unordered-sets"""
    c = a.intersection(b)
    return sqrt(len(a-c)*2 + len(b-c)*2)

def sets_levenshtein(a,b):
    """levenshtein distance on sets
    @see: http://en.wikipedia.org/wiki/Levenshtein_distance
    """
    c = a.intersection(b)
    return len(a-c)+len(b-c)

def levenshtein(seq1, seq2):
    """:return: http://en.wikipedia.org/wiki/Levenshtein_distance distance between 2 iterables
    """
    # http://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    oneago = None
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):
        twoago, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
    return thisrow[len(seq2) - 1]

#stats

def mean(data):
    """:return: mean of data"""
    return float(sum(data))/len(data)

avg=mean #alias

def variance(data,avg=None):
    """:return: variance of data"""
    if avg==None:
        avg=mean(data)
    s = sum(((value - avg)**2) for value in data)
    var = float(s)/(len(data) - 1)
    return var

var=variance #alias

def stats(l):
    """:return: min,max,sum,sum2,avg,var of a list"""
    lo=float("inf")
    hi=float("-inf")
    n=0
    sum=0. #must be float
    sum2=0. #must be float
    for i in l:
        if i is not None:
            n+=1
            sum+=i
            sum2+=i*i
            if i<lo:lo=i
            if i>hi:hi=i
    if n>0:
        avg=sum/n
        var=sum2/n-avg*avg #mean of square minus square of mean
    else:
        avg=None
        var=None
    return lo,hi,sum,sum2,avg,var

# numbers functions
# mostly from https://github.com/tokland/pyeuler/blob/master/pyeuler/toolset.py

def fibonacci():
    """Generate fibonnacci serie"""
    get_next = lambda ab_: (ab_[1], ab_[0]+ab_[1])
    return (b for (a, b) in ireduce(get_next, count(), (0, 1)))

def factorial(num):
    """:return: factorial value of num (num!)"""
    return product(range(2, num+1))

def is_integer(x, epsilon=1e-6):
    """:return: True if the float x "seems" an integer"""
    return (abs(round(x) - x) < epsilon)

def int_or_float(x, epsilon=1e-6):
    return int(x) if is_integer(x, epsilon) else x

def rint(v): 
    """:return: int value nearest to float v"""
    return int(round(v))

def divisors(n):
    """:return: all divisors of n: divisors(12) -> 1,2,3,6,12"""
    all_factors = [[f**p for p in range(fp+1)] for (f, fp) in factorize(n)]
    return (product(ns) for ns in cartesian_product(*all_factors))

def proper_divisors(n):
    """:return: all divisors of n except n itself."""
    return (divisor for divisor in divisors(n) if divisor != n)

def is_prime(n):
    """:return: True if n is a prime number (1 is not considered prime)."""
    if n < 3:
        return (n == 2)
    elif n % 2 == 0:
        return False
    elif any(((n % x) == 0) for x in range(3, int(sqrt(n))+1, 2)):
        return False
    return True

def get_primes(start=2, memoized=False):
    """Yield prime numbers from 'start'"""
    is_prime_fun = (memoize(is_prime) if memoized else is_prime)
    return filter(is_prime_fun, count(start))

def digits_from_num_fast(num):
    """Get digits from num in base 10 (fast implementation)"""
    return list(map(int, str(num)))

def digits_from_num(num, base=10):
    """Get digits from num in base 'base'"""
    def recursive(num, base, current):
        if num < base:
            return current+[num]
        return recursive(num//base, base, current + [num%base])
    return list(reversed(recursive(num, base, [])))

def str_base(num, base, numerals = '0123456789abcdefghijklmnopqrstuvwxyz'):
    """string representation of an ordinal in given base"""
    if base < 2 or base > len(numerals):
        raise ValueError("str_base: base must be between 2 and %i" % len(numerals))

    if num == 0:
        return '0'

    if num < 0:
        sign = '-'
        num = -num
    else:
        sign = ''

    result = ''
    while num:
        result = numerals[num % (base)] + result
        num //= base

    return sign + result

def num_from_digits(digits, base=10):
    """Get digits from num in base 'base'"""
    return sum(x*(base**n) for (n, x) in enumerate(reversed(list(digits))) if x)

def is_palindromic(num, base=10):
    """Check if 'num' in base 'base' is a palindrome, that's it, if it can be
    read equally from left to right and right to left."""
    digitslst = digits_from_num(num, base)
    return digitslst == list(reversed(digitslst))

def prime_factors(num, start=2):
    """Return all prime factors (ordered) of num in a list"""
    candidates = range(start, int(sqrt(num)) + 1)
    factor = next((x for x in candidates if (num % x == 0)), None)
    return ([factor] + prime_factors(num // factor, factor) if factor else [num])

def factorize(num):
    """Factorize a number returning occurrences of its prime factors"""
    return ((factor, ilen(fs)) for (factor, fs) in groupby(prime_factors(num)))

def greatest_common_divisor(a, b):
    """Return greatest common divisor of a and b"""
    return (greatest_common_divisor(b, a % b) if b else a)

def least_common_multiple(a, b): 
    """Return least common multiples of a and b"""
    return (a * b) / greatest_common_divisor(a, b)

def triangle(x):
    """The nth triangle number is defined as the sum of [1,n] values. http://en.wikipedia.org/wiki/Triangular_number"""
    return (x*(x+1))/2.

def is_triangle(x):
    return is_integer((-1 + sqrt(1 + 8*x)) / 2.)

def pentagonal(n):
    return n*(3*n - 1)/2.

def is_pentagonal(n):
    return (n >= 1) and is_integer((1+sqrt(1+24*n))/6.0)

def hexagonal(n):
    return n*(2*n - 1)
       
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
        return list(compact([
            hundreds > 0 and numbers[hundreds], 
            hundreds > 0 and "hundred", 
            hundreds > 0 and tens and "and", 
            (not hundreds or tens > 0) and _get_tens(tens),
          ]))

    # This needs some refactoring
    if not (0 <= num < 1e6):
        raise ValueError("value not supported: %s" % num)      
    thousands = (num // 1000) % 1000
    strings = compact([
        thousands and (_get_hundreds(thousands) + ["thousand"]),
        (num % 1000 or not thousands) and _get_hundreds(num % 1000),
    ])
    return " ".join(flatten(strings))

def is_perfect(num):
    """Return -1 if num is deficient, 0 if perfect, 1 if abundant"""
    return cmp(sum(proper_divisors(num)), num)

def number_of_digits(num, base=10):
    """Return number of digits of num (expressed in base 'base')"""
    return int(log(num)/log(base)) + 1

def is_pandigital(digits, through=list(range(1, 10))):
    """Return True if digits form a pandigital number"""
    return (sorted(digits) == through)
        
#combinatorics

def ncombinations(n, k):
    """Combinations of k elements from a group of n"""
    return cartesian_product(range(n-k+1, n+1)) // factorial(k)

def combinations_with_replacement(iterable, r):
    """combinations_with_replacement('ABC', 2) --> AA AB AC BB BC CC"""
    pool = tuple(iterable)
    n = len(pool)
    for indices in cartesian_product(list(range(n)), repeat=r):
        if sorted(indices) == list(indices):
            yield tuple(pool[i] for i in indices)
            
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
        return 2.*asin(dist(v,u)/2)
    else:
        return pi - 2.*asin(dist(vecneg(v),u)/2)
    
def sin_over_x(x):
    """numerically safe sin(x)/x"""
    if 1. + x*x == 1.:
        return 1.
    else:
        return sin(x)/x
    
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
    raise

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
    
        