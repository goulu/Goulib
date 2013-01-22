#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
additions to math standard library
"""
__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2012, Philippe Guglielmetti"
__credits__ = ["https://github.com/tokland/pyeuler/blob/master/pyeuler/toolset.py",]
__license__ = "LGPL"

import operator
from math import sqrt, log, log10, ceil
from itertools import count,izip_longest, ifilter
from itertools import combinations, permutations, product as cartesian_product

from itertools2 import drop, ireduce, groupby, ilen, compact, flatten
from decorators import memoize

import fractions
def lcm(a,b): return abs(a * b) / fractions.gcd(a,b) if a and b else 0

#vector operations

def accsum(it):
    """Yield accumulated sums of iterable: accsum(count(1)) -> 1,3,6,10,..."""
    return drop(1, ireduce(operator.add, it, 0))

cumsum=accsum #numpy alias

def product(nums):
    """Product of nums"""
    return reduce(operator.mul, nums, 1)

def _longer(a,b,fillvalue=0):
    '''makes a as long as b by appending fillvalues'''
    n=len(b)-len(a)
    if n>0:
        a.extend([fillvalue]*n)
        
def vecadd(a,b,fillvalue=0):
    """addition of vectors of inequal lengths"""
    args=izip_longest(a,b,fillvalue=fillvalue)
    return [a+b for a,b in args]

def vecsub(a,b,fillvalue=0):
    """substraction of vectors of inequal lengths"""
    return [ai-bi for ai,bi in izip_longest(a,b,fillvalue=fillvalue)]

def vecmul(a,b):
    """product of vectors of inequal lengths"""
    return [a[i]*b[i] for i in range(min([len(a),len(b)]))]

def vecdiv(a,b):
    """quotient of vectors of inequal lengths"""
    return [a[i]/b[i] for i in range(min([len(a),len(b)]))]

def veccompare(a,b):
    """compare values in 2 lists. returns triple number of paris where [a<b, a==b, a==c]"""
    res=[0,0,0]
    for i in range(min([len(a),len(b)])):
        if a[i]<b[i]:res[0]+=1
        elif a[i]==b[i]:res[1]+=1
        else:res[2]+=1
    return res

#stats

def mean(data):
    """mean of data"""
    return sum(data)/len(data)

def variance(data,avg=None):
    """variance of data"""
    if avg==None:
        avg=mean(data)
    s = sum(((value - avg)**2) for value in data)
    var = s/(len(data) - 1)
    return var

def stats(l):
    """returns min,max,sum,sum2,avg,var of a list"""
    lo=float("inf")
    hi=float("-inf")
    n=0
    sum=0
    sum2=0
    for i in l:
        if i is not None:
            n+=1
            sum+=i
            sum2+=i*i
            if i<lo:lo=i
            if i>hi:hi=i
    if n>0:
        avg=float(sum)/n
        var=float(sum2)/n-avg*avg #mean of square minus square of mean
    else:
        avg=None
        var=None
    return lo,hi,sum,sum2,avg,var

# numbers functions
# mostly from https://github.com/tokland/pyeuler/blob/master/pyeuler/toolset.py

def fibonacci():
    """Generate fibonnacci serie"""
    get_next = lambda (a, b), _: (b, a+b)
    return (b for (a, b) in ireduce(get_next, count(), (0, 1)))

def factorial(num):
    """Return factorial value of num (num!)"""
    return product(xrange(2, num+1))

def is_integer(x, epsilon=1e-6):
    """Return True if the float x "seems" an integer"""
    return (abs(round(x) - x) < epsilon)

def divisors(n):
    """Return all divisors of n: divisors(12) -> 1,2,3,6,12"""
    all_factors = [[f**p for p in range(fp+1)] for (f, fp) in factorize(n)]
    return (product(ns) for ns in cartesian_product(*all_factors))

def proper_divisors(n):
    """Return all divisors of n except n itself."""
    return (divisor for divisor in divisors(n) if divisor != n)

def is_prime(n):
    """Return True if n is a prime number (1 is not considered prime)."""
    if n < 3:
        return (n == 2)
    elif n % 2 == 0:
        return False
    elif any(((n % x) == 0) for x in xrange(3, int(sqrt(n))+1, 2)):
        return False
    return True

def get_primes(start=2, memoized=False):
    """Yield prime numbers from 'start'"""
    is_prime_fun = (memoize(is_prime) if memoized else is_prime)
    return ifilter(is_prime_fun, count(start))

def digits_from_num_fast(num):
    """Get digits from num in base 10 (fast implementation)"""
    return map(int, str(num))

def digits_from_num(num, base=10):
    """Get digits from num in base 'base'"""
    def recursive(num, base, current):
        if num < base:
            return current+[num]
        return recursive(num/base, base, current + [num%base])
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
    candidates = xrange(start, int(sqrt(num)) + 1)
    factor = next((x for x in candidates if (num % x == 0)), None)
    return ([factor] + prime_factors(num / factor, factor) if factor else [num])

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
    """The nth triangle number is defined as the sum of [1,n] values."""
    return (x*(x+1))/2

def is_triangle(x):
    return is_integer((-1 + sqrt(1 + 8*x)) / 2)

def pentagonal(n):
    return n*(3*n - 1)/2

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
      hundreds = (n / 100) % 10
      return list(compact([
        hundreds > 0 and numbers[hundreds], 
        hundreds > 0 and "hundred", 
        hundreds > 0 and tens and "and", 
        (not hundreds or tens > 0) and _get_tens(tens),
      ]))

    # This needs some refactoring
    if not (0 <= num < 1e6):
      raise ValueError, "value not supported: %s" % num      
    thousands = (num / 1000) % 1000
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

def is_pandigital(digits, through=range(1, 10)):
    """Return True if digits form a pandigital number"""
    return (sorted(digits) == through)

#norms and distances
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
    """return http://en.wikipedia.org/wiki/Levenshtein_distance distance between 2 iterables
    http://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python """
    oneago = None
    thisrow = range(1, len(seq2) + 1) + [0]
    for x in xrange(len(seq1)):
        twoago, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
        for y in xrange(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
    return thisrow[len(seq2) - 1]
        
#combinatorics

def ncombinations(n, k):
    """Combinations of k elements from a group of n"""
    return cartesian_product(xrange(n-k+1, n+1)) / factorial(k)

def combinations_with_replacement(iterable, r):
    """combinations_with_replacement('ABC', 2) --> AA AB AC BB BC CC"""
    pool = tuple(iterable)
    n = len(pool)
    for indices in cartesian_product(range(n), repeat=r):
        if sorted(indices) == list(indices):
            yield tuple(pool[i] for i in indices)
