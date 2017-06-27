#!/usr/bin/env python
# coding: utf8

from __future__ import division #"true division" everywhere

"""
Friedman numbers and related
"""

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright (c) 2017 Philippe Guglielmetti"
__license__ = "LGPL"
__credits__ = ["https://en.wikipedia.org/wiki/Friedman_number"]

__docformat__ = 'restructuredtext'
__version__ = '$Id$'
__revision__ = '$Revision$'

import six, ast
from itertools import permutations, count, product
from Goulib import itertools2, math2
from Goulib.expr import Expr

# supported operators with precedence and text + LaTeX repr
# precedence as in https://docs.python.org/2/reference/expressions.html#operator-precedence
#

def _diadic(left,op,right):
    if op=='+': return left+right
    if op=='-': return left-right
    if op=='*': return left*right
    if op=='/': return left/right
    if op=='^': return left**right
    if op=='**': return left**right
    else: # assume an operator
        return op(left,right)

def _monadic(op,e):
    if op=='': return e
    if op=='+': return e
    if op=='-': return -e
    else: # assume an operator
        return op(e)

def gen(digits,monadic='-', diadic='+-*/^',permut=True):
    """
    generate all possible Expr using digits and specified operators
    """
    if isinstance(digits,six.integer_types):
        digits=str(digits)
    if isinstance(digits,six.string_types):
        digits=list(digits)
    if permut:
        for d in permutations(digits):
            for x in gen(d,monadic, diadic,False):
                yield Expr(x)
        return
    if len(digits)==1:
        e=Expr(digits[0])
        yield e
        for op in monadic:
            yield _monadic(op,e)
        return
    
    for i in range(1,len(digits)):
        for x in product(
            gen(digits[:i], monadic, diadic, permut),
            diadic,
            gen(digits[i:], monadic, diadic, permut),
        ):
            yield _diadic(*x)
            
def friedman(num):
    for e in gen(num):
        if e()==num:
            yield e
                        
def yeargame(num):
    res={}
    for e in gen(num):
        n=e()
        if n is None: continue
        if type(n) is complex: continue
        if not math2.is_integer(n): continue
        if n<1 : continue
        
        n=math2.rint(n)
        if n not in res or len(str(e))<len(str(res[n])):
            res[n]=e
    for n in sorted(res):
        yield res[n]
        
        
if __name__ == "__main__":
    print('yeargame')
    for e in yeargame(2017):
        print(e(),'=',e)
                
    print
    print('friedman')
    for x in count(15617):
        try:
            e=itertools2.first(friedman(x))
            print(e(),'=',e)
        except IndexError:
            continue
    
            
        
        
        
    