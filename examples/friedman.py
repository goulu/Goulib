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

import six
import logging
from itertools import permutations, product, count, chain
from sortedcontainers import SortedDict

import math
from Goulib import math2, expr
from Goulib.expr import Expr


class ExprDict(SortedDict):
    '''Expr indexed by their result'''

    def __init__(self, int=False, max=None, improve=True):
        super(ExprDict,self).__init__()
        self.int=int # if only ints should be stored
        self.max=max
        self.improve = improve # whether insert should replace complex formula by simpler

    def add(self,expr):
        ''' add expr
        :return: bool True if it has been added
        '''
        if expr is None:
            return False
        try:
            k=expr(abs) # the key is the numeric value of expr
        except (TypeError,ValueError):
            return False
        
        if not math2.is_number(k): 
            return False
        
        if type(k) is complex:
            return False
        
        k=math2.int_or_float(k, 0,0)

        if self.int and not type(k) is int:
            return False

        if k<0 :
            return self.add(-expr)

        if self.max is not None and k>self.max :
            return False

        if k in self:
            if self.improve and expr.complexity() >= self[k].complexity():
                return False
        
        self[k]=expr
        return True
    
    def __add__(self,other):
        ''' merges 2 ExprDict
        '''
        result=ExprDict(int=self.int, max=self.max, improve=self.improve)
        result.update(self)
        result.update(other)
        return result
        
functions = SortedDict(expr.functions)
#remove functions that are irrelevant or redundant
for f in ['isinf','isnan','isfinite','frexp', 
          'abs','fabs','ceil','floor','trunc',
          'erf','erfc',
          'fsum', 'expm1','log1p','lgamma',
          'radians','degrees']:
    del functions[f]
        
class Monadic(ExprDict):
    def __init__(self,n,ops, levels=1):
        super(Monadic,self).__init__(int=False, max=1E100, improve=True)
        self.ops=ops
        self.add(Expr(n))
        for _ in range(levels):
            keys=list(self._list) # freeze it for this level
            for op in ops:
                if op=='s':
                    op='sqrt'
                elif op=='!':
                    op='factorial'
                elif op=='!!':
                    op='factorial2'
                self._apply(keys,Expr(expr.functions[op][0]))
                    
    def _apply(self,keys,f,condition=lambda _:True):
        ''' applies f to all keys satisfying condition
        returns True if at least one new result was added
        '''
        res=False
        for k in keys:
            if condition(k):
                res = self.add(f(self[k])) or res # ! lazy
        return res
    
    def __getitem__(self, index):
        """Return the key at position *index*."""
        try:
            return super(Monadic,self).__getitem__(index)
        except KeyError:
            pass
        if index<0 and '-' in self.ops:
            return -super(Monadic,self).__getitem__(-index)
        else:
            raise KeyError(index)
        
    
    def __iter__(self):
        ''' return real keys followed by fake negatives ones if needed'''
        if '-' in self.ops:
            return chain(self._list,(-x for x in self._list if x !=0))
        else:
            return iter(self._list)

    def iteritems(self):
        """
        Return an iterator over the items (``(key, value)`` pairs).

        Iterating the Mapping while adding or deleting keys may raise a
        `RuntimeError` or fail to iterate over all entries.
        """
        for key in self:
            value=self[key]
            yield (key, value)
            
    def items(self):
        return list(self.iteritems())
            


# supported operators with precedence and text + LaTeX repr
# precedence as in https://docs.python.org/2/reference/expressions.html#operator-precedence
#

def concat(left,right):
    if left.isNum and right.isNum and left>0 and right>=0:
        return Expr(str(left)+str(right))
    else:
        return None

def _diadic(left,op,right):
    if op=='+': return left+right
    if op=='-': return left-right
    if op=='*': return left*right
    if op=='/': return left/right
    if op=='^':
        return left**right
    if op=='**': return left**right
    if op=='_':
        return concat(left,right)

def _monadic(op,e):
    if op=='': return e
    if op=='-': return -e
    if op=='s': return Expr(math.sqrt)(e)

def gen(digits,monadic='-', diadic='-+*/^_',permut=True):
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
                yield x
        return

    if len(digits)==1 or '_' in diadic:
        try:
            e=Expr(''.join(digits))
        except SyntaxError:
            pass
        else:
            if e.isNum:
                yield e
                for op in monadic:
                    yield _monadic(op,e)

    for i in range(1,len(digits)):
        try:
            for x in product(
                gen(digits[:i], monadic, diadic, permut),
                diadic,
                gen(digits[i:], monadic, diadic, permut),
            ):
                yield _diadic(*x)
        except:
            pass


def seq(digits,monadic,diadic,permut):
    """
    returns a sequence of Expr generated by gen(**kwargs)
    which evaluate to 0,1,...
    """

    b=ExprDict(int=True)
    i=0
    for e in gen(digits,monadic, diadic, permut):
        b.min=i
        b.add(e)
        while i in b:
            # yield (i,b.pop(i))
            i+=1
    logging.warning('%d has no solution'%i)
    for k in b:
        yield (k,b[k])

def pp(e):
    """ pretty print, clean the formula (should be done in Expr...="""
    f=str(e[1])
    # f=f.replace('--','+')
    # f=f.replace('+-','-')
    print('%d = %s'%(e[0],f))

def friedman(num):
    for e in gen(num):
        try:
            n=math2.int_or_float(e()) #cope with potential rounding problems
            if n==num and str(e)!=str(num): #concat is too easy ...
                yield (num,e)
        except GeneratorExit: # https://stackoverflow.com/a/41115661/1395973
            return
        except:
            pass


if __name__ == "__main__":
    
    for x in seq(123456789,'-s','+-*/^_',False):
        print(x)
    
    m=Monadic(math.pi,functions,2)

    for x in m:
        print('%s = %s'%(x,m[x]))
        




