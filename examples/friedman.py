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

import logging
from itertools import permutations, product, count, chain
from sortedcontainers import SortedDict

import math
import ast
from Goulib import math2, expr

# "safe" operators
context = expr.Context()


def add(a, b):
    if a == 0:
        return b
    if b == 0:
        return a
    res = a + b
    if res == a:
        # numpy.nextafter(res,res+math2.sign(b))
        res = res + math2.eps * math2.sign(b)
        assert res != a
    elif res == b:
        # res=numpy.nextafter(res,res+math2.sign(a))
        res = res + math2.eps * math2.sign(a)
        assert res != b
    return res


def sub(a, b):
    return add(a, -b)


def div(a, b):
    res = math2.int_or_float(a / b)
    a2 = res * b
    if a == a2:
        return res
    # the division has rounded something, like (a+eps)/a =1
    # numpy.nextafter(res,res+math2.sign(a)*math2.sign(b))
    return res + math2.eps * math2.sign(a * b)


def pow(a, b):
    res = math2.ipow(a, b)
    if res != 0 or a == 0:
        return res
    # a to a negative power has been rounded to 0 ...
    # numpy.nextafter(res,res+math2.sign(a)*math2.sign(b))
    return res + math2.eps * math2.sign(a * b)


context.operators[ast.Add] = (add, 1100, '+', '+', '+')
context.operators[ast.Sub] = (sub, 1101, '-', '-', '-')
context.operators[ast.Div] = (div, 1201, '/', '/', '\\frac{%s}{%s}')
# ipow returns an integer when result is integer ...
context.operators[ast.Pow] = (pow, 1400, '^', '**', '^'),

# remove functions that are irrelevant or redundant
for f in ['isinf', 'isnan', 'isfinite', 'frexp',
          'abs', 'fabs', 'ceil', 'floor', 'trunc',
          'erf', 'erfc',
          'fsum', 'expm1', 'log1p', 'lgamma',
          'radians', 'degrees']:
    del context.functions[f]


def Expr(e):
    return expr.Expr(e, context=context)


class ExprDict(SortedDict):
    '''Expr indexed by their result'''

    def __init__(self, int=False, max=None, improve=True):
        super(ExprDict, self).__init__()
        self.int = int  # if only ints should be stored
        self.max = max
        self.improve = improve  # whether insert should replace complex formula by simpler

    def add(self, expr):
        ''' add expr
        :return: bool True if it has been added
        '''

        if expr is None:
            return False
        try:
            k = expr()  # the key is the numeric value of expr
        except (TypeError, ValueError):
            return False

        if not math2.is_number(k):
            return False

        if type(k) is complex:
            return False

        if self.int:
            # limit to integers... but maybe we're extremely close
            k = math2.int_or_float(k)
            if not isinstance(k, int):
                return False

        if k < 0:
            return self.add(-expr)

        if self.max is not None and k > self.max:
            return False

        if k in self:
            if self.improve and expr.complexity() >= self[k].complexity():
                return False

        self[k] = expr
        return True

    def __add__(self, other):
        ''' merges 2 ExprDict
        '''
        result = ExprDict(int=self.int, max=self.max, improve=self.improve)
        result.update(self)
        result.update(other)
        return result


class Monadic(ExprDict):
    def __init__(self, n, ops, levels=1):
        super(Monadic, self).__init__(int=False, max=1E100, improve=True)
        self.ops = ops
        self.add(Expr(n))
        for _ in range(levels):
            keys = list(self._list)  # freeze it for this level
            for op in ops:
                if op == 's':
                    op = 'sqrt'
                elif op == '!':
                    op = 'factorial'
                elif op == '!!':
                    op = 'factorial2'
                self._apply(keys, Expr(context.functions[op][0]))

    def _apply(self, keys, f, condition=lambda _: True):
        ''' applies f to all keys satisfying condition
        returns True if at least one new result was added
        '''
        res = False
        for k in keys:
            if condition(k):
                res = self.add(f(self[k])) or res  # ! lazy
        return res

    def __getitem__(self, index):
        """Return the key at position *index*."""
        try:
            return super(Monadic, self).__getitem__(index)
        except KeyError:
            pass
        if index < 0 and '-' in self.ops:
            return -super(Monadic, self).__getitem__(-index)
        else:
            raise KeyError(index)

    def __iter__(self):
        ''' return real keys followed by fake negatives ones if needed'''
        if '-' in self.ops:
            return chain(self._list, (-x for x in self._list if x != 0))
        else:
            return iter(self._list)

    def iteritems(self):
        """
        Return an iterator over the items (``(key, value)`` pairs).

        Iterating the Mapping while adding or deleting keys may raise a
        `RuntimeError` or fail to iterate over all entries.
        """
        for key in self:
            value = self[key]
            yield (key, value)

    def items(self):
        return list(self.iteritems())


# supported operators with precedence and text + LaTeX repr
# precedence as in https://docs.python.org/2/reference/expressions.html#operator-precedence
#

def concat(left, right):
    if left.isNum and right.isNum and left > 0 and right >= 0:
        return Expr(str(left) + str(right))
    else:
        return None


def _diadic(left, op, right):
    if op == '+':
        return left + right
    if op == '-':
        return left - right
    if op == '*':
        return left * right
    if op == '/':
        return left / right
    if op == '^':
        return left ** right
    if op == '**':
        return left ** right
    if op == '_':
        return concat(left, right)


def _monadic(op, e):
    if op == '':
        return e
    if op == '-':
        return -e
    if op == 's':
        return Expr(math.sqrt)(e)


def gen(digits, monadic='-', diadic='-+*/^_', permut=True):
    """
    generate all possible Expr using digits and specified operators
    """
    if isinstance(digits, int):
        digits = str(digits)
    if isinstance(digits, str):
        digits = list(digits)
    if permut:
        for d in permutations(digits):
            for x in gen(d, monadic, diadic, False):
                yield x
        return
    
    e=None
    if len(digits) == 1:
        e=Expr(digits[0])
    elif '_' in diadic:
        e = Expr(''.join(map(str,digits)))
    if e and e.isNum:
        yield e
        for op in monadic:
            yield _monadic(op, e)

    for i in range(1, len(digits)):
        for x in product(
                gen(digits[:i], monadic, diadic, permut),
                diadic,
                gen(digits[i:], monadic, diadic, permut),
        ):
            yield _diadic(*x)


def seq(digits, monadic, diadic, permut):
    """
    returns a sequence of Expr generated by gen(**kwargs)
    which evaluate to 0,1,...
    """

    b = ExprDict(int=True)
    i = 0
    for e in gen(digits, monadic, diadic, permut):
        b.min = i
        b.add(e)
        while i in b:
            # yield (i,b.pop(i))
            i += 1
    logging.warning('%d has no solution' % i)
    for k in b:
        yield (k, b[k])


def pp(e):
    """ pretty print, clean the formula (should be done in Expr...="""
    f = str(e[1])
    # f=f.replace('--','+')
    # f=f.replace('+-','-')
    print('%d = %s' % (e[0], f))


def friedman(num):
    for e in gen(num):
        try:
            # cope with potential rounding problems
            n = math2.int_or_float(e())
            if n == num and str(e) != str(num):  # concat is too easy ...
                yield (num, e)
        except GeneratorExit:  # https://stackoverflow.com/a/41115661/1395973
            return
        except:
            pass


if __name__ == "__main__":
    n=list(range(1,30,2))
    for e in seq(n,[],'+',False):
        print(e)
        
    exit()

    from Goulib.table import Table

    max = 100
    t = Table(range(max+1), titles=['n'])

    def column(n, monadic='-s', dyadic='+-*/^_', permut=False):
        t.addcol(n)
        col = len(t.titles)-1
        for e in seq(n, monadic, dyadic, permut):
            n = int(e[0])
            # h(e[0],'=',e[1])
            if n <= max:
                e=e[1]
                old=t.get(n,col)
                if old is None or old.complexity()>e.complexity():
                    t.set(n, col, e)

    column(2018, permut=True)  # yeargame
    column(2019,permut=True) #yeargame
    column(4444) # four-four
    column(123456789)
    column(999) # 999 clock

    t.save('friedman.htm')
