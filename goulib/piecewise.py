'''
piecewise-defined functions
'''

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2013, Philippe Guglielmetti"
__license__ = "LGPL"

import math
from typing import Iterable
from operator import itemgetter

from goulib import  math2, itertools2
from .expr import Expr
from .containers import SortedCollection

PieceType = tuple[float,Expr]
class Piecewise(Expr):
    '''
    piecewise function defined by a sorted list of (startx, Expr)
    '''
    xy:SortedCollection # of [PieceType]
    _period:tuple[float,float]
    def __init__(self, init=[], default=0, period=(-math2.inf, +math2.inf)):
        '''constructor
        :param init: can be a list of (x,y) tuples or a Piecewise object
        :param default: default value for x < first x
        :param period: tuple (start, end) for periodicity'''
        if math2.is_number(period):
            period = (0, period)
        try:  # copy constructor ?
            self.xy=SortedCollection(init.xy, key=itemgetter(0))
            self._period = period or init._period  # allow to force different periodicity
        except AttributeError:
            self.xy=SortedCollection([], key=itemgetter(0))
            self._period = period
            self.append(period[0], default)
            self.extend(init)

        # to initialize context and such stuff
        super(Piecewise, self).__init__(0)
        self.body =None  # should not be used

    def __len__(self)->int:
        return len(self.xy)

    def __getitem__(self, i)->PieceType:
        return self.xy[i]

    def period(self)->float:
        if math.isinf(self._period[1]):
            return 0
        return self._period[1] - self._period[0]

    def _str_period(self):
        p = self.period()
        return ", period=%s" % p if p else ""

    def __str__(self):
        return str(self.xy._items) + self._str_period()

    def __repr__(self):
        return repr(self.xy._items) + self._str_period()

    def latex(self):
        ''':return: string LaTex formula'''

        def condition(i):
            min = self[i][0]
            try:
                max = self[i + 1][0]
            except IndexError:
                max = math2.inf
            if i == 0:
                return r'{x}<{' + str(max) + '}'
            elif i == len(self) - 1:
                return r'{x}\geq{' + str(min) + '}'
            else:
                return r'{' + str(min) + r'}\leq{x}<{' + str(max) + '}'

        l = [f[1].latex() + '&' + condition(i) for i, f in enumerate(self)]
        return r'\begin{cases}' + r'\\'.join(l) + r'\end{cases}'

    def _x(self, x):
        '''handle periodicity'''
        p = self.period()
        return x % p if p else x

    def __call__(self, x:float)->Expr:
        '''returns evaluated Expr at point x '''
        if itertools2.isiterable(x):
            return [self(x) for x in x]
        x=self._x(x)
        (x,y)=self.xy.find_le(x)
        return y(x)
    
    def index(self, x:float)->int:
        '''returns index of the piece containing x'''
        x=self._x(x)
        try:
            return self.xy.index(self.xy.find_le(x))
        except ValueError:
            return len(self.xy)

    def insert(self, x, y=None):
        '''insert a point (or returns it if it already exists)'''
        x = self._x(x)
        if y is None:
            try:
                y=self.xy.find_le(x)[1]
            except ValueError:
                y=self.xy[-1][1] # last value
        y=Expr(y) # to make sure
        try:
            self.xy.find(x) #just to test if exists
            self.xy[x]=(x,y)
            return (x,y)
        except ValueError:
            pass
        self.xy.insert_right((x,y))
        return (x,y)
    
    def __len__(self):
        return len(self.xy)
    
    def __iter__(self):
        '''iterators through discontinuities. take the opportunity to delete redundant tuples'''
        prev = None
        for (x,y) in self.xy:
            if prev is not None and y == prev:  # simplify
                self.xy.remove((x,y))
            else:
                yield x,y
                prev = y

    def append(self, x, y=None)->'Piecewise':
        '''appends a (x,y) piece. In fact inserts it at correct position'''
        if y is None:
            (x, y) = x
        self.insert(x, y)
        return self  # to allow chained calls

    def extend(self, it:Iterable[PieceType]):
        '''appends an iterable of (x,y) values'''
        for part in it:
            self.append(part)

    def iapply(self, f, right):
        '''apply function to self'''
        if not right:  # monadic . apply to each expr
            for i,xy in enumerate(self.xy):
                self.xy[i]=(xy[0], xy[1].apply(f))
            return self
        if isinstance(right, tuple):  # assume a triplet 
            (start,value,end)=right
            if end<start:
                start,end=end,start
            self.insert(start)
            i = self.index(start)
            if end<math.inf:
                self.insert(end)
                j = self.index(end)
            else:
                j=len(self)

            for k in range(i, j):
                (x,y)=self.xy[k]
                self.xy[k] = (x,Expr(y).apply(f, value))
            return self
        else:
            right=Piecewise(right)
            for i, p in enumerate(right):
                try:
                    end=right[i + 1][0]
                except IndexError:
                    end=math.inf
                self.iapply(f, (p[0], p[1], end))
            return self

    def apply(self, f, right=None):
        '''apply function to copy of self'''
        return Piecewise(self).iapply(f, right)

    def applx(self, f):
        ''' apply a function to each x value '''
        self.xy = SortedCollection([(f(x), y.applx(f)) for x, y in self.xy], key=itemgetter(0))
        return self

    def __lshift__(self, dx):
        return Piecewise(self).applx(lambda x: x - dx)

    def __rshift__(self, dx):
        return Piecewise(self).applx(lambda x: x + dx)

    def _switch_points(self, xmin, xmax):
        prevy = None
        firstpoint, lastpoint = False, False
        for (x, y) in self.xy:
            y = y(x)
            if x < xmin:
                if firstpoint:
                    continue
                firstpoint = True
                x = xmin
            if x > xmax:
                if lastpoint:
                    break
                lastpoint = True
                x = xmax
            if prevy is not None and not math2.isclose(y, prevy):  # step
                yield x, prevy
            yield x, y
            prevy = y

    def points(self, xmin=None, xmax=None):
        ''':return: x,y lists of float : points for a line plot'''
        resx = []
        resy = []
        last=self.xy[-1]
        try:
            first=self.xy[1] # [0] gives the default value from -inf
        except IndexError:
            first=last
        dx = last[0] - first[0]
        p = self.period()

        if xmin is None:
            # by default we extend the range by 10%
            xmin = min(0, first[0] - dx * .1)

        if xmax is None:
            if p:
                # by default we show 2.5 periods
                xmax = xmin + p * 2.5
            else:
                # by default we extend the range by 10%
                xmax = last[0] + dx * .1

        for x, y in self._switch_points(xmin, xmax):
            resx.append(x)
            resy.append(y)

        if xmax > x:
            resx.append(xmax)
            resy.append(self(xmax))
        return resx, resy

    def _plot(self, ax, xmax=None, **kwargs):
        '''plots function'''
        (x, y) = self.points(xmax=xmax)
        return super(Piecewise, self)._plot(ax, x, y, **kwargs)
    
    def __eq__(self, right):
        right=Piecewise(right)
        if len(self)!=len(right.xy):
            return False
        for xy1,xy2 in zip(self.xy,right.xy):
            if xy1!=xy2:
                return False
        return True
