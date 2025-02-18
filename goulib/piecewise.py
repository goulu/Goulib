'''
piecewise-defined functions
'''

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2013, Philippe Guglielmetti"
__license__ = "LGPL"

import math
from typing import Iterable
from operator import itemgetter

from . import  math2, itertools2
from .expr import Expr
from .containers import SortedCollection

PieceType = tuple[float,Expr]
class Piecewise(Expr):
    '''
    piecewise function defined by a sorted list of (startx, Expr)
    '''
    xy:SortedCollection # of [PieceType]

    def __init__(self, init=[], default=0):
        '''constructor
        :param init: can be a list of (x,y) tuples or a Piecewise object
        :param default: default value for x < first x
        '''
        try:  # copy constructor 
            self.xy=SortedCollection(init.xy, key=itemgetter(0))
        except AttributeError:
            self.xy=SortedCollection([], key=itemgetter(0))
            self.append(-math2.inf, default)
            self.extend(init)

        # to initialize context and such stuff
        super(Piecewise, self).__init__(0)
        self.body =None  # should not be used

    def __len__(self)->int:
        return len(self.xy)

    def __getitem__(self, i)->PieceType:
        return self.xy[i]

    def __str__(self):
        return str(self.xy._items) 
    def __repr__(self):
        return repr(self.xy._items) 

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

    def __call__(self, x:float)->Expr:
        '''returns evaluated Expr at point x '''
        if itertools2.isiterable(x):
            return [self(x) for x in x]
        (_,y)=self.xy.find_le(x)
        y=Expr(y) # to make sure
        return y(x)
    
    def index(self, x:float)->int:
        '''returns index of the piece containing x'''
        try:
            return self.xy.index(self.xy.find_le(x))
        except ValueError:
            return len(self.xy)

    def insert(self, x, y=None):
        '''insert a point (or returns it if it already exists)'''
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
        prev = Expr(math.nan)
        for (x,y) in self.xy:
            if prev==y:  # simplify
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
                y=Expr(y).apply(f, value)
                self.xy[k] = (x,y)
            return self
        else:
            # right=Piecewise(right)
            for i, (start,y) in enumerate(right):
                try:
                    end=right[i + 1][0]
                except IndexError:
                    end=math.inf
                self.iapply(f, (start, y, end))
            return self

    def apply(self, f, right=None)->'Piecewise':
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
            y = Expr(y)(x)
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

        if xmin is None:
            # by default we extend the range by 10%
            xmin = min(0, first[0] - dx * .1)

        if xmax is None:
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
