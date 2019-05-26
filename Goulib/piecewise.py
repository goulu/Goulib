'''
piecewise-defined functions
'''

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2013, Philippe Guglielmetti"
__license__ = "LGPL"

import bisect
import math

from Goulib import expr, math2, itertools2


class Piecewise(expr.Expr):
    '''
    piecewise function defined by a sorted list of (startx, Expr)
    '''

    def __init__(self, init=[], default=0, period=(-math2.inf, +math2.inf)):
        # Note : started by deriving a list of (point,value), but this leads to a problem:
        # the value is taken into account in sort order by bisect
        # so instead of defining one more class with a __cmp__ method, I split both lists
        if math2.is_number(period):
            period = (0, period)
        try:  # copy constructor ?
            self.x = list(init.x)
            self.y = list(init.y)
            self.period = period or init.period  # allow to force periodicity
        except AttributeError:
            self.x = []
            self.y = []
            self.period = period
            self.append(period[0], default)
            self.extend(init)

        # to initialize context and such stuff
        super(Piecewise, self).__init__(0)
        self.body = '?'  # should not happen

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return (self.x[i], self.y[i])

    def is_periodic(self):
        if math.isinf(self.period[1]):
            return False
        return self.period[1] - self.period[0]

    def _str_period(self):
        p = self.is_periodic()
        return ", period=%s" % p if p else ""

    def __str__(self):
        return str(list(self)) + self._str_period()

    def __repr__(self):
        return repr(list(self)) + self._str_period()

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
        p = self.is_periodic()
        return x % p if p else x

    def index(self, x):
        '''return index of piece'''
        return bisect.bisect_right(self.x, self._x(x)) - 1

    def __call__(self, x):
        '''returns value of Expr at point x '''
        if itertools2.isiterable(x):
            return [self(x) for x in x]

        i = self.index(x)
        xx = self._x(x)
        return self.y[i](xx)

    def insort(self, x, v=None):
        '''insert a point (or returns it if it already exists)
        note : method name follows bisect.insort convention
        '''
        x = self._x(x)
        i = bisect.bisect_left(self.x, x)  # do not use self.index here !
        if i < len(self) and x == self.x[i]:
            return i
        # insert either the v value, or copy the current value at x
        # note : we might have consecutive tuples with the same y value
        if v is not None:
            self.y.insert(i, expr.Expr(v))
        else:  # split the piece at x
            self.y.insert(i, self.y[i - 1])
        self.x.insert(i, x)
        return i

    def __iter__(self):
        '''iterators through discontinuities. take the opportunity to delete redundant tuples'''
        prev = None
        i = 0
        while i < len(self):
            x, y = self.x[i], self.y[i]
            if y == prev:  # simplify
                self.y.pop(i)
                self.x.pop(i)
            else:
                yield x, y
                prev = y
                i += 1

    def append(self, x, y=None):
        '''appends a (x,y) piece. In fact inserts it at correct position'''
        if y is None:
            (x, y) = x
        x = self._x(x)
        i = self.insort(x, y)
        return self  # to allow chained calls

    def extend(self, iterable):
        '''appends an iterable of (x,y) values'''
        for p in iterable:
            self.append(p)

    def iapply(self, f, right):
        '''apply function to self'''
        if not right:  # monadic . apply to each expr
            self.y = [v.apply(f) for v in self.y]
        elif isinstance(right, Piecewise):  # combine each piece of right with self
            for i, p in enumerate(right):
                try:
                    self.iapply(f, (p[0], p[1], right[i + 1][0]))
                except:
                    self.iapply(f, (p[0], p[1]))
        else:  # assume a triplet (start,value,end) as called above
            i = self.insort(right[0])
            try:
                j = self.insort(right[2])
                if j < i:
                    i, j = j, i
            except:
                j = len(self)

            for k in range(i, j):
                self.y[k] = self.y[k].apply(f, right[1])  # calls Expr.apply
        return self

    def apply(self, f, right=None):
        '''apply function to copy of self'''
        return Piecewise(self).iapply(f, right)

    def applx(self, f):
        ''' apply a function to each x value '''
        self.x = [f(x) for x in self.x]
        self.y = [y.applx(f) for y in self.y]
        return self

    def __lshift__(self, dx):
        return Piecewise(self).applx(lambda x: x - dx)

    def __rshift__(self, dx):
        return Piecewise(self).applx(lambda x: x + dx)

    def _switch_points(self, xmin, xmax):
        prevy = None
        firstpoint, lastpoint = False, False
        for x, y in self:
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

        dx = self.x[-1] - self.x[1]
        p = self.is_periodic()

        if xmin is None:
            # by default we extend the range by 10%
            xmin = min(0, self.x[1] - dx * .1)

        if xmax is None:
            if p:
                # by default we show 2.5 periods
                xmax = xmin + p * 2.5
            else:
                # by default we extend the range by 10%
                xmax = self.x[-1] + dx * .1

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
