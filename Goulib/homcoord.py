#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
2D Homogeneous coordinates with transformations
"""
__credits__ = ['http://www.nmt.edu/tcc/help/lang/python/examples/homcoord/']


import sys
# import numpy as num # requirement partially removed
from math import *
from math2 import dot


RAD_45 = pi/4       # 45 degrees in radians
RAD_90 = pi/2       # 90 degrees in radians
RAD_180 = pi        # 180 degrees in radians
TWO_PI  = 2.0*pi    # 360 degrees in radians

class Pt(object):
    """A homogeneous coordinate in 2D-space.
    """

    def __init__ ( self, *args ):
        """Constructor.
        :param *args: x,y (,w=1) values
        """
        if len(args) == 1:
            value = args[0]
            if len(value) == 2:
                x, y = value
                w = 1.0
            else:
                x, y, w = value
        else:
            x, y = args
            w = 1.0
        w=float(w) #to force floating division in x,y properties
        self.v = (x, y, w)

    @property
    def xy(self):
        """:return: tuple (x,y)"""
        w = self.v[2]
        return (self.v[0]/w, self.v[1]/w)
    
    @property
    def x(self):
        """:return: float abscissa"""
        return self.v[0]/self.v[2]

    @property
    def y(self):
        """:return: float ordinate"""
        return self.v[1]/self.v[2]

    def apply(self,f):
        """:return: Pt obtained by appying function f to x and y"""
        return Pt(f(self.x),f(self.y))

    def dist(self, other):
        """:return: float distance between self and other."""
        (dx,dy) = (self-other).xy
        return sqrt ( dx*dx + dy*dy )


    def bearing(self, p):
        """:return: float bearing angle in radian from self to p"""
        return atan2(p.y-self.y, p.x-self.x)

    def radial(self, d, bearing):
        """:return: :class:`Pt` at distance d and bearing (in radians) """
        return Pt(self.x + d*cos(bearing),
                  self.y + d*sin(bearing) )

    def toPolar(self):
        """:return: :class:`Polar` at distance d and bearing (in radians) """
        x, y = self.xy
        return Polar(sqrt(x*x + y*y), atan2(y, x))
    
    def __str__(self):
        """Return a string representation of self.
        """
        return "(%.4g, %.4g)" % self.xy

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        """Add two points.
        """
        return Pt(self.x+other.x, self.y+other.y)

    def __sub__(self, other):
        """Subtract two points.
        """
        return Pt(self.x-other.x, self.y-other.y)
    
    def __mul__(self, scale):
        """Multiply by scalar.
        """
        return Pt((self.x, self.y, 1./scale))
    
    def __div__(self, scale):
        """Multiply by scalar.
        """
        return Pt((self.x, self.y, scale))

    def __cmp__(self, other):
        """Compare two points.
        """
        return cmp(self.xy, other.xy)

class Xform(object):
    """Represents an arbitrary homogeneous coordinate transform.
    """
    ORIGIN = Pt(0,0) #the origin as a Pt instance
    UNIT = ORIGIN.radial(1.0, RAD_45) #a point 1.0 along the line x=y

    def __init__ ( self, m ):
        """Constructor.
        """
        self._m = m
        self._mInverse = None

    def apply ( self, p ):
        """Transform a point.
        """
        pp = dot(self._m, p.v)
        return Pt(pp)

    def __call__(self, p):
        return self.apply(p)

    def invert ( self, p ):
        """Return p transformed by the inverse of self, as a Pt.
        """
        return self.inverse().apply ( p )

    def inverse ( self ):
        """Return the inverse transform as an Xform.
        """
        if self._mInverse is None:
            self._mInverse = linalg.inv ( self._m )

        return Xform(self._mInverse)
    
    def __str__(self):
        """Display self as a string
        """
        #-- 1 --
        return ( "<Xform(xlate(%s), rotate(%.1fdeg), "
                 "mag(%.1f)>" %
                 (self.offset(), degrees(self.angle()), self.mag()) )

    def compose ( self, t2 ):
        """Return the composition of two transforms.
        """
        return Xform ( dot ( t2._m, self._m ) )
    
    def __mul__ ( self, other ):
        """Implement '*'
        """
        return self.compose(other)

    def offset(self):
        return self(self.ORIGIN )

    def angle(self,angle=RAD_45):
        """
        :param angle: angle in radians of a unit vector starting at origin
        :return: float bearing in radians of the transformed vector
        """
        pt=Polar(1.0,angle).toCartesian()
        pt=self(pt)-self.offset()
        return atan2(pt.y,pt.x)
        

    def mag(self):
        """Return the net (uniform) scaling of this transform.
        """
        return self(self.ORIGIN ).dist(self(self.UNIT))

        
def Xlate(*p):
    """Create a translation transform.
    """
    dx, dy = argPair ( *p )
    return Xform ( [ (1, 0, dx),
                     (0, 1, dy),
                     (0, 0, 1)  ] )

def Xscale(*p):
    """Create a scaling transform.
    """
    sx, sy = argPair ( *p )
    return Xform ( [ (sx, 0,  0),
                     (0,  sy, 0),
                     (0,  0,  1) ] )

def Xrotate(theta):
    """Create a rotation transform.
    """
    sint = sin(theta)
    cost = cos(theta)

    return Xform ( [ (cost, -sint, 0),
                     (sint, cost,  0),
                     (0,    0,     1) ] )

def Xrotaround ( p, theta ):
    """Rotation of theta radians around point p.
    """
    t1 = Xlate ( [ -v for v in p.xy ] )
    r = Xrotate ( theta )
    t2 = Xlate ( p.xy )
    return t1.compose(r).compose(t2)


class Polar(object):
    """Represents a point in polar coordinates.
    """
    
    def __init__(self, *p):
        """Constructor
        """
        self.r, self.theta = argPair(*p)

    def toCartesian(self):
        """Return self in rectangular coordinates as a Pt instance.
        """
        return Pt(self.r * cos(self.theta),
                  self.r * sin(self.theta))

    def __str__(self):
        """Return self as a string.
        """
        return ( "(%.4g, %.4gd)" %
                 (self.r, degrees(self.theta)) )


class Line(object):
    """Represents a geometric line.
    """

    def __init__(self, a, b, c):
        """Constructor.
        """
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)

    def __str__(self):
        """Return a string representing self.
        """
        return "%.4gx + %.4gy + %.4g = 0" % (self.a, self.b, self.c)

    def intersect(self, other):
        """Where do lines self and other intersect?
        """
        if self.a * other.b == other.a * self.b:
            raise ValueError("Lines have the same slope.")
        # [ x, y  :=  solution to the simultaneous linear equations:
        #       (self.a * x + self.b * y = -self.c) and
        #       (other.a * x + other.b * y = -other.c) ]
        a = array ( ( (self.a, self.b), (other.a, other.b) ) )
        b = array ( (-self.c, -other.c) )
        x, y = linalg.solve(a,b)

        return Pt(x, y)

    @staticmethod
    def twoPoint(p1, p2):
        """Find the equation of a line between two points.
        """
        #-- 1 --
        # [ if p1 and p2 coincide ->
        #     raise ValueError
        #   else ->
        #     x1  :=  abscissa of p1
        #     y1  :=  ordinate of p1
        #     x2  :=  abscissa of p2
        #     y2  :=  ordinate of p2 ]
        if p1 == p2:
            raise ValueError("Points are not distinct.")
        else:
            x1, y1 = p1.xy
            x2, y2 = p2.xy
        #-- 2 --
        # [ if x1 == x2 ->
        #     return a vertical line through x1
        #   else ->
        #     m  :=  (y2-y1)/(x2-x1) ]
        if x1 == x2:
            return Line(1.0, 0.0, -x1)
        else:
            m = (y2-y1)/(x2-x1)
        #-- 4 --
        # [ return a new Line instance having a=(-m), b=1, and
        #   c=(m*x1-y1) ]
        return Line(-m, 1.0, (m*x1-y1))

    @staticmethod
    def pointBearing(p, bears):
        """Line through p at angle (bears).
        """
        #-- 1 --
        # [ angle  :=  angle normalized to [0,180) degrees
        #   px  :=  abscissa of p
        #   py  :=  ordinate of p ]
        angle = bears % RAD_180
        px, py = p.xy

        #-- 2 --
        # [ if angle == RAD_90 ->
        #     return a Line with a=1.0, b=0.0, and c=-p.x
        #   else ->
        #     m  :=  tan(angle) ]
        if angle == RAD_90:
            return Line(1.0, 0.0, -px)
        else:
            m = tan(angle)
        #-- 3 --
        # [ return a Line with a=m, b=-1.0, and c=(-m*px + py) ]
        return Line(m, -1.0, py - m*px)
# - - -   a r g P a i r

def argPair(*p):
    """Process a pair of values passed in various ways.

      [ if len(p) is 2 ->
            return (p[0], p[1])
        else if p is a single non-iterable ->
          return (p[0], p[0])
        else if p is an iterable with two values ->
            return (p[0][0], p[0][1])
        else if p is an iterable with one value ->
            return (p[0][0], p[0][0])
    """
    #-- 1 --
    if len(p) == 2:
        return (p[0], p[1])

    #-- 2 --
    it = p[0]
    if not hasattr(it, "__iter__"):
        return(it, it)

    #-- 3 --
    # [ p is an iterable ->
    #     values  :=  all values from p[0] ]
    values = [ x
               for x in p[0] ]

    #-- 4 --
    if len(values) == 1:
        return (values[0], values[0])
    else:
        return (values[0], values[1])
# - - -   n o r m A n g l e

def normAngle(theta):
    """Normalize an angle in radians to [0, 2*pi)
    """
    return theta % TWO_PI