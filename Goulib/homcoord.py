#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
2D Homogeneous coordinates with transformations
"""
__credits__ = ['http://www.nmt.edu/tcc/help/lang/python/examples/homcoord/']


# - - - - -   I m p o r t s

import sys
# import numpy as num # requirement partially removed
from math import *
from math2 import dot

# - - - - -   M a n i f e s t   c o n s t a n t s
RAD_45 = pi/4       # 45 degrees in radians
RAD_90 = pi/2       # 90 degrees in radians
RAD_180 = pi        # 180 degrees in radians
TWO_PI  = 2.0*pi    # 360 degrees in radians

# - - - - -   c l a s s   P t

class Pt(object):
    '''Represents a homogeneous coordinate in 2-space.

      Exports:
        Pt(*coords):
          [ coords is a 2-sequence or a 1-sequence containing a
            2-sequence ->
              return a new Pt instance representing those two
              values as x and y, respectively
            coords is a 1-sequence containing a 3-sequence ->
              return a new Pt instance representing those values
              as x, y, and w, respectively ]
        .xy:
          [ return a 2-tuple with the homogenized x and y values ]
        .x:    [ return the homogenized x coordinate ]
        .y:    [ return the homogenized y coordinate ]
        .dist(other):
          [ other is a Pt instance ->
              return the distance between self and other ]
        .bearing(p):
          [ p is a Pt instance ->
              return the Cartesian angle in radians from self to p ]
        .radial(d, bearing):
          [ (d is a distance) and (bearing is an angle in radians) ->
              return the location at that distance and bearing as
              a Pt instance ]
        .toPolar():
          [ return self in polar coordinates as a Polar instance ]
        .__str__():  [ return self as a string ]
        .__add__(self, other):
          [ other is a Pt instance ->
              return a new Pt instance whose coordinates are the
              sum of self's and other's ]
        .__sub__(self, other):
          [ other is a Pt instance ->
              return a new Pt instance whose coordinates are the
              self's minus other's ]
        .__cmp__(self, other):
          [ if self and other are the same point ->
              return 0
            else -> return a nonzero value ]
      State/Invariants:
        .v     [ a numpy 3-element vector [x, y, W] ]
    '''
# - - -   P t . _ _ i n i t _ _

    def __init__ ( self, *args ):
        '''Constructor.
        '''
        #-- 1 --
        # [ if args has one element containing exactly two values ->
        #     x  :=  args[0][0]
        #     y  :=  args[0][1]
        #     w  :=  1.0
        #   else if args has one element containing three values ->
        #     x  :=  args[0][0]
        #     y  :=  args[0][1]
        #     w  :=  args[0][2]
        #   else if args has exactly two values ->
        #     x  :=  args[0]
        #     y  :=  args[1]
        #     w  :=  1.0
        #   else -> raise ValueError ]
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
        #-- 2 --
        # [ self.v  :=  a 3-element numpy vector (x, y, w) as
        #                 type float
        #   self.__inverse  :=  None ]
        w=float(w) #to force floating division in x,y properties
        self.v = (x, y, w)
# - - -   P t . x y
    
    @property
    def xy(self):
        '''Return (x,y)
        '''
        w = self.v[2]
        return (self.v[0]/w, self.v[1]/w)
# - - -   P t . x
    @property
    def x(self):
        '''Return the abscissa.
        '''
        return self.v[0]/self.v[2]
# - - -   P t . y
    @property
    def y(self):
        '''Return the ordinate.
        '''
        return self.v[1]/self.v[2]

    def apply(self,f):
        """:return: Pt obtained by appying function f to x and y"""
        return Pt(f(self.x),f(self.y))

    def dist(self, other):
        '''Return the distance between self and other.
        '''
        (dx,dy) = (self-other).xy
        return sqrt ( dx*dx + dy*dy )
# - - -   P t . b e a r i n g

    def bearing(self, p):
        '''What is the bearing angle from self to p?
        '''
        return atan2(p.y-self.y, p.x-self.x)
# - - -   P t . r a d i a l

    def radial(self, d, bearing):
        '''Return the point at a given distance and bearing.
        '''
        return Pt(self.x + d*cos(bearing),
                  self.y + d*sin(bearing) )
# - - -   P t . t o P o l a r

    def toPolar(self):
        '''Convert to polar coordinates.
        '''
        x, y = self.xy
        return Polar(sqrt(x*x + y*y), atan2(y, x))
# - - -   P t . _ _ s t r _ _

    def __str__(self):
        '''Return a string representation of self.
        '''
        return "(%.4g, %.4g)" % self.xy

    def __repr__(self):
        return str(self)
# - - -   P t . _ _ a d d _ _

    def __add__(self, other):
        '''Add two points.
        '''
        return Pt(self.x+other.x, self.y+other.y)
# - - -   P t . _ _ s u b _ _

    def __sub__(self, other):
        '''Subtract two points.
        '''
        return Pt(self.x-other.x, self.y-other.y)
    
    def __mul__(self, scale):
        '''Multiply by scalar.
        '''
        return Pt((self.x, self.y, 1./scale))
    
    def __div__(self, scale):
        '''Multiply by scalar.
        '''
        return Pt((self.x, self.y, scale))

    def __cmp__(self, other):
        '''Compare two points.
        '''
        return cmp(self.xy, other.xy)

class Xform(object):
    '''Represents an arbitrary homogeneous coordinate transform.

      Exports:
        Xform(m):
          [ m is a 3x3 transform matrix as a array, or
            a sequence that array() will accept as a 3x3
            array ->
              return a new Xform instance representing that
              transform ]
        .apply(p):
          [ p is a Pt instance ->
              return a new Pt instance representing p transformed
              by self ]
        .invert(p):
          [ p is a Pt instance ->
              return a new Pt instance pp such that
              self.apply(pp) == p ]
        .inverse():
          [ return the inverse of self as an Xform instance ]
        .compose(t):
          [ t is an Xform instance ->
              return a new Xform representing the composition of
              self followed by t ]
        .offset():
          [ return the net offset that self will shift the origin,
            as a Pt instance ]
        .angle():
          [ return the net angle that self will rotate the unit
            vector from (0,0) to (1,1) ]
        .mag():
          [ return the net magnification that self will apply to the
            unit vector ]
        .__str__(self):
          [ return a string representation of self ]
      State/Invariants:
        self._m:
          [ a 3x3 array representing the argument passed
            to the constructor ]
        self._mInverse:
          [ the inverse of self._m or None ]
        self.__offset:
          [ the net translation of self or None ]
        self.__angle:
          [ the net rotation of self or None ]
        self._mag:
          [ the net uniform scaling of self or None ]
        ORIGIN:      [ the origin as a Pt instance ]
        UNIT:        [ a point 1.0 along the line x=y ]
    '''
    ORIGIN = Pt(0,0)
    UNIT = ORIGIN.radial(1.0, RAD_45)
# - - -   X f o r m . _ _ i n i t _ _

    def __init__ ( self, m ):
        '''Constructor.
        '''
        #-- 1 --
        # [ if the type of m is ndarray ->
        #     self._m  :=  m
        #   else if m is acceptable as an argument to
        #   array() ->
        #     self._m  :=  array(m)
        #   else -> raise Exception ]
        self._m = m
        #-- 2 --
        self._mInverse = None
# - - -   X f o r m . a p p l y

    def apply ( self, p ):
        '''Transform a point.
        '''
        #-- 1 --
        # [ pp  :=  a array representing the dot product of
        #           self._m and p.v ]
        pp = dot(self._m, p.v)

        #-- 2 --
        # [ return a Pt instance representing pp.v ]
        return Pt(pp)
# - - -   X f o r m . _ _ c a l l _ _

    def __call__(self, p):
        return self.apply(p)
# - - -   X f o r m . i n v e r t

    def invert ( self, p ):
        '''Return p transformed by the inverse of self, as a Pt.
        '''
        return self.inverse().apply ( p )
# - - -   X f o r m . i n v e r s e

    def inverse ( self ):
        '''Return the inverse transform as an Xform.
        '''
        #-- 1 --
        # [ if self._mInverse is None ->
        #     self._mInverse  :=  matrix inverse of self._m
        #   else -> I ]
        if self._mInverse is None:
            self._mInverse = linalg.inv ( self._m )

        #-- 2 --
        return Xform(self._mInverse)
# - - -   X f o r m . _ _ s t r _ _

    def __str__(self):
        '''Display self as a string
        '''
        #-- 1 --
        return ( "<Xform(xlate(%s), rotate(%.1fdeg), "
                 "mag(%.1f)>" %
                 (self.offset(), degrees(self.angle()), self.mag()) )

    def compose ( self, t2 ):
        '''Return the composition of two transforms.
        '''
        return Xform ( dot ( t2._m, self._m ) )
    
    def __mul__ ( self, other ):
        '''Implement '*'
        '''
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
        '''Return the net (uniform) scaling of this transform.
        '''
        return self(self.ORIGIN ).dist(self(self.UNIT))

        
# - - -   X l a t e

def Xlate(*p):
    '''Create a translation transform.
    '''
    #-- 1 --
    # [ dx  :=  first value from p
    #   dy  :=  second value from p ]
    dx, dy = argPair ( *p )

    #-- 2 --
    return Xform ( [ (1, 0, dx),
                     (0, 1, dy),
                     (0, 0, 1)  ] )
# - - -   X s c a l e

def Xscale(*p):
    '''Create a scaling transform.
    '''
    #-- 1 --
    # [ if p is a single value or single-valued iterable ->
    #     sx  :=  that value
    #     sy  :=  that value
    #   else ->
    #     sx  :=  the first value from p
    #     sy  :=  the second value from p ]
    sx, sy = argPair ( *p )

    #-- 2 --
    # [ return an Xform for scaling x by sx and scaling y by sy ]
    return Xform ( [ (sx, 0,  0),
                     (0,  sy, 0),
                     (0,  0,  1) ] )
# - - -   X r o t a t e

def Xrotate(theta):
    '''Create a rotation transform.
    '''
    #-- 1 --
    sint = sin(theta)
    cost = cos(theta)

    #-- 2 --
    return Xform ( [ (cost, -sint, 0),
                     (sint, cost,  0),
                     (0,    0,     1) ] )
# - - -   X r o t a r o u n d

def Xrotaround ( p, theta ):
    '''Rotation of theta radians around point p.
    '''
    #-- 1 --
    # [ t1  :=  an Xform that translates point p to the origin
    #   r  :=  an Xform that rotates theta radians around the origin
    #   t2  :=  an Xform that translates the origin to point p ]
    t1 = Xlate ( [ -v
                   for v in p.xy ] )
    r = Xrotate ( theta )
    t2 = Xlate ( p.xy )

    #-- 2 --
    # [ return an Xform instance representing t1, then r, then t2 ]
    return t1.compose(r).compose(t2)

# - - - - -   c l a s s   P o l a r

class Polar(object):
    '''Represents a point in polar coordinates.

      Exports:
        Polar(r, theta):
          [ r and theta are numbers ->
              return a new Polar instance representing radius r
              and angle theta ]
        .r, .theta:  [ as passed to constructor ]
        .toCartesian():
          [ return self in Cartesian coordinates as a Pt instance ]
        .__str__():
          [ return self as a string "(r, theta)" ]
    '''
# - - -   P o l a r . _ _ i n i t _ _

    def __init__(self, *p):
        '''Constructor
        '''
        self.r, self.theta = argPair(*p)
# - - -   P o l a r . t o C a r t e s i a n

    def toCartesian(self):
        '''Return self in rectangular coordinates as a Pt instance.
        '''
        return Pt(self.r * cos(self.theta),
                  self.r * sin(self.theta))
# - - -   P o l a r . _ _ s t r _ _

    def __str__(self):
        '''Return self as a string.
        '''
        return ( "(%.4g, %.4gd)" %
                 (self.r, degrees(self.theta)) )

# - - - - -   c l a s s   L i n e

class Line(object):
    '''Represents a geometric line.

      Exports:
        Line(a, b, c):
          [ a, b, and c are floats ->
              return a Line instance representing ax+by+c=0 ]
        .a, .b, .c:  [ as passed to constructor, read-only ]
        .__str__(self):   [ return self as a string ]
        .intersect(other):
          [ other is a Line instance ->
              if self and other intersect ->
                return the intersection as a Pt
              else -> raise ValueError ]
        Line.twoPoint(p1, p2):       # Static method
          [ p1 and p2 are Pt instances ->
              if p1 and p2 are distinct ->
                return a Line instance representing the line that
                intersects p1 and p2
              else -> raise ValueError ]
        Line.pointBearing(p, bears):   # Static method
          [ (p is a Pt instance) and
            (bears is a Cartesian bearing in radians) ->
              return the line through p at bearing (bears) ]
    '''
# - - -   L i n e . _ _ i n i t _ _

    def __init__(self, a, b, c):
        '''Constructor.
        '''
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
# - - -   L i n e . _ _ s t r _ _

    def __str__(self):
        '''Return a string representing self.
        '''
        return "%.4gx + %.4gy + %.4g = 0" % (self.a, self.b, self.c)
# - - -   L i n e . i n t e r s e c t

    def intersect(self, other):
        '''Where do lines self and other intersect?
        '''
        #-- 1 --
        # [ if self and other have the same slope ->
        #     raise ValueError
        #   else -> I ]
        if self.a * other.b == other.a * self.b:
            raise ValueError("Lines have the same slope.")
        #-- 2 --
        # [ x, y  :=  solution to the simultaneous linear equations
        #       (self.a * x + self.b * y = -self.c) and
        #       (other.a * x + other.b * y = -other.c) ]
        a = array ( ( (self.a, self.b), (other.a, other.b) ) )
        b = array ( (-self.c, -other.c) )
        x, y = linalg.solve(a,b)

        #-- 3 --
        return Pt(x, y)
# - - -   L i n e . t w o P o i n t

    @staticmethod
    def twoPoint(p1, p2):
        '''Find the equation of a line between two points.
        '''
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
# - - -   L i n e . p o i n t B e a r i n g

    @staticmethod
    def pointBearing(p, bears):
        '''Line through p at angle (bears).
        '''
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
    '''Process a pair of values passed in various ways.

      [ if len(p) is 2 ->
            return (p[0], p[1])
        else if p is a single non-iterable ->
          return (p[0], p[0])
        else if p is an iterable with two values ->
            return (p[0][0], p[0][1])
        else if p is an iterable with one value ->
            return (p[0][0], p[0][0])
    '''
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
    '''Normalize an angle in radians to [0, 2*pi)
    '''
    return theta % TWO_PI