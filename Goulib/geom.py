#!/usr/bin/python

"""
A module providing vector, matrix and quaternion operations for use in 2D and 3D graphics applications.

based on euclid http://code.google.com/p/pyeuclid
"""

__author__ = "Alex Holkner, Philippe Guglielmetti"
__copyright__ = "Copyright (c) 2006 Alex Holkner"
__license__ = "LGPL"
__credits__ = ['http://code.google.com/p/pyeuclid', 'http://www.nmt.edu/tcc/help/lang/python/examples/homcoord/']

__docformat__ = 'restructuredtext'
__version__ = '$Id$'
__revision__ = '$Revision$'

import operator, types
from math import *

precision = 1e-9 #for equality comparizons

class Vector2(object):
    """
   
    Two mutable vector types are available: *Vector2* and *Vector3*,
    for 2D and 3D vectors, respectively.  Vectors are assumed to hold
    floats, but most operations will also work if you use ints or longs
    instead.  Construct a vector in the obvious way::
    
        >>> Vector2(1.5, 2.0)
        Vector2(1.50, 2.00)
    
        >>> Vector3(1.0, 2.0, 3.0)
        Vector3(1.00, 2.00, 3.00)
    
    **Element access**
    
    Components may be accessed as attributes (examples that follow use
    *Vector3*, but all results are similar for *Vector2*, using only the *x*
    and *y* components)::
    
        >>> v = Vector3(1, 2, 3)
        >>> v.x
        1
        >>> v.y
        2
        >>> v.z
        3
    
    Vectors support the list interface via slicing::
    
        >>> v = Vector3(1, 2, 3)
        >>> len(v)
        3
        >>> v[0]
        1
        >>> v[:]
        (1, 2, 3)
    
    You can also "swizzle" the components (*a la* GLSL or Cg)::
    
        >>> v = Vector3(1, 2, 3)
        >>> v.xyz
        (1, 2, 3)
        >>> v.zx
        (3, 1)
        >>> v.zzzz
        (3, 3, 3, 3)
    
    All of the above accessors are also mutators[1]::
    
        >>> v = Vector3(1, 2, 3)
        >>> v.x = 5
        >>> v
        Vector3(5.00, 2.00, 3.00)
        >>> v[1:] = (10, 20)
        >>> v
        Vector3(5.00, 10.00, 20.00)
    
    [1] assignment via a swizzle (e.g., ``v.xyz = (1, 2, 3)``) is supported
    only if the ``_enable_swizzle_set`` variable is set.  This is disabled
    by default, as it impacts on the performance of ordinary attribute
    setting, and is slower than setting components sequentially anyway.
    
    **Operators**
    
    Addition and subtraction are supported via operator overloading (note 
    that in-place operators perform faster than those that create a new object)::
    
        >>> v1 = Vector3(1, 2, 3)
        >>> v2 = Vector3(4, 5, 6)
        >>> v1 + v2
        Vector3(5.00, 7.00, 9.00)
        >>> v1 -= v2
        >>> v1
        Vector3(-3.00, -3.00, -3.00)
    
    Multiplication and division can be performed with a scalar only::
    
        >>> Vector3(1, 2, 3) * 2
        Vector3(2.00, 4.00, 6.00)
        >>> v1 = Vector3(1., 2., 3.)
        >>> v1 /= 2
        >>> v1
        Vector3(0.50, 1.00, 1.50)
    
    The magnitude of a vector can be found with ``abs``::
    
        >>> v = Vector3(1., 2., 3.)
        >>> abs(v)
        3.7416573867739413
    
    A vector can be normalized in-place (note that the in-place method also
    returns ``self``, so you can chain it with further operators)::
    
        >>> v = Vector3(1., 2., 3.)
        >>> v.normalize()
        Vector3(0.27, 0.53, 0.80)
        >>> v
        Vector3(0.27, 0.53, 0.80)
    
    The following methods do *not* alter the original vector or their arguments:
    
    ``copy()``
        Returns a copy of the vector.  ``__copy__`` is also implemented.
    
    ``magnitude()``
        Returns the magnitude of the vector; equivalent to ``abs(v)``.  Example::
    
            >>> v = Vector3(1., 2., 3.)
            >>> v.magnitude()
            3.7416573867739413
    
    ``magnitude_squared()``
        Returns the sum of the squares of each component.  Useful for comparing
        the length of two vectors without the expensive square root operation.
        Example::
    
            >>> v = Vector3(1., 2., 3.)
            >>> v.magnitude_squared()
            14.0
    
    ``normalized()``
        Return a unit length vector in the same direction.  Note that this
        method differs from ``normalize`` in that it does not modify the
        vector in-place.  Example::
    
            >>> v = Vector3(1., 2., 3.)
            >>> v.normalized()
            Vector3(0.27, 0.53, 0.80)
            >>> v
            Vector3(1.00, 2.00, 3.00)
    
    ``dot(other)``
        Return the scalar "dot" product of two vectors.  Example::
    
            >>> v1 = Vector3(1., 2., 3.)
            >>> v2 = Vector3(4., 5., 6.)
            >>> v1.dot(v2)
            32.0
    
    ``cross()`` and ``cross(other)``
        Return the cross product of a vector (for **Vector2**), or the cross
        product of two vectors (for **Vector3**).  The return type is a
        vector.  Example::
    
            >>> v1 = Vector3(1., 2., 3.)
            >>> v2 = Vector3(4., 5., 6.)
            >>> v1.cross(v2)
            Vector3(-3.00, 6.00, -3.00)
    
        In two dimensions there can be no argument to ``cross``::
    
            >>> v1 = Vector2(1., 2.)
            >>> v1.cross()
            Vector2(2.00, -1.00)
    
    ``reflect(normal)``
        Return the vector reflected about the given normal.  In two dimensions,
        *normal* is the normal to a line, in three dimensions it is the normal
        to a plane.  The normal must have unit length.  Example::
    
            >>> v = Vector3(1., 2., 3.)
            >>> v.reflect(Vector3(0, 1, 0))
            Vector3(1.00, -2.00, 3.00)
            >>> v = Vector2(1., 2.)
            >>> v.reflect(Vector2(1, 0))
            Vector2(-1.00, 2.00)
    
    ``rotate_around(axes, theta)``
        For 3D vectors, return the vector rotated around axis by the angle theta.
    
            >>> v = Vector3(1., 2., 3.)
            >>> axes = Vector3(1.,1.,0)
            >>> v.rotate_around(axes,math.pi/4)
            Vector3(2.65, 0.35, 2.62)
    
    ``angle(other)``
        Return the angle between two vectors.
        
    ``project(other)``
        Return the projection (the component) of the vector on other.
    
    Tests for equality include comparing against other sequences::
    
        >>> v2 = Vector2(1, 2)
        >>> v2 == Vector2(3, 4)
        False
        >>> v2 != Vector2(1, 2)
        False
        >>> v2 == (1, 2)
        True
    
        >>> v3 = Vector3(1, 2, 3)
        >>> v3 == Vector3(3, 4, 5)
        False
        >>> v3 != Vector3(1, 2, 3)
        False
        >>> v3 == (1, 2, 3)
        True
    
    Vectors are not hashable, and hence cannot be put in sets nor used as
    dictionary keys::
    
        >>> {Vector2(): 0}
        Traceback (most recent call last):
            ...
        TypeError: unhashable type: 'Vector2'
    
        >>> {Vector3(): 0}
        Traceback (most recent call last):
            ...
        TypeError: unhashable type: 'Vector3'

    """

    def __init__ ( self, *args ):
        """Constructor.
        :param *args: x,y values
        """
        self.x,self.y=argPair(*args)
        
    @property
    def xy(self):
        """:return: tuple (x,y)"""
        return (self.x, self.y)
        
    def __copy__(self):
        return self.__class__(self.xy)

    copy = __copy__

    def __repr__(self):
        return '%s%s' % (self.__class__.__name__,self.xy)

    def __eq__(self, other):
        try: #quick
            if self.xy == other.xy : return True
        except: pass
        try:
            if self.x == other[0] and self.y == other[1]: return True
        except: pass
        return (self-Vector2(other)).mag()<precision

    def __ne__(self, other):
        return not self.__eq__(other)

    def __nonzero__(self):
        return self.x != 0 or self.y != 0

    def __len__(self):
        return 2

    def __iter__(self):
        return iter(self.xy)

    def __add__(self, other):
        if isinstance(other, Vector2):
            # Vector + Vector -> Vector
            # Vector + Point -> Point
            # Point + Point -> Vector
            if self.__class__ is other.__class__:
                _class = Vector2
            else:
                _class = Point2
            return _class(self.x + other.x,
                          self.y + other.y)
        else:
            return Vector2(self.x + other[0],
                           self.y + other[1])
    __radd__ = __add__

    def __iadd__(self, other):
        if isinstance(other, Vector2):
            self.x += other.x
            self.y += other.y
        else:
            self.x += other[0]
            self.y += other[1]
        return self

    def __sub__(self, other):
        if isinstance(other, Vector2):
            # Vector - Vector -> Vector
            # Vector - Point -> Point
            # Point - Point -> Vector
            if self.__class__ is other.__class__:
                _class = Vector2
            else:
                _class = Point2
            return _class(self.x - other.x,
                          self.y - other.y)
        else:
            assert hasattr(other, '__len__') and len(other) == 2
            return Vector2(self.x - other[0],
                           self.y - other[1])

   
    def __rsub__(self, other):
        if isinstance(other, Vector2):
            return Vector2(other.x - self.x,
                           other.y - self.y)
        else:
            assert hasattr(other, '__len__') and len(other) == 2
            return Vector2(other.x - self[0],
                           other.y - self[1])

    def __mul__(self, other):
        assert type(other) in (int, long, float)
        return Vector2(self.x * other,
                       self.y * other)

    __rmul__ = __mul__

    def __imul__(self, other):
        assert type(other) in (int, long, float)
        self.x *= other
        self.y *= other
        return self

    def __div__(self, other):
        assert type(other) in (int, long, float)
        return Vector2(operator.div(self.x, other),
                       operator.div(self.y, other))


    def __rdiv__(self, other):
        assert type(other) in (int, long, float)
        return Vector2(operator.div(other, self.x),
                       operator.div(other, self.y))

    def __floordiv__(self, other):
        assert type(other) in (int, long, float)
        return Vector2(operator.floordiv(self.x, other),
                       operator.floordiv(self.y, other))


    def __rfloordiv__(self, other):
        assert type(other) in (int, long, float)
        return Vector2(operator.floordiv(other, self.x),
                       operator.floordiv(other, self.y))

    def __truediv__(self, other):
        assert type(other) in (int, long, float)
        return Vector2(operator.truediv(self.x, other),
                       operator.truediv(self.y, other))


    def __rtruediv__(self, other):
        assert type(other) in (int, long, float)
        return Vector2(operator.truediv(other, self.x),
                       operator.truediv(other, self.y))
    
    def __neg__(self):
        return Vector2(-self.x,
                        -self.y)

    __pos__ = __copy__
    
    def __abs__(self):
        return sqrt(self.x ** 2 + self.y ** 2)

    mag = __abs__

    def mag2(self):
        return self.x ** 2 + self.y ** 2

    def normalize(self):
        d = self.mag()
        if d:
            self.x /= d
            self.y /= d
        return self

    def normalized(self):
        d = self.mag()
        if d:
            return Vector2(self.x / d, 
                           self.y / d)
        return self.copy()

    def dot(self, other):
        assert isinstance(other, Vector2)
        return self.x * other.x + self.y * other.y

    def cross(self):
        return Vector2(self.y, -self.x)

    def reflect(self, normal):
        # assume normal is normalized
        assert isinstance(normal, Vector2)
        d = 2 * (self.x * normal.x + self.y * normal.y)
        return Vector2(self.x - d * normal.x,
                       self.y - d * normal.y)

    def angle(self, other):
        """Return the angle to the vector other"""
        if other:
            return acos(self.dot(other) / (self.mag()*other.mag()))
        else:
            return atan2(self.y,self.x)

    def project(self, other):
        """Return one vector projected on the vector other"""
        n = other.normalized()
        return self.dot(n)*n

class Vector3(object):

    def __init__(self, *args):
        """Constructor.
        :param *args: x,y,z values
        """
        if len(args) == 1:
            value = args[0]
            assert(len(value) == 3)
            x, y, z = value
        else:
            x, y,z = args
        self.x = x
        self.y = y 
        self.z = z
        
    @property
    def xyz(self):
        """:return: tuple (x,y,z)"""
        return (self.x, self.y, self.z)

    def __copy__(self):
        return self.__class__(self.xyz)

    copy = __copy__

    def __repr__(self):
        return '%s%s' % (self.__class__.__name__,self.xyz)

    def __eq__(self, other):
        if isinstance(other, Vector3):
            return self.xyz == other.xyz
        else:
            assert hasattr(other, '__len__') and len(other) == 3
            return self.x == other[0] and \
                   self.y == other[1] and \
                   self.z == other[2]

    def __ne__(self, other):
        return not self.__eq__(other)

    def __nonzero__(self):
        return self.x != 0 or self.y != 0 or self.z != 0

    def __len__(self):
        return 3

    def __iter__(self):
        return iter(self.xyz)

    def __add__(self, other):
        if isinstance(other, Vector3):
            # Vector + Vector -> Vector
            # Vector + Point -> Point
            # Point + Point -> Vector
            if self.__class__ is other.__class__:
                _class = Vector3
            else:
                _class = Point3
            return _class(self.x + other.x,
                          self.y + other.y,
                          self.z + other.z)
        else:
            return Vector3(self.x + other[0],
                           self.y + other[1],
                           self.z + other[2])
    __radd__ = __add__

    def __iadd__(self, other):
        if isinstance(other, Vector3):
            self.x += other.x
            self.y += other.y
            self.z += other.z
        else:
            self.x += other[0]
            self.y += other[1]
            self.z += other[2]
        return self

    def __sub__(self, other):
        if isinstance(other, Vector3):
            # Vector - Vector -> Vector
            # Vector - Point -> Point
            # Point - Point -> Vector
            if self.__class__ is other.__class__:
                _class = Vector3
            else:
                _class = Point3
            return Vector3(self.x - other.x,
                           self.y - other.y,
                           self.z - other.z)
        else:
            assert hasattr(other, '__len__') and len(other) == 3
            return Vector3(self.x - other[0],
                           self.y - other[1],
                           self.z - other[2])

   
    def __rsub__(self, other):
        if isinstance(other, Vector3):
            return Vector3(other.x - self.x,
                           other.y - self.y,
                           other.z - self.z)
        else:
            assert hasattr(other, '__len__') and len(other) == 3
            return Vector3(other.x - self[0],
                           other.y - self[1],
                           other.z - self[2])

    def __mul__(self, other):
        if isinstance(other, Vector3):
            # TODO component-wise mul/div in-place and on Vector2; docs.
            if self.__class__ is Point3 or other.__class__ is Point3:
                _class = Point3
            else:
                _class = Vector3
            return _class(self.x * other.x,
                          self.y * other.y,
                          self.z * other.z)
        else: 
            assert type(other) in (int, long, float)
            return Vector3(self.x * other,
                           self.y * other,
                           self.z * other)

    __rmul__ = __mul__

    def __imul__(self, other):
        assert type(other) in (int, long, float)
        self.x *= other
        self.y *= other
        self.z *= other
        return self

    def __div__(self, other):
        assert type(other) in (int, long, float)
        return Vector3(operator.div(self.x, other),
                       operator.div(self.y, other),
                       operator.div(self.z, other))


    def __rdiv__(self, other):
        assert type(other) in (int, long, float)
        return Vector3(operator.div(other, self.x),
                       operator.div(other, self.y),
                       operator.div(other, self.z))

    def __floordiv__(self, other):
        assert type(other) in (int, long, float)
        return Vector3(operator.floordiv(self.x, other),
                       operator.floordiv(self.y, other),
                       operator.floordiv(self.z, other))


    def __rfloordiv__(self, other):
        assert type(other) in (int, long, float)
        return Vector3(operator.floordiv(other, self.x),
                       operator.floordiv(other, self.y),
                       operator.floordiv(other, self.z))

    def __truediv__(self, other):
        assert type(other) in (int, long, float)
        return Vector3(operator.truediv(self.x, other),
                       operator.truediv(self.y, other),
                       operator.truediv(self.z, other))


    def __rtruediv__(self, other):
        assert type(other) in (int, long, float)
        return Vector3(operator.truediv(other, self.x),
                       operator.truediv(other, self.y),
                       operator.truediv(other, self.z))
    
    def __neg__(self):
        return Vector3(-self.x,
                        -self.y,
                        -self.z)

    __pos__ = __copy__
    
    def __abs__(self):
        return sqrt(self.x ** 2 + \
                         self.y ** 2 + \
                         self.z ** 2)

    mag = __abs__

    def mag2(self):
        return self.x ** 2 + \
               self.y ** 2 + \
               self.z ** 2

    def normalize(self):
        d = self.mag()
        if d:
            self.x /= d
            self.y /= d
            self.z /= d
        return self

    def normalized(self):
        d = self.mag()
        if d:
            return Vector3(self.x / d, 
                           self.y / d, 
                           self.z / d)
        return self.copy()

    def dot(self, other):
        assert isinstance(other, Vector3)
        return self.x * other.x + \
               self.y * other.y + \
               self.z * other.z

    def cross(self, other):
        assert isinstance(other, Vector3)
        return Vector3(self.y * other.z - self.z * other.y,
                       -self.x * other.z + self.z * other.x,
                       self.x * other.y - self.y * other.x)

    def reflect(self, normal):
        # assume normal is normalized
        assert isinstance(normal, Vector3)
        d = 2 * (self.x * normal.x + self.y * normal.y + self.z * normal.z)
        return Vector3(self.x - d * normal.x,
                       self.y - d * normal.y,
                       self.z - d * normal.z)

    def rotate_around(self, axis, theta):
        """Return the vector rotated around axis through angle theta. Right hand rule applies"""

        # Adapted from equations published by Glenn Murray.
        # http://inside.mines.edu/~gmurray/ArbitraryAxisRotation/ArbitraryAxisRotation.html
        x, y, z = self.x, self.y,self.z
        u, v, w = axis.x, axis.y, axis.z

        # Extracted common factors for simplicity and efficiency
        r2 = u**2 + v**2 + w**2
        r = sqrt(r2)
        ct = cos(theta)
        st = sin(theta) / r
        dt = (u*x + v*y + w*z) * (1 - ct) / r2
        return Vector3((u * dt + x * ct + (-w * y + v * z) * st),
                       (v * dt + y * ct + ( w * x - u * z) * st),
                       (w * dt + z * ct + (-v * x + u * y) * st))

    def angle(self, other):
        """Return the angle to the vector other"""
        return acos(self.dot(other) / (self.mag()*other.mag()))

    def project(self, other):
        """Return one vector projected on the vector other"""
        n = other.normalized()
        return self.dot(n)*n


class Matrix3(object):
    """
    Two matrix classes are supplied, *Matrix3*, a 3x3 matrix for working with 2D
    affine transformations, and *Matrix4*, a 4x4 matrix for working with 3D
    affine transformations.
    
    The default constructor intializes the matrix to the identity::
    
        >>> Matrix3()
        Matrix3([    1.00     0.00     0.00
                     0.00     1.00     0.00
                     0.00     0.00     1.00])
        >>> Matrix4()
        Matrix4([    1.00     0.00     0.00     0.00
                     0.00     1.00     0.00     0.00
                     0.00     0.00     1.00     0.00
                     0.00     0.00     0.00     1.00])
    
    **Element access**
    
    Internally each matrix is stored as a set of attributes named ``a`` to ``p``.
    The layout for Matrix3 is::
    
        # a b c 
        # e f g 
        # i j k 
    
    and for Matrix4::
    
        # a b c d
        # e f g h
        # i j k l
        # m n o p
    
    If you wish to set or retrieve a number of elements at once, you can
    do so with a slice::
    
        >>> m = Matrix4()
        >>> m[:]
        [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0]
        >>> m[12:15] = (5, 5, 5)
        >>> m
        Matrix4([    1.00     0.00     0.00     5.00
                     0.00     1.00     0.00     5.00
                     0.00     0.00     1.00     5.00
                     0.00     0.00     0.00     1.00])
    
    Note that slices operate in column-major order, which makes them
    suitable for working directly with OpenGL's ``glLoadMatrix`` and
    ``glGetFloatv`` functions.
    
    **Class constructors**
    
    There are class constructors for the most common types of transform.
    
    ``new_identity``
        Equivalent to the default constructor.  Example::
    
            >>> m = Matrix4.new_identity()
            >>> m
            Matrix4([    1.00     0.00     0.00     0.00
                         0.00     1.00     0.00     0.00
                         0.00     0.00     1.00     0.00
                         0.00     0.00     0.00     1.00])
    
    ``new_scale(x, y)`` and ``new_scale(x, y, z)``
        The former is defined on **Matrix3**, the latter on **Matrix4**.
        Equivalent to the OpenGL call ``glScalef``.
        Example::
    
            >>> m = Matrix4.new_scale(2.0, 3.0, 4.0)
            >>> m
            Matrix4([    2.00     0.00     0.00     0.00
                         0.00     3.00     0.00     0.00
                         0.00     0.00     4.00     0.00
                         0.00     0.00     0.00     1.00])
        
    ``new_translate(x, y)`` and ``new_translate(x, y, z)``
        The former is defined on **Matrix3**, the latter on **Matrix4**.
        Equivalent to the OpenGL call ``glTranslatef``.
        Example::
    
            >>> m = Matrix4.new_translate(3.0, 4.0, 5.0)
            >>> m
            Matrix4([    1.00     0.00     0.00     3.00
                         0.00     1.00     0.00     4.00
                         0.00     0.00     1.00     5.00
                         0.00     0.00     0.00     1.00])
    
    ``new_rotate(angle)``
        Create a **Matrix3** for a rotation around the origin.  *angle* is
        specified in radians, anti-clockwise.  This is not implemented in
        **Matrix4** (see below for equivalent methods).
        Example::
    
            >>> import math
            >>> m = Matrix3.new_rotate(math.pi / 2)
            >>> m
            Matrix3([    0.00    -1.00     0.00
                         1.00     0.00     0.00
                         0.00     0.00     1.00])
    
    The following constructors are defined for **Matrix4** only.
    
    ``new``
        Construct a matrix with 16 values in column-major order.
    
    ``new_rotatex(angle)``, ``new_rotatey(angle)``, ``new_rotatez(angle)``
        Create a **Matrix4** for a rotation around the X, Y or Z axis, respectively.
        *angle* is specified in radians.  Example::
    
            >>> m = Matrix4.new_rotatex(math.pi / 2)
            >>> m
            Matrix4([    1.00     0.00     0.00     0.00
                         0.00     0.00    -1.00     0.00
                         0.00     1.00     0.00     0.00
                         0.00     0.00     0.00     1.00])
    
    ``new_rotate_axis(angle, axis)``
        Create a **Matrix4** for a rotation around the given axis.  *angle*
        is specified in radians, and *axis* must be an instance of **Vector3**.
        It is not necessary to normalize the axis.  Example::
    
            >>> m = Matrix4.new_rotate_axis(math.pi / 2, Vector3(1.0, 0.0, 0.0))
            >>> m        
            Matrix4([    1.00     0.00     0.00     0.00
                         0.00     0.00    -1.00     0.00
                         0.00     1.00     0.00     0.00
                         0.00     0.00     0.00     1.00])
    
    ``new_rotate_euler(heading, attitude, bank)``
        Create a **Matrix4** for the given Euler rotation.  *heading* is a rotation
        around the Y axis, *attitude* around the X axis and *bank* around the Z
        axis.  All rotations are performed simultaneously, so this method avoids
        "gimbal lock" and is the usual method for implemented 3D rotations in a
        game.  Example::
    
            >>> m = Matrix4.new_rotate_euler(math.pi / 2, math.pi / 2, 0.0)
            >>> m
            Matrix4([    0.00    -0.00     1.00     0.00
                         1.00     0.00    -0.00     0.00
                        -0.00     1.00     0.00     0.00
                         0.00     0.00     0.00     1.00])
    
    ``new_perspective(fov_y, aspect, near, far)``
        Create a **Matrix4** for projection onto the 2D viewing plane.  This
        method is equivalent to the OpenGL call ``gluPerspective``.  *fov_y* is
        the view angle in the Y direction, in radians.  *aspect* is the aspect
        ration *width* / *height* of the viewing plane.  *near* and *far* are
        the distance to the near and far clipping planes.  They must be
        positive and non-zero.  Example::
    
            >>> m = Matrix4.new_perspective(math.pi / 2, 1024.0 / 768, 1.0, 100.0)
            >>> m
            Matrix4([    0.75     0.00     0.00     0.00
                         0.00     1.00     0.00     0.00
                         0.00     0.00    -1.02    -2.02
                         0.00     0.00    -1.00     0.00])
    
    **Operators**
    
    Matrices of the same dimension may be multiplied to give a new matrix.
    For example, to create a transform which translates and scales::
    
        >>> m1 = Matrix3.new_translate(5.0, 6.0)
        >>> m2 = Matrix3.new_scale(1.0, 2.0)
        >>> m1 * m2
        Matrix3([    1.00     0.00     5.00
                     0.00     2.00     6.00
                     0.00     0.00     1.00])
    
    Note that multiplication is not commutative (the order that you apply
    transforms matters)::
    
        >>> m2 * m1
        Matrix3([    1.00     0.00     5.00
                     0.00     2.00    12.00
                     0.00     0.00     1.00])
    
    In-place multiplication is also permitted (and optimised)::
    
        >>> m1 *= m2
        >>> m1
        Matrix3([    1.00     0.00     5.00
                     0.00     2.00     6.00
                     0.00     0.00     1.00])
    
    Multiplying a matrix by a vector returns a vector, and is used to
    transform a vector::
    
        >>> m1 = Matrix3.new_rotate(math.pi / 2)
        >>> m1 * Vector2(1.0, 1.0)
        Vector2(-1.00, 1.00)
    
    Note that translations have no effect on vectors.  They do affect
    points, however::
    
        >>> m1 = Matrix3.new_translate(5.0, 6.0)
        >>> m1 * Vector2(1.0, 2.0)
        Vector2(1.00, 2.00)
        >>> m1 * Point2(1.0, 2.0)
        Point2(6.00, 8.00)
    
    Multiplication is currently incorrect between matrices and vectors -- the
    projection component is ignored.  Use the **Matrix4.transform** method
    instead.
    
    Matrix4 also defines **transpose** (in-place), **transposed** (functional),
    **determinant** and **inverse** (functional) methods.
    
    A **Matrix3** can be multiplied with a **Vector2** or any of the 2D geometry
    objects (**Point2**, **Line2**, **Circle**, etc).  
    
    A **Matrix4** can be multiplied with a **Vector3** or any of the 3D geometry
    objects (**Point3**, **Line3**, **Sphere**, etc).
    
    For convenience, each of the matrix constructors are also available as
    in-place operators.  For example, instead of writing::
    
        >>> m1 = Matrix3.new_translate(5.0, 6.0)
        >>> m2 = Matrix3.new_scale(1.0, 2.0)
        >>> m1 *= m2
    
    you can apply the scale directly to *m1*::
    
        >>> m1 = Matrix3.new_translate(5.0, 6.0)
        >>> m1.scale(1.0, 2.0)
        Matrix3([    1.00     0.00     5.00
                     0.00     2.00     6.00
                     0.00     0.00     1.00])
        >>> m1
        Matrix3([    1.00     0.00     5.00
                     0.00     2.00     6.00
                     0.00     0.00     1.00])
    
    Note that these methods operate in-place (they modify the original matrix),
    and they also return themselves as a result.  This allows you to chain
    transforms together directly::
    
        >>> Matrix3().translate(1.0, 2.0).rotate(math.pi / 2).scale(4.0, 4.0)
        Matrix3([    0.00    -4.00     1.00
                     4.00     0.00     2.00
                     0.00     0.00     1.00])
    
    All constructors have an equivalent in-place method.  For **Matrix3**, they
    are ``identity``, ``translate``, ``scale`` and ``rotate``.  For **Matrix4**,
    they are ``identity``, ``translate``, ``scale``, ``rotatex``, ``rotatey``, 
    ``rotatez``, ``rotate_axis`` and ``rotate_euler``.  Both **Matrix3** and
    **Matrix4** also have an in-place ``transpose`` method.
    
    The ``copy`` method is also implemented in both matrix classes and
    behaves in the obvious way.
    """

    def __init__(self):
        self.identity()

    def __copy__(self):
        M = Matrix3()
        M.a = self.a
        M.b = self.b
        M.c = self.c
        M.e = self.e 
        M.f = self.f
        M.g = self.g
        M.i = self.i
        M.j = self.j
        M.k = self.k
        return M

    copy = __copy__
    def __repr__(self):
        return ('%s([%g %g %g\n'  \
                '         %g %g %g\n'  \
                '         %g %g %g])') \
                % (self.__class__.__name__,self.a, self.b, self.c,
                   self.e, self.f, self.g,
                   self.i, self.j, self.k)

    def __getitem__(self, key):
        return [self.a, self.e, self.i,
                self.b, self.f, self.j,
                self.c, self.g, self.k][key]

    def __setitem__(self, key, value):
        L = self[:]
        L[key] = value
        (self.a, self.e, self.i,
         self.b, self.f, self.j,
         self.c, self.g, self.k) = L

    def __mul__(self, other):
        if isinstance(other, Matrix3):
            # Caching repeatedly accessed attributes in local variables
            # apparently increases performance by 20%.  Attrib: Will McGugan.
            Aa = self.a
            Ab = self.b
            Ac = self.c
            Ae = self.e
            Af = self.f
            Ag = self.g
            Ai = self.i
            Aj = self.j
            Ak = self.k
            Ba = other.a
            Bb = other.b
            Bc = other.c
            Be = other.e
            Bf = other.f
            Bg = other.g
            Bi = other.i
            Bj = other.j
            Bk = other.k
            C = Matrix3()
            C.a = Aa * Ba + Ab * Be + Ac * Bi
            C.b = Aa * Bb + Ab * Bf + Ac * Bj
            C.c = Aa * Bc + Ab * Bg + Ac * Bk
            C.e = Ae * Ba + Af * Be + Ag * Bi
            C.f = Ae * Bb + Af * Bf + Ag * Bj
            C.g = Ae * Bc + Af * Bg + Ag * Bk
            C.i = Ai * Ba + Aj * Be + Ak * Bi
            C.j = Ai * Bb + Aj * Bf + Ak * Bj
            C.k = Ai * Bc + Aj * Bg + Ak * Bk
            return C
        elif isinstance(other, Point2):
            A = self
            B = other
            P = Point2(0, 0)
            P.x = A.a * B.x + A.b * B.y + A.c
            P.y = A.e * B.x + A.f * B.y + A.g
            return P
        elif isinstance(other, Vector2):
            A = self
            B = other
            V = Vector2(0, 0)
            V.x = A.a * B.x + A.b * B.y 
            V.y = A.e * B.x + A.f * B.y 
            return V
        else:
            other = other.copy()
            other._apply_transform(self)
            return other
        
    def __call__(self,other):
        return self*other

    def __imul__(self, other):
        assert isinstance(other, Matrix3)
        # Cache attributes in local vars (see Matrix3.__mul__).
        Aa = self.a
        Ab = self.b
        Ac = self.c
        Ae = self.e
        Af = self.f
        Ag = self.g
        Ai = self.i
        Aj = self.j
        Ak = self.k
        Ba = other.a
        Bb = other.b
        Bc = other.c
        Be = other.e
        Bf = other.f
        Bg = other.g
        Bi = other.i
        Bj = other.j
        Bk = other.k
        self.a = Aa * Ba + Ab * Be + Ac * Bi
        self.b = Aa * Bb + Ab * Bf + Ac * Bj
        self.c = Aa * Bc + Ab * Bg + Ac * Bk
        self.e = Ae * Ba + Af * Be + Ag * Bi
        self.f = Ae * Bb + Af * Bf + Ag * Bj
        self.g = Ae * Bc + Af * Bg + Ag * Bk
        self.i = Ai * Ba + Aj * Be + Ak * Bi
        self.j = Ai * Bb + Aj * Bf + Ak * Bj
        self.k = Ai * Bc + Aj * Bg + Ak * Bk
        return self

    def identity(self):
        self.a = self.f = self.k = 1.
        self.b = self.c = self.e = self.g = self.i = self.j = 0
        return self

    def scale(self, x, y=None):
        if y is None: y=x
        return Matrix3.new_scale(x, y)*self
    
    def offset(self):
        return self*Point2(0,0)

    def angle(self,angle):
        """
        :param angle: angle in radians of a unit vector starting at origin
        :return: float bearing in radians of the transformed vector
        """
        v=self*Polar(1.0,angle)
        return atan2(v.y,v.x)
    
    def mag(self,v=None):
        """Return the net (uniform) scaling of this transform.
        """
        if not v:
            v=Vector2(1,1)
        return (self*v).mag()/v.mag()

    def translate(self, *args ):
        """
        :param *args: x,y values
        """
        x,y=argPair(*args)
        return Matrix3.new_translate(x,y)*self

    def rotate(self, angle):
        return Matrix3.new_rotate(angle)*self

    # Static constructors
    def new_identity(cls):
        self = cls()
        return self
    new_identity = classmethod(new_identity)

    def new_scale(cls, x, y):
        self = cls()
        self.a = x
        self.f = y
        return self
    new_scale = classmethod(new_scale)

    def new_translate(cls, x, y):
        self = cls()
        self.c = x
        self.g = y
        return self
    new_translate = classmethod(new_translate)

    def new_rotate(cls, angle):
        self = cls()
        s = sin(angle)
        c = cos(angle)
        self.a = self.f = c
        self.b = -s
        self.e = s
        return self
    new_rotate = classmethod(new_rotate)

    def determinant(self):
        return (self.a*self.f*self.k 
                + self.b*self.g*self.i 
                + self.c*self.e*self.j 
                - self.a*self.g*self.j 
                - self.b*self.e*self.k 
                - self.c*self.f*self.i)

    def inverse(self):
        tmp = Matrix3()
        d = self.determinant()

        if abs(d) < 0.001:
            # No inverse, return identity
            return tmp
        else:
            d = 1.0 / d

            tmp.a = d * (self.f*self.k - self.g*self.j)
            tmp.b = d * (self.c*self.j - self.b*self.k)
            tmp.c = d * (self.b*self.g - self.c*self.f)
            tmp.e = d * (self.g*self.i - self.e*self.k)
            tmp.f = d * (self.a*self.k - self.c*self.i)
            tmp.g = d * (self.c*self.e - self.a*self.g)
            tmp.i = d * (self.e*self.j - self.f*self.i)
            tmp.j = d * (self.b*self.i - self.a*self.j)
            tmp.k = d * (self.a*self.f - self.b*self.e)

            return tmp

# a b c d
# e f g h
# i j k l
# m n o p

class Matrix4(object):

    def __init__(self):
        self.identity()

    def __copy__(self):
        M = Matrix4()
        M.a = self.a
        M.b = self.b
        M.c = self.c
        M.d = self.d
        M.e = self.e 
        M.f = self.f
        M.g = self.g
        M.h = self.h
        M.i = self.i
        M.j = self.j
        M.k = self.k
        M.l = self.l
        M.m = self.m
        M.n = self.n
        M.o = self.o
        M.p = self.p
        return M

    copy = __copy__


    def __repr__(self):
        return ('%s([%g %g %g %g\n'  \
                '         %g %g %g %g\n'  \
                '         %g %g %g %g\n'  \
                '         %g %g %g %g])') \
                % (self.__class__.__name__,self.a, self.b, self.c, self.d,
                   self.e, self.f, self.g, self.h,
                   self.i, self.j, self.k, self.l,
                   self.m, self.n, self.o, self.p)

    def __getitem__(self, key):
        return [self.a, self.e, self.i, self.m,
                self.b, self.f, self.j, self.n,
                self.c, self.g, self.k, self.o,
                self.d, self.h, self.l, self.p][key]

    def __setitem__(self, key, value):
        L = self[:]
        L[key] = value
        (self.a, self.e, self.i, self.m,
         self.b, self.f, self.j, self.n,
         self.c, self.g, self.k, self.o,
         self.d, self.h, self.l, self.p) = L

    def __mul__(self, other):
        if isinstance(other, Matrix4):
            # Cache attributes in local vars (see Matrix3.__mul__).
            Aa = self.a
            Ab = self.b
            Ac = self.c
            Ad = self.d
            Ae = self.e
            Af = self.f
            Ag = self.g
            Ah = self.h
            Ai = self.i
            Aj = self.j
            Ak = self.k
            Al = self.l
            Am = self.m
            An = self.n
            Ao = self.o
            Ap = self.p
            Ba = other.a
            Bb = other.b
            Bc = other.c
            Bd = other.d
            Be = other.e
            Bf = other.f
            Bg = other.g
            Bh = other.h
            Bi = other.i
            Bj = other.j
            Bk = other.k
            Bl = other.l
            Bm = other.m
            Bn = other.n
            Bo = other.o
            Bp = other.p
            C = Matrix4()
            C.a = Aa * Ba + Ab * Be + Ac * Bi + Ad * Bm
            C.b = Aa * Bb + Ab * Bf + Ac * Bj + Ad * Bn
            C.c = Aa * Bc + Ab * Bg + Ac * Bk + Ad * Bo
            C.d = Aa * Bd + Ab * Bh + Ac * Bl + Ad * Bp
            C.e = Ae * Ba + Af * Be + Ag * Bi + Ah * Bm
            C.f = Ae * Bb + Af * Bf + Ag * Bj + Ah * Bn
            C.g = Ae * Bc + Af * Bg + Ag * Bk + Ah * Bo
            C.h = Ae * Bd + Af * Bh + Ag * Bl + Ah * Bp
            C.i = Ai * Ba + Aj * Be + Ak * Bi + Al * Bm
            C.j = Ai * Bb + Aj * Bf + Ak * Bj + Al * Bn
            C.k = Ai * Bc + Aj * Bg + Ak * Bk + Al * Bo
            C.l = Ai * Bd + Aj * Bh + Ak * Bl + Al * Bp
            C.m = Am * Ba + An * Be + Ao * Bi + Ap * Bm
            C.n = Am * Bb + An * Bf + Ao * Bj + Ap * Bn
            C.o = Am * Bc + An * Bg + Ao * Bk + Ap * Bo
            C.p = Am * Bd + An * Bh + Ao * Bl + Ap * Bp
            return C
        elif isinstance(other, Point3):
            A = self
            B = other
            P = Point3(0, 0, 0)
            P.x = A.a * B.x + A.b * B.y + A.c * B.z + A.d
            P.y = A.e * B.x + A.f * B.y + A.g * B.z + A.h
            P.z = A.i * B.x + A.j * B.y + A.k * B.z + A.l
            return P
        elif isinstance(other, Vector3):
            A = self
            B = other
            V = Vector3(0, 0, 0)
            V.x = A.a * B.x + A.b * B.y + A.c * B.z
            V.y = A.e * B.x + A.f * B.y + A.g * B.z
            V.z = A.i * B.x + A.j * B.y + A.k * B.z
            return V
        else:
            other = other.copy()
            other._apply_transform(self)
            return other
        
    def __call__(self,other):
        return self*other

    def __imul__(self, other):
        assert isinstance(other, Matrix4)
        # Cache attributes in local vars (see Matrix3.__mul__).
        Aa = self.a
        Ab = self.b
        Ac = self.c
        Ad = self.d
        Ae = self.e
        Af = self.f
        Ag = self.g
        Ah = self.h
        Ai = self.i
        Aj = self.j
        Ak = self.k
        Al = self.l
        Am = self.m
        An = self.n
        Ao = self.o
        Ap = self.p
        Ba = other.a
        Bb = other.b
        Bc = other.c
        Bd = other.d
        Be = other.e
        Bf = other.f
        Bg = other.g
        Bh = other.h
        Bi = other.i
        Bj = other.j
        Bk = other.k
        Bl = other.l
        Bm = other.m
        Bn = other.n
        Bo = other.o
        Bp = other.p
        self.a = Aa * Ba + Ab * Be + Ac * Bi + Ad * Bm
        self.b = Aa * Bb + Ab * Bf + Ac * Bj + Ad * Bn
        self.c = Aa * Bc + Ab * Bg + Ac * Bk + Ad * Bo
        self.d = Aa * Bd + Ab * Bh + Ac * Bl + Ad * Bp
        self.e = Ae * Ba + Af * Be + Ag * Bi + Ah * Bm
        self.f = Ae * Bb + Af * Bf + Ag * Bj + Ah * Bn
        self.g = Ae * Bc + Af * Bg + Ag * Bk + Ah * Bo
        self.h = Ae * Bd + Af * Bh + Ag * Bl + Ah * Bp
        self.i = Ai * Ba + Aj * Be + Ak * Bi + Al * Bm
        self.j = Ai * Bb + Aj * Bf + Ak * Bj + Al * Bn
        self.k = Ai * Bc + Aj * Bg + Ak * Bk + Al * Bo
        self.l = Ai * Bd + Aj * Bh + Ak * Bl + Al * Bp
        self.m = Am * Ba + An * Be + Ao * Bi + Ap * Bm
        self.n = Am * Bb + An * Bf + Ao * Bj + Ap * Bn
        self.o = Am * Bc + An * Bg + Ao * Bk + Ap * Bo
        self.p = Am * Bd + An * Bh + Ao * Bl + Ap * Bp
        return self

    def transform(self, other):
        A = self
        B = other
        P = Point3(0, 0, 0)
        P.x = A.a * B.x + A.b * B.y + A.c * B.z + A.d
        P.y = A.e * B.x + A.f * B.y + A.g * B.z + A.h
        P.z = A.i * B.x + A.j * B.y + A.k * B.z + A.l
        w =   A.m * B.x + A.n * B.y + A.o * B.z + A.p
        if w != 0:
            P.x /= w
            P.y /= w
            P.z /= w
        return P

    def identity(self):
        self.a = self.f = self.k = self.p = 1.
        self.b = self.c = self.d = self.e = self.g = self.h = \
        self.i = self.j = self.l = self.m = self.n = self.o = 0
        return self

    def scale(self, x, y, z):
        self *= Matrix4.new_scale(x, y, z)
        return self

    def translate(self, x, y, z):
        self *= Matrix4.new_translate(x, y, z)
        return self 

    def rotatex(self, angle):
        self *= Matrix4.new_rotatex(angle)
        return self

    def rotatey(self, angle):
        self *= Matrix4.new_rotatey(angle)
        return self

    def rotatez(self, angle):
        self *= Matrix4.new_rotatez(angle)
        return self

    def rotate_axis(self, angle, axis):
        self *= Matrix4.new_rotate_axis(angle, axis)
        return self

    def rotate_euler(self, heading, attitude, bank):
        self *= Matrix4.new_rotate_euler(heading, attitude, bank)
        return self

    def rotate_triple_axis(self, x, y, z):
        self *= Matrix4.new_rotate_triple_axis(x, y, z)
        return self

    def transpose(self):
        (self.a, self.e, self.i, self.m,
         self.b, self.f, self.j, self.n,
         self.c, self.g, self.k, self.o,
         self.d, self.h, self.l, self.p) = \
        (self.a, self.b, self.c, self.d,
         self.e, self.f, self.g, self.h,
         self.i, self.j, self.k, self.l,
         self.m, self.n, self.o, self.p)

    def transposed(self):
        M = self.copy()
        M.transpose()
        return M

    # Static constructors
    def new(cls, *values):
        M = cls()
        M[:] = values
        return M
    new = classmethod(new)

    def new_identity(cls):
        self = cls()
        return self
    new_identity = classmethod(new_identity)

    def new_scale(cls, x, y, z):
        self = cls()
        self.a = x
        self.f = y
        self.k = z
        return self
    new_scale = classmethod(new_scale)

    def new_translate(cls, x, y, z):
        self = cls()
        self.d = x
        self.h = y
        self.l = z
        return self
    new_translate = classmethod(new_translate)

    def new_rotatex(cls, angle):
        self = cls()
        s = sin(angle)
        c = cos(angle)
        self.f = self.k = c
        self.g = -s
        self.j = s
        return self
    new_rotatex = classmethod(new_rotatex)

    def new_rotatey(cls, angle):
        self = cls()
        s = sin(angle)
        c = cos(angle)
        self.a = self.k = c
        self.c = s
        self.i = -s
        return self    
    new_rotatey = classmethod(new_rotatey)
    
    def new_rotatez(cls, angle):
        self = cls()
        s = sin(angle)
        c = cos(angle)
        self.a = self.f = c
        self.b = -s
        self.e = s
        return self
    new_rotatez = classmethod(new_rotatez)

    def new_rotate_axis(cls, angle, axis):
        assert(isinstance(axis, Vector3))
        vector = axis.normalized()
        x = vector.x
        y = vector.y
        z = vector.z

        self = cls()
        s = sin(angle)
        c = cos(angle)
        c1 = 1. - c
        
        # from the glRotate man page
        self.a = x * x * c1 + c
        self.b = x * y * c1 - z * s
        self.c = x * z * c1 + y * s
        self.e = y * x * c1 + z * s
        self.f = y * y * c1 + c
        self.g = y * z * c1 - x * s
        self.i = x * z * c1 - y * s
        self.j = y * z * c1 + x * s
        self.k = z * z * c1 + c
        return self
    new_rotate_axis = classmethod(new_rotate_axis)

    def new_rotate_euler(cls, heading, attitude, bank):
        # from http://www.euclideanspace.com/
        ch = cos(heading)
        sh = sin(heading)
        ca = cos(attitude)
        sa = sin(attitude)
        cb = cos(bank)
        sb = sin(bank)

        self = cls()
        self.a = ch * ca
        self.b = sh * sb - ch * sa * cb
        self.c = ch * sa * sb + sh * cb
        self.e = sa
        self.f = ca * cb
        self.g = -ca * sb
        self.i = -sh * ca
        self.j = sh * sa * cb + ch * sb
        self.k = -sh * sa * sb + ch * cb
        return self
    new_rotate_euler = classmethod(new_rotate_euler)

    def new_rotate_triple_axis(cls, x, y, z):
        m = cls()
        
        m.a, m.b, m.c = x.x, y.x, z.x
        m.e, m.f, m.g = x.y, y.y, z.y
        m.i, m.j, m.k = x.z, y.z, z.z
        
        return m
    new_rotate_triple_axis = classmethod(new_rotate_triple_axis)

    def new_look_at(cls, eye, at, up):
        z = (eye - at).normalized()
        x = up.cross(z).normalized()
        y = z.cross(x)
        
        m = cls.new_rotate_triple_axis(x, y, z)
        m.d, m.h, m.l = eye.x, eye.y, eye.z
        return m
    
    new_look_at = classmethod(new_look_at)
    
    def new_perspective(cls, fov_y, aspect, near, far):
        # from the gluPerspective man page
        f = 1 / tan(fov_y / 2)
        self = cls()
        assert near != 0.0 and near != far
        self.a = f / aspect
        self.f = f
        self.k = (far + near) / (near - far)
        self.l = 2 * far * near / (near - far)
        self.o = -1
        self.p = 0
        return self
    new_perspective = classmethod(new_perspective)

    def determinant(self):
        return ((self.a * self.f - self.e * self.b)
              * (self.k * self.p - self.o * self.l)
              - (self.a * self.j - self.i * self.b)
              * (self.g * self.p - self.o * self.h)
              + (self.a * self.n - self.m * self.b)
              * (self.g * self.l - self.k * self.h)
              + (self.e * self.j - self.i * self.f)
              * (self.c * self.p - self.o * self.d)
              - (self.e * self.n - self.m * self.f)
              * (self.c * self.l - self.k * self.d)
              + (self.i * self.n - self.m * self.j)
              * (self.c * self.h - self.g * self.d))

    def inverse(self):
        tmp = Matrix4()
        d = self.determinant();

        if abs(d) < 0.001:
            # No inverse, return identity
            return tmp
        else:
            d = 1.0 / d;

            tmp.a = d * (self.f * (self.k * self.p - self.o * self.l) + self.j * (self.o * self.h - self.g * self.p) + self.n * (self.g * self.l - self.k * self.h));
            tmp.e = d * (self.g * (self.i * self.p - self.m * self.l) + self.k * (self.m * self.h - self.e * self.p) + self.o * (self.e * self.l - self.i * self.h));
            tmp.i = d * (self.h * (self.i * self.n - self.m * self.j) + self.l * (self.m * self.f - self.e * self.n) + self.p * (self.e * self.j - self.i * self.f));
            tmp.m = d * (self.e * (self.n * self.k - self.j * self.o) + self.i * (self.f * self.o - self.n * self.g) + self.m * (self.j * self.g - self.f * self.k));
            
            tmp.b = d * (self.j * (self.c * self.p - self.o * self.d) + self.n * (self.k * self.d - self.c * self.l) + self.b * (self.o * self.l - self.k * self.p));
            tmp.f = d * (self.k * (self.a * self.p - self.m * self.d) + self.o * (self.i * self.d - self.a * self.l) + self.c * (self.m * self.l - self.i * self.p));
            tmp.j = d * (self.l * (self.a * self.n - self.m * self.b) + self.p * (self.i * self.b - self.a * self.j) + self.d * (self.m * self.j - self.i * self.n));
            tmp.n = d * (self.i * (self.n * self.c - self.b * self.o) + self.m * (self.b * self.k - self.j * self.c) + self.a * (self.j * self.o - self.n * self.k));
            
            tmp.c = d * (self.n * (self.c * self.h - self.g * self.d) + self.b * (self.g * self.p - self.o * self.h) + self.f * (self.o * self.d - self.c * self.p));
            tmp.g = d * (self.o * (self.a * self.h - self.e * self.d) + self.c * (self.e * self.p - self.m * self.h) + self.g * (self.m * self.d - self.a * self.p));
            tmp.k = d * (self.p * (self.a * self.f - self.e * self.b) + self.d * (self.e * self.n - self.m * self.f) + self.h * (self.m * self.b - self.a * self.n));
            tmp.o = d * (self.m * (self.f * self.c - self.b * self.g) + self.a * (self.n * self.g - self.f * self.o) + self.e * (self.b * self.o - self.n * self.c));
            
            tmp.d = d * (self.b * (self.k * self.h - self.g * self.l) + self.f * (self.c * self.l - self.k * self.d) + self.j * (self.g * self.d - self.c * self.h));
            tmp.h = d * (self.c * (self.i * self.h - self.e * self.l) + self.g * (self.a * self.l - self.i * self.d) + self.k * (self.e * self.d - self.a * self.h));
            tmp.l = d * (self.d * (self.i * self.f - self.e * self.j) + self.h * (self.a * self.j - self.i * self.b) + self.l * (self.e * self.b - self.a * self.f));
            tmp.p = d * (self.a * (self.f * self.k - self.j * self.g) + self.e * (self.j * self.c - self.b * self.k) + self.i * (self.b * self.g - self.f * self.c));

        return tmp;
        
class Quaternion:
    """
    A quaternion represents a three-dimensional rotation or reflection
    transformation.  They are the preferred way to store and manipulate
    rotations in 3D applications, as they do not suffer the same numerical
    degredation that matrices do.
    
    The quaternion constructor initializes to the identity transform::
    
        >>> q = Quaternion()
        >>> q
        Quaternion(real=1.00, imag=<0.00, 0.00, 0.00>)
    
    **Element access**
    
    Internally, the quaternion is stored as four attributes: ``x``, ``y`` and
    ``z`` forming the imaginary vector, and ``w`` the real component.
    
    **Constructors**
    
    Rotations can be formed using the constructors:
    
    ``new_identity()``
        Equivalent to the default constructor.
    
    ``new_rotate_axis(angle, axis)``
        Equivalent to the Matrix4 constructor of the same name.  *angle* is
        specified in radians, *axis* is an instance of **Vector3**.  It is
        not necessary to normalize the axis.  Example::
    
            >>> q = Quaternion.new_rotate_axis(math.pi / 2, Vector3(1, 0, 0))
            >>> q
            Quaternion(real=0.71, imag=<0.71, 0.00, 0.00>)
    
    ``new_rotate_euler(heading, attitude, bank)``
        Equivalent to the Matrix4 constructor of the same name.  *heading*
        is a rotation around the Y axis, *attitude* around the X axis and
        *bank* around the Z axis.  All angles are given in radians.  Example::
    
            >>> q = Quaternion.new_rotate_euler(math.pi / 2, math.pi / 2, 0)
            >>> q
            Quaternion(real=0.50, imag=<0.50, 0.50, 0.50>)
    
    ``new_interpolate(q1, q2, t)``
        Create a quaternion which gives a (SLERP) interpolated rotation
        between *q1* and *q2*.  *q1* and *q2* are instances of **Quaternion**,
        and *t* is a value between 0.0 and 1.0.  For example::
    
            >>> q1 = Quaternion.new_rotate_axis(math.pi / 2, Vector3(1, 0, 0))
            >>> q2 = Quaternion.new_rotate_axis(math.pi / 2, Vector3(0, 1, 0))
            >>> for i in range(11):
            ...     print Quaternion.new_interpolate(q1, q2, i / 10.0)
            ...
            Quaternion(real=0.71, imag=<0.71, 0.00, 0.00>)
            Quaternion(real=0.75, imag=<0.66, 0.09, 0.00>)
            Quaternion(real=0.78, imag=<0.61, 0.17, 0.00>)
            Quaternion(real=0.80, imag=<0.55, 0.25, 0.00>)
            Quaternion(real=0.81, imag=<0.48, 0.33, 0.00>)
            Quaternion(real=0.82, imag=<0.41, 0.41, 0.00>)
            Quaternion(real=0.81, imag=<0.33, 0.48, 0.00>)
            Quaternion(real=0.80, imag=<0.25, 0.55, 0.00>)
            Quaternion(real=0.78, imag=<0.17, 0.61, 0.00>)
            Quaternion(real=0.75, imag=<0.09, 0.66, 0.00>)
            Quaternion(real=0.71, imag=<0.00, 0.71, 0.00>)
    
    
    **Operators**
    
    Quaternions may be multiplied to compound rotations.  For example, to
    rotate 90 degrees around the X axis and then 90 degrees around the Y axis::
    
        >>> q1 = Quaternion.new_rotate_axis(math.pi / 2, Vector3(1, 0, 0))
        >>> q2 = Quaternion.new_rotate_axis(math.pi / 2, Vector3(0, 1, 0))
        >>> q1 * q2
        Quaternion(real=0.50, imag=<0.50, 0.50, 0.50>)
    
    Multiplying a quaternion by a vector gives a vector, transformed
    appropriately::
    
        >>> q = Quaternion.new_rotate_axis(math.pi / 2, Vector3(0, 1, 0))
        >>> q * Vector3(1.0, 0, 0)
        Vector3(0.00, 0.00, -1.00)
    
    Similarly, any 3D object can be multiplied (e.g., **Point3**, **Line3**,
    **Sphere**, etc)::
    
        >>> q * Ray3(Point3(1., 1., 1.), Vector3(1., 1., 1.))
        Ray3(<1.00, 1.00, -1.00> + u<1.00, 1.00, -1.00>)
    
    As with the matrix classes, the constructors are also available as in-place
    operators.  These are named ``identity``, ``rotate_euler`` and
    ``rotate_axis``.  For example::
    
        >>> q1 = Quaternion()
        >>> q1.rotate_euler(math.pi / 2, math.pi / 2, 0)
        Quaternion(real=0.50, imag=<0.50, 0.50, 0.50>)
        >>> q1
        Quaternion(real=0.50, imag=<0.50, 0.50, 0.50>)
    
    Quaternions are usually unit length, but you may wish to use sized
    quaternions.  In this case, you can find the magnitude using ``abs``,
    ``magnitude`` and ``magnitude_squared``, as with the vector classes.
    Example::
    
        >>> q1 = Quaternion()
        >>> abs(q1)
        1.0
        >>> q1.magnitude()
        1.0
    
    Similarly, the class implements ``normalize`` and ``normalized`` in the
    same way as the vectors.
    
    The following methods do not alter the quaternion:
    
    ``conjugated()``
        Returns a quaternion that is the conjugate of the instance.  For
        example::
            
            >>> q1 = Quaternion.new_rotate_axis(math.pi / 2, Vector3(1, 0, 0))
            >>> q1.conjugated()
            Quaternion(real=0.71, imag=<-0.71, -0.00, -0.00>)
            >>> q1
            Quaternion(real=0.71, imag=<0.71, 0.00, 0.00>)
    
    ``get_angle_axis()``
        Returns a tuple (angle, axis), giving the angle to rotate around an
        axis equivalent to the quaternion.  For example::
    
            >>> q1 = Quaternion.new_rotate_axis(math.pi / 2, Vector3(1, 0, 0))
            >>> q1.get_angle_axis()
            (1.5707963267948966, Vector3(1.00, 0.00, 0.00))
    
    ``get_matrix()``
        Returns a **Matrix4** implementing the transformation of the quaternion.
        For example::
            
            >>> q1 = Quaternion.new_rotate_axis(math.pi / 2, Vector3(1, 0, 0))
            >>> q1.get_matrix()
            Matrix4([    1.00     0.00     0.00     0.00
                         0.00     0.00    -1.00     0.00
                         0.00     1.00     0.00     0.00
                         0.00     0.00     0.00     1.00])
    """
    # All methods and naming conventions based off 
    # http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions

    # w is the real part, (x, y, z) are the imaginary parts

    def __init__(self, w=1, x=0, y=0, z=0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __copy__(self):
        Q = Quaternion()
        Q.w = self.w
        Q.x = self.x
        Q.y = self.y
        Q.z = self.z
        return Q

    copy = __copy__

    def __repr__(self):
        return '%s(%g,%g,%g,%g)' % (self.__class__.__name__,self.w, self.x, self.y, self.z)

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            Ax = self.x
            Ay = self.y
            Az = self.z
            Aw = self.w
            Bx = other.x
            By = other.y
            Bz = other.z
            Bw = other.w
            Q = Quaternion()
            Q.x =  Ax * Bw + Ay * Bz - Az * By + Aw * Bx    
            Q.y = -Ax * Bz + Ay * Bw + Az * Bx + Aw * By
            Q.z =  Ax * By - Ay * Bx + Az * Bw + Aw * Bz
            Q.w = -Ax * Bx - Ay * By - Az * Bz + Aw * Bw
            return Q
        elif isinstance(other, Vector3):
            w = self.w
            x = self.x
            y = self.y
            z = self.z
            Vx = other.x
            Vy = other.y
            Vz = other.z
            ww = w * w
            w2 = w * 2
            wx2 = w2 * x
            wy2 = w2 * y
            wz2 = w2 * z
            xx = x * x
            x2 = x * 2
            xy2 = x2 * y
            xz2 = x2 * z
            yy = y * y
            yz2 = 2 * y * z
            zz = z * z
            return other.__class__(\
               ww * Vx + wy2 * Vz - wz2 * Vy + \
               xx * Vx + xy2 * Vy + xz2 * Vz - \
               zz * Vx - yy * Vx,
               xy2 * Vx + yy * Vy + yz2 * Vz + \
               wz2 * Vx - zz * Vy + ww * Vy - \
               wx2 * Vz - xx * Vy,
               xz2 * Vx + yz2 * Vy + \
               zz * Vz - wy2 * Vx - yy * Vz + \
               wx2 * Vy - xx * Vz + ww * Vz)
        else:
            other = other.copy()
            other._apply_transform(self)
            return other

    def __imul__(self, other):
        assert isinstance(other, Quaternion)
        Ax = self.x
        Ay = self.y
        Az = self.z
        Aw = self.w
        Bx = other.x
        By = other.y
        Bz = other.z
        Bw = other.w
        self.x =  Ax * Bw + Ay * Bz - Az * By + Aw * Bx    
        self.y = -Ax * Bz + Ay * Bw + Az * Bx + Aw * By
        self.z =  Ax * By - Ay * Bx + Az * Bw + Aw * Bz
        self.w = -Ax * Bx - Ay * By - Az * Bz + Aw * Bw
        return self

    def __abs__(self):
        return sqrt(self.w ** 2 + \
                         self.x ** 2 + \
                         self.y ** 2 + \
                         self.z ** 2)

    mag = __abs__

    def mag2(self):
        return self.w ** 2 + \
               self.x ** 2 + \
               self.y ** 2 + \
               self.z ** 2 

    def identity(self):
        self.w = 1
        self.x = 0
        self.y = 0
        self.z = 0
        return self

    def rotate_axis(self, angle, axis):
        self *= Quaternion.new_rotate_axis(angle, axis)
        return self

    def rotate_euler(self, heading, attitude, bank):
        self *= Quaternion.new_rotate_euler(heading, attitude, bank)
        return self

    def rotate_matrix(self, m):
        self *= Quaternion.new_rotate_matrix(m)
        return self

    def conjugated(self):
        Q = Quaternion()
        Q.w = self.w
        Q.x = -self.x
        Q.y = -self.y
        Q.z = -self.z
        return Q

    def normalize(self):
        d = self.mag()
        if d != 0:
            self.w /= d
            self.x /= d
            self.y /= d
            self.z /= d
        return self

    def normalized(self):
        d = self.mag()
        if d != 0:
            Q = Quaternion()
            Q.w = self.w / d
            Q.x = self.x / d
            Q.y = self.y / d
            Q.z = self.z / d
            return Q
        else:
            return self.copy()

    def get_angle_axis(self):
        if self.w > 1:
            self = self.normalized()
        angle = 2 * acos(self.w)
        s = sqrt(1 - self.w ** 2)
        if s < 0.001:
            return angle, Vector3(1, 0, 0)
        else:
            return angle, Vector3(self.x / s, self.y / s, self.z / s)

    def get_euler(self):
        t = self.x * self.y + self.z * self.w
        if t > 0.4999:
            heading = 2 * atan2(self.x, self.w)
            attitude = pi / 2
            bank = 0
        elif t < -0.4999:
            heading = -2 * atan2(self.x, self.w)
            attitude = -pi / 2
            bank = 0
        else:
            sqx = self.x ** 2
            sqy = self.y ** 2
            sqz = self.z ** 2
            heading = atan2(2 * self.y * self.w - 2 * self.x * self.z,
                                 1 - 2 * sqy - 2 * sqz)
            attitude = asin(2 * t)
            bank = atan2(2 * self.x * self.w - 2 * self.y * self.z,
                              1 - 2 * sqx - 2 * sqz)
        return heading, attitude, bank

    def get_matrix(self):
        xx = self.x ** 2
        xy = self.x * self.y
        xz = self.x * self.z
        xw = self.x * self.w
        yy = self.y ** 2
        yz = self.y * self.z
        yw = self.y * self.w
        zz = self.z ** 2
        zw = self.z * self.w
        M = Matrix4()
        M.a = 1 - 2 * (yy + zz)
        M.b = 2 * (xy - zw)
        M.c = 2 * (xz + yw)
        M.e = 2 * (xy + zw)
        M.f = 1 - 2 * (xx + zz)
        M.g = 2 * (yz - xw)
        M.i = 2 * (xz - yw)
        M.j = 2 * (yz + xw)
        M.k = 1 - 2 * (xx + yy)
        return M

    # Static constructors
    def new_identity(cls):
        return cls()
    new_identity = classmethod(new_identity)

    def new_rotate_axis(cls, angle, axis):
        assert(isinstance(axis, Vector3))
        axis = axis.normalized()
        s = sin(angle / 2)
        Q = cls()
        Q.w = cos(angle / 2)
        Q.x = axis.x * s
        Q.y = axis.y * s
        Q.z = axis.z * s
        return Q
    new_rotate_axis = classmethod(new_rotate_axis)

    def new_rotate_euler(cls, heading, attitude, bank):
        Q = cls()
        c1 = cos(heading / 2)
        s1 = sin(heading / 2)
        c2 = cos(attitude / 2)
        s2 = sin(attitude / 2)
        c3 = cos(bank / 2)
        s3 = sin(bank / 2)

        Q.w = c1 * c2 * c3 - s1 * s2 * s3
        Q.x = s1 * s2 * c3 + c1 * c2 * s3
        Q.y = s1 * c2 * c3 + c1 * s2 * s3
        Q.z = c1 * s2 * c3 - s1 * c2 * s3
        return Q
    new_rotate_euler = classmethod(new_rotate_euler)
    
    def new_rotate_matrix(cls, m):
        if m[0*4 + 0] + m[1*4 + 1] + m[2*4 + 2] > 0.00000001:
            t = m[0*4 + 0] + m[1*4 + 1] + m[2*4 + 2] + 1.0
            s = 0.5/sqrt(t)
        
            return cls(
              s*t,
              (m[1*4 + 2] - m[2*4 + 1])*s,
              (m[2*4 + 0] - m[0*4 + 2])*s,
              (m[0*4 + 1] - m[1*4 + 0])*s
              )
        
        elif m[0*4 + 0] > m[1*4 + 1] and m[0*4 + 0] > m[2*4 + 2]:
            t = m[0*4 + 0] - m[1*4 + 1] - m[2*4 + 2] + 1.0
            s = 0.5/sqrt(t)
            
            return cls(
              (m[1*4 + 2] - m[2*4 + 1])*s,
              s*t,
              (m[0*4 + 1] + m[1*4 + 0])*s,
              (m[2*4 + 0] + m[0*4 + 2])*s
              )
        
        elif m[1*4 + 1] > m[2*4 + 2]:
            t = -m[0*4 + 0] + m[1*4 + 1] - m[2*4 + 2] + 1.0
            s = 0.5/sqrt(t)
            
            return cls(
              (m[2*4 + 0] - m[0*4 + 2])*s,
              (m[0*4 + 1] + m[1*4 + 0])*s,
              s*t,
              (m[1*4 + 2] + m[2*4 + 1])*s
              )
        
        else:
            t = -m[0*4 + 0] - m[1*4 + 1] + m[2*4 + 2] + 1.0
            s = 0.5/sqrt(t)
        
            return cls(
              (m[0*4 + 1] - m[1*4 + 0])*s,
              (m[2*4 + 0] + m[0*4 + 2])*s,
              (m[1*4 + 2] + m[2*4 + 1])*s,
              s*t
              )
            
    new_rotate_matrix = classmethod(new_rotate_matrix)
    
    def new_interpolate(cls, q1, q2, t):
        assert isinstance(q1, Quaternion) and isinstance(q2, Quaternion)
        Q = cls()

        costheta = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z
        if costheta < 0.:
            costheta = -costheta
            q1 = q1.conjugated()
        elif costheta > 1:
            costheta = 1

        theta = acos(costheta)
        if abs(theta) < 0.01:
            Q.w = q2.w
            Q.x = q2.x
            Q.y = q2.y
            Q.z = q2.z
            return Q

        sintheta = sqrt(1.0 - costheta * costheta)
        if abs(sintheta) < 0.01:
            Q.w = (q1.w + q2.w) * 0.5
            Q.x = (q1.x + q2.x) * 0.5
            Q.y = (q1.y + q2.y) * 0.5
            Q.z = (q1.z + q2.z) * 0.5
            return Q

        ratio1 = sin((1 - t) * theta) / sintheta
        ratio2 = sin(t * theta) / sintheta

        Q.w = q1.w * ratio1 + q2.w * ratio2
        Q.x = q1.x * ratio1 + q2.x * ratio2
        Q.y = q1.y * ratio1 + q2.y * ratio2
        Q.z = q1.z * ratio1 + q2.z * ratio2
        return Q
    new_interpolate = classmethod(new_interpolate)

# Geometry
# Much maths thanks to Paul Bourke, http://astronomy.swin.edu.au/~pbourke
# ---------------------------------------------------------------------------

from abc import ABCMeta #Abstract Base Class
    
class Geometry(object):
    """
    The following classes are available for dealing with simple 2D geometry.
    The interface to each shape is similar; in particular, the ``connect``
    and ``distance`` methods are defined identically for each.
    
    For example, to find the closest point on a line to a circle::
    
        >>> circ = Circle(Point2(3., 2.), 2.)
        >>> line = Line2(Point2(0., 0.), Point2(-1., 1.))
        >>> line.connect(circ).p1
        Point2(0.50, -0.50)
    
    To find the corresponding closest point on the circle to the line::
    
        >>> line.connect(circ).p2
        Point2(1.59, 0.59)
    """
    __metaclass__ = ABCMeta
    def _connect_unimplemented(self, other):
        raise AttributeError, 'Cannot connect %s to %s' % \
            (self.__class__, other.__class__)

    def _intersect_unimplemented(self, other):
        raise AttributeError, 'Cannot intersect %s and %s' % \
            (self.__class__, other.__class__)

    _intersect_point2 = _intersect_unimplemented
    _intersect_line2 = _intersect_unimplemented
    _intersect_circle = _intersect_unimplemented
    _connect_point2 = _connect_unimplemented
    _connect_line2 = _connect_unimplemented
    _connect_circle = _connect_unimplemented

    _intersect_point3 = _intersect_unimplemented
    _intersect_line3 = _intersect_unimplemented
    _intersect_sphere = _intersect_unimplemented
    _intersect_plane = _intersect_unimplemented
    _connect_point3 = _connect_unimplemented
    _connect_line3 = _connect_unimplemented
    _connect_sphere = _connect_unimplemented
    _connect_plane = _connect_unimplemented
    
    def point(self, u):
        ":return: Point2 or Point3 at parameter u"
        raise NotImplementedError
    
    def tangent(self, u):
        ":return: Vector2 or Vector3 tangent at parameter u"
        raise NotImplementedError
    
    def intersect(self, other):
        raise NotImplementedError

    def connect(self, other):
        ":return: Geometry shortest (Segment2 or Segment3) that connects self to other"
        raise NotImplementedError

    def distance(self, other):
        c = self.connect(other)
        if c:
            return c.length
        else:
            return None

def _intersect_point2_circle(P, C):
    return abs(P - C.c) <= C.r
    
def _intersect_line2_line2(A, B):
    d = B.v.y * A.v.x - B.v.x * A.v.y
    if d == 0:
        return None

    dy = A.p.y - B.p.y
    dx = A.p.x - B.p.x
    ua = (B.v.x * dy - B.v.y * dx) / d
    if not A._u_in(ua):
        return None
    ub = (A.v.x * dy - A.v.y * dx) / d
    if not B._u_in(ub):
        return None

    return Point2(A.p.x + ua * A.v.x,
                  A.p.y + ua * A.v.y)

def _intersect_line2_circle(L, C):
    a = L.v.mag2()
    b = 2 * (L.v.x * (L.p.x - C.c.x) + \
             L.v.y * (L.p.y - C.c.y))
    c = C.c.mag2() + \
        L.p.mag2() - \
        2 * C.c.dot(L.p) - \
        C.r ** 2
    det = b ** 2 - 4 * a * c
    if det < 0:
        return None
    sq = sqrt(det)
    u1 = (-b + sq) / (2 * a)
    u2 = (-b - sq) / (2 * a)
    if not L._u_in(u1):
        u1 = max(min(u1, 1.0), 0.0)
    if not L._u_in(u2):
        u2 = max(min(u2, 1.0), 0.0)

    # Tangent
    if u1 == u2:
        return Point2(L.p.x + u1 * L.v.x,
                      L.p.y + u1 * L.v.y)

    return Segment2(Point2(L.p.x + u1 * L.v.x,
                               L.p.y + u1 * L.v.y),
                        Point2(L.p.x + u2 * L.v.x,
                               L.p.y + u2 * L.v.y))

def _connect_point2_line2(P, L):
    d = L.v.mag2()
    assert d != 0
    u = ((P.x - L.p.x) * L.v.x + \
         (P.y - L.p.y) * L.v.y) / d
    if not L._u_in(u):
        u = max(min(u, 1.0), 0.0)
    return Segment2(P, 
                        Point2(L.p.x + u * L.v.x,
                               L.p.y + u * L.v.y))

def _connect_point2_circle(P, C):
    v = P - C.c
    v.normalize()
    v *= C.r
    return Segment2(P, Point2(C.c.x + v.x, C.c.y + v.y))

def _connect_line2_line2(A, B):
    d = B.v.y * A.v.x - B.v.x * A.v.y
    if d == 0:
        # Parallel, connect an endpoint with a line
        if isinstance(B, (Ray2,Segment2)):
            p1, p2 = _connect_point2_line2(B.p, A)
            return p2, p1
        # No endpoint (or endpoint is on A), possibly choose arbitrary point
        # on line.
        return _connect_point2_line2(A.p, B)

    dy = A.p.y - B.p.y
    dx = A.p.x - B.p.x
    ua = (B.v.x * dy - B.v.y * dx) / d
    if not A._u_in(ua):
        ua = max(min(ua, 1.0), 0.0)
    ub = (A.v.x * dy - A.v.y * dx) / d
    if not B._u_in(ub):
        ub = max(min(ub, 1.0), 0.0)

    return Segment2(Point2(A.p.x + ua * A.v.x, A.p.y + ua * A.v.y),
                        Point2(B.p.x + ub * B.v.x, B.p.y + ub * B.v.y))

def _connect_circle_line2(C, L):
    d = L.v.mag2()
    assert d != 0
    u = ((C.c.x - L.p.x) * L.v.x + (C.c.y - L.p.y) * L.v.y) / d
    if not L._u_in(u):
        u = max(min(u, 1.0), 0.0)
    point = Point2(L.p.x + u * L.v.x, L.p.y + u * L.v.y)
    v = (point - C.c)
    v.normalize()
    v *= C.r
    return Segment2(Point2(C.c.x + v.x, C.c.y + v.y), point)

def _connect_circle_circle(A, B):
    v = B.c - A.c
    d = v.mag()
    if A.r >= B.r and d < A.r:
        #centre B inside A
        s1,s2 = +1, +1
    elif B.r > A.r and d < B.r:
        #centre A inside B
        s1,s2 = -1, -1
    elif d >= A.r and d >= B.r:
        s1,s2 = +1, -1
    v.normalize()
    return Segment2(Point2(A.c.x + s1 * v.x * A.r, A.c.y + s1 * v.y * A.r),
                        Point2(B.c.x + s2 * v.x * B.r, B.c.y + s2 * v.y * B.r))

class Point2(Vector2, Geometry):
    """
    A point on a 2D plane.  Construct in the obvious way::

    >>> p = Point2(1.0, 2.0)
    >>> p
    Point2(1.00, 2.00)

    **Point2** subclasses **Vector2**, so all of **Vector2** operators and
    methods apply.  In particular, subtracting two points gives a vector::

    >>> Point2(2.0, 3.0) - Point2(1.0, 0.0)
    Vector2(1.00, 3.00)

    The following methods are also defined:

    ``connect(other)``
        Returns a **LineSegment2** which is the minimum length line segment
        that can connect the two shapes.  *other* may be a **Point2**, **Line2**,
        **Ray2**, **LineSegment2** or **Circle**.
    
    ``distance(other)``
        Returns the absolute minimum distance to *other*.  Internally this
        simply returns the length of the result of ``connect``. 
    """
    
    def dist(self,other):
        return (self-other).mag()

    def intersect(self, other):
        return other._intersect_point2(self)

    def _intersect_circle(self, other):
        return _intersect_point2_circle(self, other)

    def connect(self, other):
        return other._connect_point2(self)

    def _connect_point2(self, other):
        return Segment2(other, self)
    
    def _connect_line2(self, other):
        c = _connect_point2_line2(self, other)
        if c:
            return c._swap()

    def _connect_circle(self, other):
        c = _connect_point2_circle(self, other)
        if c:
            return c._swap()
        
def Polar(mag,angle):
    return Vector2(mag*cos(angle),mag*sin(angle))


class Line2(Geometry):
    """
    A **Line2** is a line on a 2D plane extending to infinity in both directions;
    a **Ray2** has a finite end-point and extends to infinity in a single
    direction; a **LineSegment2** joins two points.  
    
    All three classes support the same constructors, operators and methods,
    but may behave differently when calculating intersections etc.
    
    You may construct a line, ray or line segment using any of:
    
    * another line, ray or line segment
    * two points
    * a point and a vector
    * a point, a vector and a length
    
    For example::
    
        >>> Line2(Point2(1.0, 1.0), Point2(2.0, 3.0))
        Line2(<1.00, 1.00> + u<1.00, 2.00>)
        >>> Line2(Point2(1.0, 1.0), Vector2(1.0, 2.0))
        Line2(<1.00, 1.00> + u<1.00, 2.00>)
        >>> Ray2(Point2(1.0, 1.0), Vector2(1.0, 2.0), 1.0)
        Ray2(<1.00, 1.00> + u<0.45, 0.89>)
    
    Internally, lines, rays and line segments store a Point2 *p* and a
    Vector2 *v*.  You can also access (but not set) the two endpoints
    *p1* and *p2*.  These may or may not be meaningful for all types of lines.
    
    The following methods are supported by all three classes:
    
    ``intersect(other)``
        If *other* is a **Line2**, **Ray2** or **LineSegment2**, returns
        a **Point2** of intersection, or None if the lines are parallel.
    
        If *other* is a **Circle**, returns a **LineSegment2** or **Point2** giving
        the part of the line that intersects the circle, or None if there
        is no intersection.
    
    ``connect(other)``
        Returns a **LineSegment2** which is the minimum length line segment
        that can connect the two shapes.  For two parallel lines, this
        line segment may be in an arbitrary position.  *other* may be
        a **Point2**, **Line2**, **Ray2**, **LineSegment2** or **Circle**.
    
    ``distance(other)``
        Returns the absolute minimum distance to *other*.  Internally this
        simply returns the length of the result of ``connect``.
    
    **LineSegment2** also has a *length* property which is read-only.
    """

    def __init__(self, *args):
        if len(args) == 1: # Line2 or derived class
            self.p = args[0].p.copy()
            self.v = args[0].v.copy()
        else:
            self.p = Point2(args[0])
            if isinstance(args[1], Point2):
                self.v = args[1] - args[0]
            else:
                self.v = Vector2(args[1])
            
            if len(args) == 3:
                self.v=self.v*args[2]/abs(self.v)
        
    def __eq__(self, other):
        """lines are "equal" only if base points and vector are strictly equal.
        to compare if lines are "same", use line1.distance(line2)==0
        """
        try:
            return self.p==other.p and self.v==other.v
        except:
            return False

    def __copy__(self):
        return self.__class__(self.p, self.v)

    copy = __copy__

    def __repr__(self):
        return '%s(%s,%s)' % (self.__class__.__name__,self.p,self.v)
            
    def _u_in(self, u):
        return True
    
    def point(self, u):
        ":return: Point2 at parameter u"
        if self._u_in(u):
            return self.p+u*self.v
        else:
            return None
    
    def tangent(self, u):
        ":return: Vector2 tangent at parameter u. Warning : tangent is generally not a unit vector"
        if self._u_in(u):
            return self.v
        else:
            return None

    def _apply_transform(self, t):
        self.p = t * self.p
        self.v = t * self.v

    def intersect(self, other):
        return other._intersect_line2(self)

    def _intersect_line2(self, other):
        return _intersect_line2_line2(self, other)

    def _intersect_circle(self, other):
        return _intersect_line2_circle(self, other)

    def connect(self, other):
        return other._connect_line2(self)

    def _connect_point2(self, other):
        return _connect_point2_line2(other, self)

    def _connect_line2(self, other):
        return _connect_line2_line2(other, self)

    def _connect_circle(self, other):
        return _connect_circle_line2(other, self)
    


class Ray2(Line2):

    def _u_in(self, u):
        return u >= 0.0

class Segment2(Line2):
    p1 = property(lambda self: self.p)
    p2 = property(lambda self: Point2(self.p.x + self.v.x, 
                                      self.p.y + self.v.y))
    
    def __repr__(self):
        return '%s(%s,%s)' % (self.__class__.__name__,self.p,self.p2)

    def _u_in(self, u):
        return u >= 0.0 and u <= 1.0

    def __abs__(self):
        return abs(self.v)

    def mag2(self):
        return self.v.mag2()

    length = property(lambda self: abs(self.v))
    
    def _swap(self):
        # used by connect methods to switch order of points
        self.p = self.p2
        self.v *= -1
        return self

class Circle(Geometry):
    """
    Circles are constructed with a center **Point2** and a radius::

    >>> c = Circle(Point2(1.0, 1.0), 0.5)
    >>> c
    Circle(<1.00, 1.00>, radius=0.50)

    Internally there are two attributes: *c*, giving the center point and
    *r*, giving the radius.
    
    The following methods are supported:
    
    ``intersect(other)``
        If *other* is a **Line2**, **Ray2** or **LineSegment2**, returns
        a **LineSegment2** giving the part of the line that intersects the
        circle, or None if there is no intersection.
    
    ``connect(other)``
        Returns a **LineSegment2** which is the minimum length line segment
        that can connect the two shapes. *other* may be a **Point2**, **Line2**,
        **Ray2**, **LineSegment2** or **Circle**.
    
    ``distance(other)``
        Returns the absolute minimum distance to *other*.  Internally this
        simply returns the length of the result of ``connect``. 
    """

    def __init__(self, center, radius):
        self.c = Point2(center) if not isinstance(center,Point2) else center
        if type(radius) == float:
            self.r = radius
            self.p = center+Vector2(radius,0) #for coherency + transform
        else:
            self.p=radius #one point on circle
            self.r=abs(self.p-self.c)

    def point(self, u):
        ":return: Point2 at angle u radians"
        return self.c+Polar(self.r,u)
    
    def tangent(self, u):
        ":return: Vector2 tangent at angle u. Warning : tangent has magnitude r != 1"
        return Polar(self.r,u+pi/2.)
        
    def __copy__(self):
        return self.__class__(self.c, self.r)
    
    def __eq__(self, other):
        if not isinstance(other,Circle):
            return False
        return self.c==other.c and self.r==other.r

    copy = __copy__

    def __repr__(self):
        return '%s(%s,%g)' % (self.__class__.__name__,self.c,self.r)

    def _apply_transform(self, t):
        self.c = t * self.c
        self.p = t * self.p
        self.r=abs(self.p-self.c)
        
    def __abs__(self):
        """:return: float perimeter"""
        return 2.0*pi*self.r
    
    length = property(lambda self: abs(self))

    def intersect(self, other):
        return other._intersect_circle(self)

    def _intersect_point2(self, other):
        return _intersect_point2_circle(other, self)

    def _intersect_line2(self, other):
        return _intersect_line2_circle(other, self)

    def connect(self, other):
        return other._connect_circle(self)

    def _connect_point2(self, other):
        return _connect_point2_circle(other, self)

    def _connect_line2(self, other):
        c = _connect_circle_line2(self, other)
        if c:
            return c._swap()

    def _connect_circle(self, other):
        return _connect_circle_circle(other, self)
    
class Arc2(Circle):
    def __init__(self, center, p1,p2,r=0):
        c=Point2(center) if not isinstance(center,Point2) else center
        if isinstance(p1,(int,float)):
            p=c+Polar(r,p1)
        else:
            p=Point2(p1)
        super(Arc2,self).__init__(c,p)
        if isinstance(p2,(int,float)):
            self.p2=c+Polar(r,p2)
        else:
            self.p2=Point2(p2)
            
        self._apply_transform(None) #to set start/end angles
        
    def angle(self):
        """:return: float arc angle"""
        l=self.b-self.a
        if l<0 :
            l=2.*pi+l
        return l
        
    def __abs__(self):
        """:return: float arc length"""
        return self.r*self.angle()
        
    def _u_in(self, u): #unlike Circle, Arc2 is parametrized on [0,1] for coherency with Segment2
        return u >= 0.0 and u <= 1.0
    
    def point(self, u):
        ":return: Point2 at parameter u"
        if self._u_in(u):
            return self.p+u*self.v
        else:
            return None
    
    def tangent(self, u):
        ":return: Vector2 tangent at parameter u. Warning : tangent not a unit vector"
        if self._u_in(u):
            return self.v
        else:
            return None
            
    def _apply_transform(self, t):
        if t:
            super(Arc2,self)._apply_transform(t)
            self.p2 = t * self.p2
        self.a=(self.p-self.c).angle(None) #start angle
        self.b=(self.p2-self.c).angle(None) #end angle
        
    def __copy__(self):
        return self.__class__(self.c, self.p, self.p2)

    copy = __copy__
    
    def __eq__(self, other):
        if not super(Arc2,self).__eq__(other): #support Circles must be the same
            return False
        return self.p==other.p and self.p2==other.p2
        
    def __repr__(self):
        return '%s(%s,%s)' % (self.__class__.__name__,self.c,self.r,self.p,self.p2)

    def _swap(self):
        # used by connect methods to switch order of points
        self.p,self.p2 = self.p2,self.p
        self.a,self.b = self.b,self.a
        return self


# 3D Geometry
# -------------------------------------------------------------------------

"""
**3D Geometry**

The following classes are available for dealing with simple 3D geometry.
The interfaces are very similar to the 2D classes (but note that you
cannot mix and match 2D and 3D operations).

For example, to find the closest point on a line to a sphere::

    >>> sphere = Sphere(Point3(1., 2., 3.,), 2.)
    >>> line = Line3(Point3(0., 0., 0.), Point3(-1., -1., 0.))
    >>> line.connect(sphere).p1
    Point3(1.50, 1.50, 0.00)

To find the corresponding closest point on the sphere to the line::

    >>> line.connect(sphere).p2
    Point3(1.32, 1.68, 1.05)

XXX I have not checked if these are correct.
"""

def _connect_point3_line3(P, L):
    d = L.v.mag2()
    assert d != 0
    u = ((P.x - L.p.x) * L.v.x + \
         (P.y - L.p.y) * L.v.y + \
         (P.z - L.p.z) * L.v.z) / d
    if not L._u_in(u):
        u = max(min(u, 1.0), 0.0)
    return Segment3(P, Point3(L.p.x + u * L.v.x,
                                  L.p.y + u * L.v.y,
                                  L.p.z + u * L.v.z))

def _connect_point3_sphere(P, S):
    v = P - S.c
    v.normalize()
    v *= S.r
    return Segment3(P, Point3(S.c.x + v.x, S.c.y + v.y, S.c.z + v.z))

def _connect_point3_plane(p, plane):
    n = plane.n.normalized()
    d = p.dot(plane.n) - plane.k
    return Segment3(p, Point3(p.x - n.x * d, p.y - n.y * d, p.z - n.z * d))

def _connect_line3_line3(A, B):
    assert A.v and B.v
    p13 = A.p - B.p
    d1343 = p13.dot(B.v)
    d4321 = B.v.dot(A.v)
    d1321 = p13.dot(A.v)
    d4343 = B.v.mag2()
    denom = A.v.mag2() * d4343 - d4321 ** 2
    if denom == 0:
        # Parallel, connect an endpoint with a line
        if isinstance(B, Ray3) or isinstance(B, Segment3):
            return _connect_point3_line3(B.p, A)._swap()
        # No endpoint (or endpoint is on A), possibly choose arbitrary
        # point on line.
        return _connect_point3_line3(A.p, B)

    ua = (d1343 * d4321 - d1321 * d4343) / denom
    if not A._u_in(ua):
        ua = max(min(ua, 1.0), 0.0)
    ub = (d1343 + d4321 * ua) / d4343
    if not B._u_in(ub):
        ub = max(min(ub, 1.0), 0.0)
    return Segment3(Point3(A.p.x + ua * A.v.x,
                               A.p.y + ua * A.v.y,
                               A.p.z + ua * A.v.z),
                        Point3(B.p.x + ub * B.v.x,
                               B.p.y + ub * B.v.y,
                               B.p.z + ub * B.v.z))

def _connect_line3_plane(L, P):
    d = P.n.dot(L.v)
    if not d:
        # Parallel, choose an endpoint
        return _connect_point3_plane(L.p, P)
    u = (P.k - P.n.dot(L.p)) / d
    if not L._u_in(u):
        # intersects out of range, choose nearest endpoint
        u = max(min(u, 1.0), 0.0)
        return _connect_point3_plane(Point3(L.p.x + u * L.v.x,
                                            L.p.y + u * L.v.y,
                                            L.p.z + u * L.v.z), P)
    # Intersection
    return None

def _connect_sphere_line3(S, L):
    d = L.v.mag2()
    assert d != 0
    u = ((S.c.x - L.p.x) * L.v.x + \
         (S.c.y - L.p.y) * L.v.y + \
         (S.c.z - L.p.z) * L.v.z) / d
    if not L._u_in(u):
        u = max(min(u, 1.0), 0.0)
    point = Point3(L.p.x + u * L.v.x, L.p.y + u * L.v.y, L.p.z + u * L.v.z)
    v = (point - S.c)
    v.normalize()
    v *= S.r
    return Segment3(Point3(S.c.x + v.x, S.c.y + v.y, S.c.z + v.z), 
                        point)

def _connect_sphere_sphere(A, B):
    v = B.c - A.c
    d = v.mag()
    if A.r >= B.r and d < A.r:
        #centre B inside A
        s1,s2 = +1, +1
    elif B.r > A.r and d < B.r:
        #centre A inside B
        s1,s2 = -1, -1
    elif d >= A.r and d >= B.r:
        s1,s2 = +1, -1

    v.normalize()
    return Segment3(Point3(A.c.x + s1* v.x * A.r,
                               A.c.y + s1* v.y * A.r,
                               A.c.z + s1* v.z * A.r),
                        Point3(B.c.x + s2* v.x * B.r,
                               B.c.y + s2* v.y * B.r,
                               B.c.z + s2* v.z * B.r))

def _connect_sphere_plane(S, P):
    c = _connect_point3_plane(S.c, P)
    if not c:
        return None
    p2 = c.p2
    v = p2 - S.c
    v.normalize()
    v *= S.r
    return Segment3(Point3(S.c.x + v.x, S.c.y + v.y, S.c.z + v.z), 
                        p2)

def _connect_plane_plane(A, B):
    if A.n.cross(B.n):
        # Planes intersect
        return None
    else:
        # Planes are parallel, connect to arbitrary point
        return _connect_point3_plane(A._get_point(), B)

def _intersect_point3_sphere(P, S):
    return abs(P - S.c) <= S.r
    
def _intersect_line3_sphere(L, S):
    a = L.v.mag2()
    b = 2 * (L.v.x * (L.p.x - S.c.x) + \
             L.v.y * (L.p.y - S.c.y) + \
             L.v.z * (L.p.z - S.c.z))
    c = S.c.mag2() + \
        L.p.mag2() - \
        2 * S.c.dot(L.p) - \
        S.r ** 2
    det = b ** 2 - 4 * a * c
    if det < 0:
        return None
    sq = sqrt(det)
    u1 = (-b + sq) / (2 * a)
    u2 = (-b - sq) / (2 * a)
    if not L._u_in(u1):
        u1 = max(min(u1, 1.0), 0.0)
    if not L._u_in(u2):
        u2 = max(min(u2, 1.0), 0.0)
    return Segment3(Point3(L.p.x + u1 * L.v.x,
                               L.p.y + u1 * L.v.y,
                               L.p.z + u1 * L.v.z),
                        Point3(L.p.x + u2 * L.v.x,
                               L.p.y + u2 * L.v.y,
                               L.p.z + u2 * L.v.z))

def _intersect_line3_plane(L, P):
    d = P.n.dot(L.v)
    if not d:
        # Parallel
        return None
    u = (P.k - P.n.dot(L.p)) / d
    if not L._u_in(u):
        return None
    return Point3(L.p.x + u * L.v.x,
                  L.p.y + u * L.v.y,
                  L.p.z + u * L.v.z)

def _intersect_plane_plane(A, B):
    n1_m = A.n.mag2()
    n2_m = B.n.mag2()
    n1d2 = A.n.dot(B.n)
    det = n1_m * n2_m - n1d2 ** 2
    if det == 0:
        # Parallel
        return None
    c1 = (A.k * n2_m - B.k * n1d2) / det
    c2 = (B.k * n1_m - A.k * n1d2) / det
    return Line3(Point3(c1 * A.n.x + c2 * B.n.x,
                        c1 * A.n.y + c2 * B.n.y,
                        c1 * A.n.z + c2 * B.n.z), 
                 A.n.cross(B.n))

class Point3(Vector3, Geometry):
    """
    A point on a 3D plane.  Construct in the obvious way::

        >>> p = Point3(1.0, 2.0, 3.0)
        >>> p
        Point3(1.00, 2.00, 3.00)
    
    **Point3** subclasses **Vector3**, so all of **Vector3** operators and
    methods apply.  In particular, subtracting two points gives a vector::
    
        >>> Point3(1.0, 2.0, 3.0) - Point3(1.0, 0.0, -2.0)
        Vector3(0.00, 2.00, 5.00)
    
    The following methods are also defined:
    
    ``intersect(other)``
        If *other* is a **Sphere**, returns ``True`` iff the point lies within
        the sphere.
    
    ``connect(other)``
        Returns a **LineSegment3** which is the minimum length line segment
        that can connect the two shapes.  *other* may be a **Point3**, **Line3**,
        **Ray3**, **LineSegment3**, **Sphere** or **Plane**.
    
    ``distance(other)``
        Returns the absolute minimum distance to *other*.  Internally this
        simply returns the length of the result of ``connect``. 
    """

    def intersect(self, other):
        return other._intersect_point3(self)

    def _intersect_sphere(self, other):
        return _intersect_point3_sphere(self, other)

    def connect(self, other):
        return other._connect_point3(self)

    def _connect_point3(self, other):
        if self != other:
            return Segment3(other, self)
        return None

    def _connect_line3(self, other):
        c = _connect_point3_line3(self, other)
        if c:
            return c._swap()
        
    def _connect_sphere(self, other):
        c = _connect_point3_sphere(self, other)
        if c:
            return c._swap()

    def _connect_plane(self, other):
        c = _connect_point3_plane(self, other)
        if c:
            return c._swap()

class Line3(Geometry):
    """
    A **Line3** is a line on a 3D plane extending to infinity in both directions;
    a **Ray3** has a finite end-point and extends to infinity in a single
    direction; a **LineSegment3** joins two points.  
    
    All three classes support the same constructors, operators and methods,
    but may behave differently when calculating intersections etc.
    
    You may construct a line, ray or line segment using any of:
    
    * another line, ray or line segment
    * two points
    * a point and a vector
    * a point, a vector and a length
    
    For example::
    
        >>> Line3(Point3(1.0, 1.0, 1.0), Point3(1.0, 2.0, 3.0))
        Line3(<1.00, 1.00, 1.00> + u<0.00, 1.00, 2.00>)
        >>> Line3(Point3(0.0, 1.0, 1.0), Vector3(1.0, 1.0, 2.0))
        Line3(<0.00, 1.00, 1.00> + u<1.00, 1.00, 2.00>)
        >>> Ray3(Point3(1.0, 1.0, 1.0), Vector3(1.0, 1.0, 2.0), 1.0)
        Ray3(<1.00, 1.00, 1.00> + u<0.41, 0.41, 0.82>)
    
    Internally, lines, rays and line segments store a Point3 *p* and a
    Vector3 *v*.  You can also access (but not set) the two endpoints
    *p1* and *p2*.  These may or may not be meaningful for all types of lines.
    
    The following methods are supported by all three classes:
    
    ``intersect(other)``
        If *other* is a **Sphere**, returns a **LineSegment3** which is the
        intersection of the sphere and line, or ``None`` if there is no
        intersection.
    
        If *other* is a **Plane**, returns a **Point3** of intersection, or
        ``None``.
    
    ``connect(other)``
        Returns a **LineSegment3** which is the minimum length line segment
        that can connect the two shapes.  For two parallel lines, this
        line segment may be in an arbitrary position.  *other* may be
        a **Point3**, **Line3**, **Ray3**, **LineSegment3**, **Sphere** or
        **Plane**.
    
    ``distance(other)``
        Returns the absolute minimum distance to *other*.  Internally this
        simply returns the length of the result of ``connect``.
    
    **LineSegment3** also has a *length* property which is read-only.
    """
    
    def __init__(self, *args):
        if len(args) == 3:
            assert isinstance(args[0], Point3) and \
                   isinstance(args[1], Vector3) and \
                   type(args[2]) == float
            self.p = args[0].copy()
            self.v = args[1] * args[2] / abs(args[1])
        elif len(args) == 2:
            if isinstance(args[0], Point3) and isinstance(args[1], Point3):
                self.p = args[0].copy()
                self.v = args[1] - args[0]
            elif isinstance(args[0], Point3) and isinstance(args[1], Vector3):
                self.p = args[0].copy()
                self.v = args[1].copy()
            else:
                raise AttributeError, '%r' % (args,)
        elif len(args) == 1:
            if isinstance(args[0], Line3):
                self.p = args[0].p.copy()
                self.v = args[0].v.copy()
            else:
                raise AttributeError, '%r' % (args,)
        else:
            raise AttributeError, '%r' % (args,)
        
        # XXX This is annoying.
        #if not self.v:
        #    raise AttributeError, 'Line has zero-length vector'

    def __copy__(self):
        return self.__class__(self.p, self.v)

    copy = __copy__

    def __repr__(self):
        return '%s(%s,%s)' % (self.__class__.__name__,self.p,self.v)

    p1 = property(lambda self: self.p)
    p2 = property(lambda self: Point3(self.p.x + self.v.x, 
                                      self.p.y + self.v.y,
                                      self.p.z + self.v.z))

    def _apply_transform(self, t):
        self.p = t * self.p
        self.v = t * self.v

    def _u_in(self, u):
        return True

    def intersect(self, other):
        return other._intersect_line3(self)

    def _intersect_sphere(self, other):
        return _intersect_line3_sphere(self, other)

    def _intersect_plane(self, other):
        return _intersect_line3_plane(self, other)

    def connect(self, other):
        return other._connect_line3(self)

    def _connect_point3(self, other):
        return _connect_point3_line3(other, self)

    def _connect_line3(self, other):
        return _connect_line3_line3(other, self)

    def _connect_sphere(self, other):
        return _connect_sphere_line3(other, self)

    def _connect_plane(self, other):
        c = _connect_line3_plane(self, other)
        if c:
            return c

class Ray3(Line3):
    def _u_in(self, u):
        return u >= 0.0

class Segment3(Line3):
    def __repr__(self):
        return '%s(%s,%s)' % (self.__class__.__name__,self.p,self.p2)

    def _u_in(self, u):
        return u >= 0.0 and u <= 1.0

    def __abs__(self):
        return abs(self.v)

    def mag2(self):
        return self.v.mag2()

    def _swap(self):
        # used by connect methods to switch order of points
        self.p = self.p2
        self.v *= -1
        return self

    length = property(lambda self: abs(self.v))

class Sphere(Geometry):
    """
    Spheres are constructed with a center **Point3** and a radius::

    >>> s = Sphere(Point3(1.0, 1.0, 1.0), 0.5)
    >>> s
    Sphere(<1.00, 1.00, 1.00>, radius=0.50)

    Internally there are two attributes: *c*, giving the center point and
    *r*, giving the radius.
    
    The following methods are supported:
    
    ``intersect(other)``:
        If *other* is a **Point3**, returns ``True`` iff the point lies
        within the sphere.
    
        If *other* is a **Line3**, **Ray3** or **LineSegment3**, returns
        a **LineSegment3** giving the intersection, or ``None`` if the
        line does not intersect the sphere.
    
    ``connect(other)``
        Returns a **LineSegment3** which is the minimum length line segment
        that can connect the two shapes. *other* may be a **Point3**, **Line3**,
        **Ray3**, **LineSegment3**, **Sphere** or **Plane**.
    
    ``distance(other)``
        Returns the absolute minimum distance to *other*.  Internally this
        simply returns the length of the result of ``connect``.
    """

    def __init__(self, center, radius):
        assert isinstance(center, Vector3) and type(radius) == float
        self.c = center.copy()
        self.r = radius

    def __copy__(self):
        return self.__class__(self.c, self.r)

    copy = __copy__

    def __repr__(self):
        return '%s(%s,%g)' % (self.__class__.__name__,self.c,self.r)

    def _apply_transform(self, t):
        self.c = t * self.c

    def intersect(self, other):
        return other._intersect_sphere(self)

    def _intersect_point3(self, other):
        return _intersect_point3_sphere(other, self)

    def _intersect_line3(self, other):
        return _intersect_line3_sphere(other, self)

    def connect(self, other):
        return other._connect_sphere(self)

    def _connect_point3(self, other):
        return _connect_point3_sphere(other, self)

    def _connect_line3(self, other):
        c = _connect_sphere_line3(self, other)
        if c:
            return c._swap()

    def _connect_sphere(self, other):
        return _connect_sphere_sphere(other, self)

    def _connect_plane(self, other):
        c = _connect_sphere_plane(self, other)
        if c:
            return c

class Plane:
    """
    Planes can be constructed with any of:

    * three **Point3**'s lying on the plane
    * a **Point3** on the plane and the **Vector3** normal
    * a **Vector3** normal and *k*, described below.
    
    Internally, planes are stored with the normal *n* and constant *k* such
    that *n.p* = *k* for any point on the plane *p*.
    
    The following methods are supported:
    
    ``intersect(other)``
        If *other* is a **Line3**, **Ray3** or **LineSegment3**, returns a
        **Point3** of intersection, or ``None`` if there is no intersection.
    
        If *other* is a **Plane**, returns the **Line3** of intersection.
    
    ``connect(other)``
        Returns a **LineSegment3** which is the minimum length line segment
        that can connect the two shapes. *other* may be a **Point3**, **Line3**,
        **Ray3**, **LineSegment3**, **Sphere** or **Plane**.
    
    ``distance(other)``
        Returns the absolute minimum distance to *other*.  Internally this
        simply returns the length of the result of ``connect``.
    """
    # n.p = k, where n is normal, p is point on plane, k is constant scalar

    def __init__(self, *args):
        if len(args) == 3:
            assert isinstance(args[0], Point3) and \
                   isinstance(args[1], Point3) and \
                   isinstance(args[2], Point3)
            self.n = (args[1] - args[0]).cross(args[2] - args[0])
            self.n.normalize()
            self.k = self.n.dot(args[0])
        elif len(args) == 2:
            if isinstance(args[0], Point3) and isinstance(args[1], Vector3):
                self.n = args[1].normalized()
                self.k = self.n.dot(args[0])
            elif isinstance(args[0], Vector3) and type(args[1]) == float:
                self.n = args[0].normalized()
                self.k = args[1]
            else:
                raise AttributeError, '%r' % (args,)

        else:
            raise AttributeError, '%r' % (args,)
        
        if not self.n:
            raise AttributeError, 'Points on plane are colinear'

    def __copy__(self):
        return self.__class__(self.n, self.k)

    copy = __copy__

    def __repr__(self):
        return 'Plane(<%.2f, %.2f, %.2f>.p = %.2f)' % \
            (self.n.x, self.n.y, self.n.z, self.k)

    def _get_point(self):
        # Return an arbitrary point on the plane
        if self.n.z:
            return Point3(0., 0., self.k / self.n.z)
        elif self.n.y:
            return Point3(0., self.k / self.n.y, 0.)
        else:
            return Point3(self.k / self.n.x, 0., 0.)

    def _apply_transform(self, t):
        p = t * self._get_point()
        self.n = t * self.n
        self.k = self.n.dot(p)

    def intersect(self, other):
        return other._intersect_plane(self)

    def _intersect_line3(self, other):
        return _intersect_line3_plane(other, self)

    def _intersect_plane(self, other):
        return _intersect_plane_plane(self, other)

    def connect(self, other):
        return other._connect_plane(self)

    def _connect_point3(self, other):
        return _connect_point3_plane(other, self)

    def _connect_line3(self, other):
        return _connect_line3_plane(other, self)

    def _connect_sphere(self, other):
        return _connect_sphere_plane(other, self)

    def _connect_plane(self, other):
        return _connect_plane_plane(other, self)

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

    it = p[0]
    if not hasattr(it, "__iter__"):
        return(it, it)

    values = [ x for x in p[0] ]

    #-- 4 --
    if len(values) == 1:
        return (values[0], values[0])
    else:
        return (values[0], values[1])
    