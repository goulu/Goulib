#!/usr/bin/env python
# coding: utf8
from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *

from Goulib.geom3d import *

from math import *

class TestVector3:
    @classmethod
    def setup_class(self):
        self.v00=Vector3(0,0,0)
        self.v10=Vector3(1,0,0)
        self.v01=Vector3(0,1,0)
        self.v11=Vector3(1,1,1)
        self.v011=Vector3(0,1,1)
        self.v110=Vector3(1,1,0)
        self.v123=Vector3(1,2,3)
        self.v456=Vector3(4,5,6)

    def test___init__(self):
        #test copy constructor
        v10=Vector3(self.v10)
        assert_equal(v10,self.v10)
        assert_false(v10 is self.v10)

    def test___repr__(self):
        assert_equal(repr(self.v10),'Vector3(1, 0, 0)')

    def test___len__(self):
        assert_equal(len(self.v11),3)

    def test___iter__(self):
        v= [v for v in self.v11]
        assert_equal(v,[1]*len(self.v11))

    def test_xyz(self):
        assert_equal(self.v11.xyz,(1,1,1))

    def test___eq__(self):
        assert_true(self.v11 == (1,1,1))
        assert_true(self.v11 == self.v11)
        assert_false(self.v11 == self.v01)
        #more tests embedded below

    def test___ne__(self):
        assert_false(self.v11 != self.v11)
        assert_true(self.v11 != self.v01)

    def test___nonzero__(self):
        assert_true(self.v11)
        assert_false(self.v00)

    def test___copy__(self):
        v10=copy(self.v10)
        assert_true(v10==self.v10)
        assert_false(v10 is self.v10)

    def test_mag2(self):
        assert_equal(self.v11.mag2(), 3)

    def test___abs__(self):
        assert_equal(abs(self.v11), sqrt(3))

    def test_normalized(self):
        assert_true(self.v10.normalized())
        assert_equal(abs(self.v11.normalized()),1)

    def test_normalize(self):
        v=copy(self.v11)
        v.normalize()
        assert_equal(abs(v),1)

    def test___add__(self):
        assert_equal(self.v10+self.v01, self.v110) # Vector + Vector -> Vector
        assert_equal(self.v10+Point3(0,1,1),Point3(1,1,1))# Vector + Point -> Point
        assert_equal(Point3(1,0,0.5)+Point3(0,1,0.5),self.v11)# Point + Point -> Vector

    def test___iadd__(self):
        v=copy(self.v10)
        v+=self.v01
        assert_equal(v, self.v110)

    def test___neg__(self):
        assert_equal(-self.v11,Vector3(-1,-1,-1))

    def test___sub__(self):
        assert_equal(self.v11-self.v10,self.v011) # Vector - Vector -> Vector
        assert_equal(self.v11-Point3(1,0,0),Point3(0,1,1)) # Vector - Point -> Point
        assert_equal(Point3(1,1,1)-Point3(1,0,1),self.v01) # Point - Point -> Vector

    def test___rsub__(self):
        assert_equal(Point3(1,1,1)-self.v10,Point3(0,1,1)) # Point - Vector -> Point

    def test___mul__(self):
        assert_equal(2*self.v11,Vector3(2,2,2))
        assert_equal(self.v11*2,Vector3(2,2,2))

    def test___imul__(self):
        v=copy(self.v10)
        v*=2
        assert_equal(v,Vector3(2,0,0))

    def test___div__(self):
        assert_equal(self.v11/2.,Vector3(0.5,0.5,0.5))

    def test___floordiv__(self):
        assert_equal(self.v11//2.,self.v00)

    def test___rdiv__(self):
        assert_equal(2./self.v11,Vector3(2,2,2))

    def test___rfloordiv__(self):
        assert_equal(2.//self.v11,Vector3(2,2,2))

    def test___truediv__(self):
        assert_equal(operator.truediv(self.v11,3),Vector3(1/3.,1/3.,1/3.))

    def test___rtruediv__(self):
        assert_equal(operator.truediv(3,self.v11),Vector3(3,3,3))

    def test_dot(self):
        assert_equal(self.v10.dot(self.v01),0)
        assert_equal(self.v11.dot(self.v01),1)

    def test_angle(self):
        assert_equal(self.v10.angle(self.v01),pi/2.)
        assert_equal(self.v11.angle(self.v01),acos(1/sqrt(3)))

    def test_cross(self):
        assert_equal(self.v123.cross(self.v456),Vector3(-3.00, 6.00, -3.00))

    def test_project(self):
        assert_equal(self.v10.project(self.v11),Vector3(1/3.,1/3.,1/3.))

    def test_reflect(self):
        assert_equal(self.v11.reflect(self.v10),Vector3(-1,1,1))

    def test_rotate_around(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.rotate_around(axis, theta))
        raise SkipTest

    def test___bool__(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.__bool__())
        raise SkipTest

    def test___pos__(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.__pos__())
        raise SkipTest 



class TestMatrix4:
    @classmethod
    def setup_class(self):
        self.mat123=Matrix4(1,2,3,0, 4,5,6,0, 7,8,9,0, 0,0,0,1)

    def test___init__(self):
        #default constructor makes an identity matrix
        assert_equal(Matrix4(), Matrix4(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1))
        #copy constructor
        mat123=Matrix4(self.mat123)
        assert_equal(mat123,self.mat123)
        assert_false(mat123 is self.mat123)

    def test___copy__(self):
        mat123=copy(self.mat123)
        assert_equal(mat123,self.mat123)
        assert_false(mat123 is self.mat123)

    def test___call__(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.__call__(other))
        raise SkipTest

    def test___getitem__(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.__getitem__(key))
        raise SkipTest

    def test___imul__(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.__imul__(other))
        raise SkipTest

    def test___mul__(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.__mul__(other))
        raise SkipTest

    def test___repr__(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.__repr__())
        raise SkipTest

    def test___setitem__(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.__setitem__(key, value))
        raise SkipTest

    def test_determinant(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.determinant())
        raise SkipTest

    def test_identity(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.identity())
        raise SkipTest

    def test_inverse(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.inverse())
        raise SkipTest

    def test_new(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new(*values))
        raise SkipTest

    def test_new_identity(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_identity())
        raise SkipTest

    def test_new_look_at(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_look_at(eye, at, up))
        raise SkipTest

    def test_new_perspective(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_perspective(fov_y, aspect, near, far))
        raise SkipTest

    def test_new_rotate_axis(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_rotate_axis(angle, axis))
        raise SkipTest

    def test_new_rotate_euler(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_rotate_euler(heading, attitude, bank))
        raise SkipTest

    def test_new_rotate_triple_axis(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_rotate_triple_axis(x, y, z))
        raise SkipTest

    def test_new_rotatex(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_rotatex(angle))
        raise SkipTest

    def test_new_rotatey(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_rotatey(angle))
        raise SkipTest

    def test_new_rotatez(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_rotatez(angle))
        raise SkipTest

    def test_new_scale(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_scale(x, y, z))
        raise SkipTest

    def test_new_translate(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_translate(x, y, z))
        raise SkipTest

    def test_rotate_axis(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.rotate_axis(angle, axis))
        raise SkipTest

    def test_rotate_euler(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.rotate_euler(heading, attitude, bank))
        raise SkipTest

    def test_rotate_triple_axis(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.rotate_triple_axis(x, y, z))
        raise SkipTest

    def test_rotatex(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.rotatex(angle))
        raise SkipTest

    def test_rotatey(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.rotatey(angle))
        raise SkipTest

    def test_rotatez(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.rotatez(angle))
        raise SkipTest

    def test_scale(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.scale(x, y, z))
        raise SkipTest

    def test_transform(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.transform(other))
        raise SkipTest

    def test_translate(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.translate(x, y, z))
        raise SkipTest

    def test_transpose(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.transpose())
        raise SkipTest

    def test_transposed(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.transposed())
        raise SkipTest

    def test___iter__(self):
        # matrix4 = Matrix4(*args)
        # assert_equal(expected, matrix4.__iter__())
        raise SkipTest 

class TestQuaternion:
    @classmethod
    def setup_class(self):
        pass

    def test___init__(self):
        # quaternion = Quaternion(w, x, y, z)
        raise SkipTest

    def test___abs__(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.__abs__())
        raise SkipTest

    def test___copy__(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.__copy__())
        raise SkipTest

    def test___imul__(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.__imul__(other))
        raise SkipTest



    def test___mul__(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.__mul__(other))
        raise SkipTest

    def test___repr__(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.__repr__())
        raise SkipTest

    def test_conjugated(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.conjugated())
        raise SkipTest

    def test_get_angle_axis(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.get_angle_axis())
        raise SkipTest

    def test_get_euler(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.get_euler())
        raise SkipTest

    def test_get_matrix(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.get_matrix())
        raise SkipTest

    def test_identity(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.identity())
        raise SkipTest

    def test_mag2(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.mag2())
        raise SkipTest

    def test_new_identity(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.new_identity())
        raise SkipTest

    def test_new_interpolate(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.new_interpolate(q1, q2, t))
        raise SkipTest

    def test_new_rotate_axis(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.new_rotate_axis(angle, axis))
        raise SkipTest

    def test_new_rotate_euler(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.new_rotate_euler(heading, attitude, bank))
        raise SkipTest

    def test_new_rotate_matrix(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.new_rotate_matrix(m))
        raise SkipTest

    def test_normalize(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.normalize())
        raise SkipTest

    def test_normalized(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.normalized())
        raise SkipTest

    def test_rotate_axis(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.rotate_axis(angle, axis))
        raise SkipTest

    def test_rotate_euler(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.rotate_euler(heading, attitude, bank))
        raise SkipTest

    def test_rotate_matrix(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.rotate_matrix(m))
        raise SkipTest



class TestPoint3:
    @classmethod
    def setup_class(self):
        pass

    def test___repr__(self):
        # point3 = Point3()
        # assert_equal(expected, point3.__repr__())
        raise SkipTest

    def test_connect(self):
        # point3 = Point3()
        # assert_equal(expected, point3.connect(other))
        raise SkipTest

    def test_intersect(self):
        # point3 = Point3()
        # assert_equal(expected, point3.intersect(other))
        raise SkipTest

class TestLine3:
    @classmethod
    def setup_class(self):
        pass

    def test___copy__(self):
        # line3 = Line3(*args)
        # assert_equal(expected, line3.__copy__())
        raise SkipTest

    def test___init__(self):
        # line3 = Line3(*args)
        raise SkipTest

    def test___repr__(self):
        # line3 = Line3(*args)
        # assert_equal(expected, line3.__repr__())
        raise SkipTest

    def test_connect(self):
        # line3 = Line3(*args)
        # assert_equal(expected, line3.connect(other))
        raise SkipTest

    def test_intersect(self):
        # line3 = Line3(*args)
        # assert_equal(expected, line3.intersect(other))
        raise SkipTest

class TestRay3:
    def test___repr__(self):
        # ray3 = Ray3()
        # assert_equal(expected, ray3.__repr__())
        raise SkipTest

class TestSegment3:
    @classmethod
    def setup_class(self):
        pass

    def test___abs__(self):
        # segment3 = Segment3()
        # assert_equal(expected, segment3.__abs__())
        raise SkipTest

    def test___repr__(self):
        # segment3 = Segment3()
        # assert_equal(expected, segment3.__repr__())
        raise SkipTest

    def test_mag2(self):
        # segment3 = Segment3()
        # assert_equal(expected, segment3.mag2())
        raise SkipTest

    def test_swap(self):
        # segment3 = Segment3()
        # assert_equal(expected, segment3.swap())
        raise SkipTest

class TestSphere:
    @classmethod
    def setup_class(self):
        pass

    def test___copy__(self):
        # sphere = Sphere(center, radius)
        # assert_equal(expected, sphere.__copy__())
        raise SkipTest

    def test___init__(self):
        # sphere = Sphere(center, radius)
        raise SkipTest

    def test___repr__(self):
        # sphere = Sphere(center, radius)
        # assert_equal(expected, sphere.__repr__())
        raise SkipTest

    def test_connect(self):
        # sphere = Sphere(center, radius)
        # assert_equal(expected, sphere.connect(other))
        raise SkipTest

    def test_intersect(self):
        # sphere = Sphere(center, radius)
        # assert_equal(expected, sphere.intersect(other))
        raise SkipTest

class TestPlane:
    @classmethod
    def setup_class(self):
        pass

    def test___copy__(self):
        # plane = Plane(*args)
        # assert_equal(expected, plane.__copy__())
        raise SkipTest

    def test___init__(self):
        # plane = Plane(*args)
        raise SkipTest

    def test___repr__(self):
        # plane = Plane(*args)
        # assert_equal(expected, plane.__repr__())
        raise SkipTest

    def test_connect(self):
        # plane = Plane(*args)
        # assert_equal(expected, plane.connect(other))
        raise SkipTest

    def test_intersect(self):
        # plane = Plane(*args)
        # assert_equal(expected, plane.intersect(other))
        raise SkipTest

    def test_distance(self):
        # https://www.hackerrank.com/challenges/sherlock-and-planes
        p1,p2,p3,p4=Point3(1,2,0),Point3(2,3,0),Point3(4,0,0),Point3(0,0,0)
        plane=Plane(p1,p2,p3)
        assert_equal(plane.distance(p4),0)
        assert_equal(Point3(1,2,3).distance(plane),3)

if __name__ == "__main__":
    runmodule()

