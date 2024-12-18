#!/usr/bin/env python
# coding: utf8
from nose.tools import assert_equal
from nose import SkipTest
# lines above are inserted automatically by pythoscope. Line below overrides them
from goulib.tests import *

from goulib.geom3d import *

from math import *


class TestVector3:
    @classmethod
    def setup_class(self):
        self.v00 = Vector3(0, 0, 0)
        self.v10 = Vector3(1, 0, 0)
        self.v01 = Vector3(0, 1, 0)
        self.v11 = Vector3(1, 1, 1)
        self.v011 = Vector3(0, 1, 1)
        self.v110 = Vector3(1, 1, 0)
        self.v123 = Vector3(1, 2, 3)
        self.v456 = Vector3(4, 5, 6)

    def test___init__(self):
        # test copy constructor
        v10 = Vector3(self.v10)
        assert v10 == self.v10
        assert not v10 is self.v10

    def test___repr__(self):
        assert repr(self.v10) == 'Vector3(1, 0, 0)'

    def test___len__(self):
        assert len(self.v11) == 3

    def test___iter__(self):
        v = [v for v in self.v11]
        assert v == [1]*len(self.v11)

    def test_xyz(self):
        assert self.v11.xyz == (1, 1, 1)

    def test___eq__(self):
        assert self.v11 == (1, 1, 1)
        assert self.v11 == self.v11
        assert not self.v11 == self.v01
        # more tests embedded below

    def test___ne__(self):
        assert not self.v11 != self.v11
        assert self.v11 != self.v01

    def test___nonzero__(self):
        assert self.v11
        assert not self.v00

    def test___copy__(self):
        v10 = copy(self.v10)
        assert v10 == self.v10
        assert not v10 is self.v10

    def test_mag2(self):
        assert self.v11.mag2() == 3

    def test___abs__(self):
        assert abs(self.v11) == sqrt(3)

    def test_normalized(self):
        assert self.v10.normalized()
        assert abs(self.v11.normalized()) == 1

    def test_normalize(self):
        v = copy(self.v11)
        v.normalize()
        assert abs(v) == 1

    def test___add__(self):
        assert self.v10+self.v01 == self.v110  # Vector + Vector -> Vector
        # Vector + Point -> Point
        assert self.v10+Point3(0, 1, 1) == Point3(1, 1, 1)
        # Point + Point -> Vector
        assert Point3(1, 0, 0.5)+Point3(0, 1, 0.5) == self.v11

    def test___iadd__(self):
        v = copy(self.v10)
        v += self.v01
        assert v == self.v110

    def test___neg__(self):
        assert -self.v11 == Vector3(-1, -1, -1)

    def test___sub__(self):
        assert self.v11-self.v10 == self.v011  # Vector - Vector -> Vector
        # Vector - Point -> Point
        assert self.v11-Point3(1, 0, 0) == Point3(0, 1, 1)
        # Point - Point -> Vector
        assert Point3(1, 1, 1)-Point3(1, 0, 1) == self.v01

    def test___rsub__(self):
        assert Point3(1, 1, 1)-self.v10 == Point3(0,
                                                  1, 1)  # Point - Vector -> Point

    def test___mul__(self):
        assert 2*self.v11 == Vector3(2, 2, 2)
        assert self.v11*2 == Vector3(2, 2, 2)

    def test___imul__(self):
        v = copy(self.v10)
        v *= 2
        assert v == Vector3(2, 0, 0)

    def test___div__(self):
        assert self.v11/2. == Vector3(0.5, 0.5, 0.5)

    def test___floordiv__(self):
        assert self.v11//2. == self.v00

    def test___rdiv__(self):
        assert 2./self.v11 == Vector3(2, 2, 2)

    def test___rfloordiv__(self):
        assert 2.//self.v11 == Vector3(2, 2, 2)

    def test___truediv__(self):
        assert operator.truediv(self.v11, 3) == Vector3(1/3., 1/3., 1/3.)

    def test___rtruediv__(self):
        assert operator.truediv(3, self.v11) == Vector3(3, 3, 3)

    def test_dot(self):
        assert self.v10.dot(self.v01) == 0
        assert self.v11.dot(self.v01) == 1

    def test_angle(self):
        assert self.v10.angle(self.v01) == pi/2.
        assert self.v11.angle(self.v01) == acos(1/sqrt(3))

    def test_cross(self):
        assert self.v123.cross(self.v456) == Vector3(-3.00, 6.00, -3.00)

    def test_project(self):
        assert self.v10.project(self.v11) == Vector3(1/3., 1/3., 1/3.)

    def test_reflect(self):
        assert self.v11.reflect(self.v10) == Vector3(-1, 1, 1)

    def test_rotate_around(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.rotate_around(axis, theta))
        pass  # TODO: implement

    def test___bool__(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.__bool__())
        pass  # TODO: implement

    def test___pos__(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.__pos__())
        pass  # TODO: implement


class TestMatrix4:
    @classmethod
    def setup_class(self):
        self.mat123 = Matrix4(1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0, 0, 0, 0, 1)

    def test___init__(self):
        # default constructor makes an identity matrix
        assert Matrix4() == Matrix4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
        # copy constructor
        mat123 = Matrix4(self.mat123)
        assert mat123 == self.mat123
        assert not mat123 is self.mat123

    def test___copy__(self):
        mat123 = copy(self.mat123)
        assert mat123 == self.mat123
        assert not mat123 is self.mat123

    def test___call__(self):
        assert self.mat123(Vector3(1, 10, 100)) == (741, 852, 963)

    def test___getitem__(self):
        assert self.mat123[1*4+2] == 6

    def test___imul__(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.__imul__(other))
        pass  # TODO: implement

    def test___mul__(self):
        assert self.mat123*Vector3(1, 10, 100) == (741, 852, 963)

    def test___repr__(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.__repr__())
        pass  # TODO: implement

    def test___setitem__(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.__setitem__(key, value))
        pass  # TODO: implement

    def test_determinant(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.determinant())
        pass  # TODO: implement

    def test_identity(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.identity())
        pass  # TODO: implement

    def test_inverse(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.inverse())
        pass  # TODO: implement

    def test_new(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new(*values))
        pass  # TODO: implement

    def test_new_identity(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_identity())
        pass  # TODO: implement

    def test_new_look_at(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_look_at(eye, at, up))
        pass  # TODO: implement

    def test_new_perspective(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_perspective(fov_y, aspect, near, far))
        pass  # TODO: implement

    def test_new_rotate_axis(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_rotate_axis(angle, axis))
        pass  # TODO: implement

    def test_new_rotate_euler(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_rotate_euler(heading, attitude, bank))
        pass  # TODO: implement

    def test_new_rotate_triple_axis(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_rotate_triple_axis(x, y, z))
        pass  # TODO: implement

    def test_new_rotatex(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_rotatex(angle))
        pass  # TODO: implement

    def test_new_rotatey(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_rotatey(angle))
        pass  # TODO: implement

    def test_new_rotatez(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_rotatez(angle))
        pass  # TODO: implement

    def test_new_scale(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_scale(x, y, z))
        pass  # TODO: implement

    def test_new_translate(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_translate(x, y, z))
        pass  # TODO: implement

    def test_rotate_axis(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.rotate_axis(angle, axis))
        pass  # TODO: implement

    def test_rotate_euler(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.rotate_euler(heading, attitude, bank))
        pass  # TODO: implement

    def test_rotate_triple_axis(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.rotate_triple_axis(x, y, z))
        pass  # TODO: implement

    def test_rotatex(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.rotatex(angle))
        pass  # TODO: implement

    def test_rotatey(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.rotatey(angle))
        pass  # TODO: implement

    def test_rotatez(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.rotatez(angle))
        pass  # TODO: implement

    def test_scale(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.scale(x, y, z))
        pass  # TODO: implement

    def test_transform(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.transform(other))
        pass  # TODO: implement

    def test_translate(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.translate(x, y, z))
        pass  # TODO: implement

    def test_transpose(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.transpose())
        pass  # TODO: implement

    def test_transposed(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.transposed())
        pass  # TODO: implement

    def test___iter__(self):
        # matrix4 = Matrix4(*args)
        # assert_equal(expected, matrix4.__iter__())
        pass  # TODO: implement


class TestQuaternion:
    @classmethod
    def setup_class(self):
        pass

    def test___init__(self):
        # quaternion = Quaternion(w, x, y, z)
        pass  # TODO: implement

    def test___abs__(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.__abs__())
        pass  # TODO: implement

    def test___copy__(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.__copy__())
        pass  # TODO: implement

    def test___imul__(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.__imul__(other))
        pass  # TODO: implement

    def test___mul__(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.__mul__(other))
        pass  # TODO: implement

    def test___repr__(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.__repr__())
        pass  # TODO: implement

    def test_conjugated(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.conjugated())
        pass  # TODO: implement

    def test_get_angle_axis(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.get_angle_axis())
        pass  # TODO: implement

    def test_get_euler(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.get_euler())
        pass  # TODO: implement

    def test_get_matrix(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.get_matrix())
        pass  # TODO: implement

    def test_identity(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.identity())
        pass  # TODO: implement

    def test_mag2(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.mag2())
        pass  # TODO: implement

    def test_new_identity(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.new_identity())
        pass  # TODO: implement

    def test_new_interpolate(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.new_interpolate(q1, q2, t))
        pass  # TODO: implement

    def test_new_rotate_axis(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.new_rotate_axis(angle, axis))
        pass  # TODO: implement

    def test_new_rotate_euler(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.new_rotate_euler(heading, attitude, bank))
        pass  # TODO: implement

    def test_new_rotate_matrix(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.new_rotate_matrix(m))
        pass  # TODO: implement

    def test_normalize(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.normalize())
        pass  # TODO: implement

    def test_normalized(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.normalized())
        pass  # TODO: implement

    def test_rotate_axis(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.rotate_axis(angle, axis))
        pass  # TODO: implement

    def test_rotate_euler(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.rotate_euler(heading, attitude, bank))
        pass  # TODO: implement

    def test_rotate_matrix(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.rotate_matrix(m))
        pass  # TODO: implement


class TestPoint3:
    @classmethod
    def setup_class(self):
        pass

    def test___repr__(self):
        # point3 = Point3()
        # assert_equal(expected, point3.__repr__())
        pass  # TODO: implement

    def test_connect(self):
        # point3 = Point3()
        # assert_equal(expected, point3.connect(other))
        pass  # TODO: implement

    def test_intersect(self):
        # point3 = Point3()
        # assert_equal(expected, point3.intersect(other))
        pass  # TODO: implement


class TestLine3:
    @classmethod
    def setup_class(self):
        pass

    def test___copy__(self):
        # line3 = Line3(*args)
        # assert_equal(expected, line3.__copy__())
        pass  # TODO: implement

    def test___init__(self):
        # line3 = Line3(*args)
        pass  # TODO: implement

    def test___repr__(self):
        # line3 = Line3(*args)
        # assert_equal(expected, line3.__repr__())
        pass  # TODO: implement

    def test_connect(self):
        # line3 = Line3(*args)
        # assert_equal(expected, line3.connect(other))
        pass  # TODO: implement

    def test_intersect(self):
        # line3 = Line3(*args)
        # assert_equal(expected, line3.intersect(other))
        pass  # TODO: implement

    def test_point(self):
        # line3 = Line3(*args)
        # assert_equal(expected, line3.point(u))
        pass  # TODO: implement  # implement your test here


class TestRay3:
    def test___repr__(self):
        # ray3 = Ray3()
        # assert_equal(expected, ray3.__repr__())
        pass  # TODO: implement


class TestSegment3:
    @classmethod
    def setup_class(self):
        pass

    def test___abs__(self):
        # segment3 = Segment3()
        # assert_equal(expected, segment3.__abs__())
        pass  # TODO: implement

    def test___repr__(self):
        # segment3 = Segment3()
        # assert_equal(expected, segment3.__repr__())
        pass  # TODO: implement

    def test_mag2(self):
        # segment3 = Segment3()
        # assert_equal(expected, segment3.mag2())
        pass  # TODO: implement

    def test_swap(self):
        # segment3 = Segment3()
        # assert_equal(expected, segment3.swap())
        pass  # TODO: implement


class TestSphere:
    @classmethod
    def setup_class(self):
        pass

    def test___copy__(self):
        # sphere = Sphere(center, radius)
        # assert_equal(expected, sphere.__copy__())
        pass  # TODO: implement

    def test___init__(self):
        # sphere = Sphere(center, radius)
        pass  # TODO: implement

    def test___repr__(self):
        # sphere = Sphere(center, radius)
        # assert_equal(expected, sphere.__repr__())
        pass  # TODO: implement

    def test_connect(self):
        # sphere = Sphere(center, radius)
        # assert_equal(expected, sphere.connect(other))
        pass  # TODO: implement

    def test_intersect(self):
        # sphere = Sphere(center, radius)
        # assert_equal(expected, sphere.intersect(other))
        pass  # TODO: implement

    def test___contains__(self):
        # sphere = Sphere(*args)
        # assert_equal(expected, sphere.__contains__(pt))
        pass  # TODO: implement

    def test_distance_on_sphere(self):
        # sphere = Sphere(*args)
        # assert_equal(expected, sphere.distance_on_sphere(phi1, theta1, phi2, theta2))
        pass  # TODO: implement

    def test_point(self):
        # sphere = Sphere(*args)
        # assert_equal(expected, sphere.point(u, v))
        pass  # TODO: implement


class TestPlane:
    @classmethod
    def setup_class(self):
        u = Vector3(2, 2, 1)
        v = Vector3(2, 1, 1)
        p = Plane((0, 0, 0), u, v)

    def test___copy__(self):
        # plane = Plane(*args)
        # assert_equal(expected, plane.__copy__())
        pass  # TODO: implement

    def test___init__(self):
        # plane = Plane(*args)
        pass  # TODO: implement

    def test___repr__(self):
        # plane = Plane(*args)
        # assert_equal(expected, plane.__repr__())
        pass  # TODO: implement

    def test_connect(self):
        # plane = Plane(*args)
        # assert_equal(expected, plane.connect(other))
        pass  # TODO: implement

    def test_intersect(self):
        # plane = Plane(*args)
        # assert_equal(expected, plane.intersect(other))
        pass  # TODO: implement

    def test_distance(self):
        # https://www.hackerrank.com/challenges/sherlock-and-planes
        p1, p2, p3, p4 = Point3(1, 2, 0), Point3(
            2, 3, 0), Point3(4, 0, 0), Point3(0, 0, 0)
        plane = Plane(p1, p2, p3)
        assert plane.distance(p4) == 0
        assert Point3(1, 2, 3).distance(plane) == 3


class TestSpherical:
    def test_spherical(self):
        # assert_equal(expected, Spherical(r, theta, phi))
        pass  # TODO: implement


if __name__ == "__main__":
    runmodule()
