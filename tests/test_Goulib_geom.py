from nose.tools import assert_equal, assert_true, assert_false, assert_almost_equal, raises
from nose import SkipTest

from Goulib.geom import *

from math import *

class TestArgPair:
    def test_arg_pair(self):
        assert_equal(argPair(1,2),(1,2))
        assert_equal(argPair((1,2)),(1,2))
        assert_equal(argPair([1,2]),(1,2))
        
        assert_equal(argPair(1),(1,1))
        

class TestVector2:
    
    def setup(self):
        self.v00=Vector2(0,0)
        self.v10=Vector2(1,0)
        self.v01=Vector2(0,1)
        self.v11=Vector2(1) #both components are 1 ...
        
    def test___init__(self):
        pass #tested above
    
    def test___repr__(self):
        assert_equal(repr(self.v10),'Vector2(1, 0)')
    
    def test___len__(self):
        assert_equal(len(self.v11),2)
    
    def test___iter__(self):
        v= [v for v in self.v11]
        assert_equal(v,[1]*len(self.v11))

    def test_xy(self):
        assert_equal(self.v11.xy,(1,1))
    
    def test___eq__(self):
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
        assert_true(self.v10.copy()==self.v10)
        assert_false(self.v10.copy()==self.v01)
        
    def test_mag2(self):
        assert_equal(self.v11.mag2(), 2)
    
    def test___abs__(self):
        assert_equal(abs(self.v11), sqrt(2))
        
    def test_normalized(self):
        assert_true(self.v10.normalized())
        assert_almost_equal(abs(self.v11.normalized()),1)
        
    def test_normalize(self):
        v=self.v11.copy()
        v.normalize()
        assert_almost_equal(abs(v),1)
        
    def test___add__(self):
        assert_equal(self.v10+self.v01, self.v11) # Vector + Vector -> Vector
        assert_equal(self.v10+Point2(0,1),Point2(1,1))# Vector + Point -> Point
        assert_equal(Point2(1,0)+Point2(0,1),self.v11)# Point + Point -> Vector
        
    def test___iadd__(self):
        v=self.v10.copy()
        v+=self.v01
        assert_equal(v, self.v11)
        
    def test___neg__(self):
        assert_equal(-self.v11,Vector2(-1,-1))

    def test___sub__(self):
        assert_equal(self.v11-self.v10,self.v01) # Vector - Vector -> Vector
        assert_equal(self.v11-Point2(1,0),Point2(0,1)) # Vector - Point -> Point
        assert_equal(Point2(1,1)-Point2(1,0),self.v01) # Point - Point -> Vector
        
    def test___rsub__(self):
        assert_equal(Point2(1,1)-self.v10,Point2(0,1)) # Point - Vector -> Point
        
    def test___mul__(self):
        assert_equal(2*self.v11,Vector2(2,2))
        assert_equal(self.v11*2,Vector2(2))
        
    def test___imul__(self):
        v=self.v10.copy()
        v*=2
        assert_equal(v,Vector2(2,0))

    def test___div__(self):
        assert_equal(self.v11/2.,Vector2(0.5))

    def test___floordiv__(self):
        # vector2 = Vector2(*args)
        # assert_equal(expected, vector2.__floordiv__(other))
        raise SkipTest # TODO: implement your test here

    def test___rdiv__(self):
        # vector2 = Vector2(*args)
        # assert_equal(expected, vector2.__rdiv__(other))
        raise SkipTest # TODO: implement your test here

    def test___rfloordiv__(self):
        # vector2 = Vector2(*args)
        # assert_equal(expected, vector2.__rfloordiv__(other))
        raise SkipTest # TODO: implement your test here

    def test___rtruediv__(self):
        # vector2 = Vector2(*args)
        # assert_equal(expected, vector2.__rtruediv__(other))
        raise SkipTest # TODO: implement your test here

    def test___truediv__(self):
        # vector2 = Vector2(*args)
        # assert_equal(expected, vector2.__truediv__(other))
        raise SkipTest # TODO: implement your test here

    def test_dot(self):
        assert_equal(self.v10.dot(self.v01),0)
        assert_equal(self.v11.dot(self.v01),1)
    
    def test_angle(self):
        assert_almost_equal(self.v10.angle(self.v01),pi/2.)
        assert_almost_equal(self.v11.angle(self.v01),pi/4.)

    def test_cross(self):
        assert_equal(self.v10.cross(),-self.v01)

    def test_project(self):
        assert_almost_equal(self.v10.project(self.v11),Vector2(.5,.5))

    def test_reflect(self):
        # vector2 = Vector2(*args)
        # assert_equal(expected, vector2.reflect(normal))
        raise SkipTest # TODO: implement your test here


class TestVector3:
    def test___abs__(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.__abs__())
        raise SkipTest # TODO: implement your test here

    def test___add__(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.__add__(other))
        raise SkipTest # TODO: implement your test here

    def test___copy__(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.__copy__())
        raise SkipTest # TODO: implement your test here

    def test___div__(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.__div__(other))
        raise SkipTest # TODO: implement your test here

    def test___eq__(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.__eq__(other))
        raise SkipTest # TODO: implement your test here

    def test___floordiv__(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.__floordiv__(other))
        raise SkipTest # TODO: implement your test here

    def test___iadd__(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.__iadd__(other))
        raise SkipTest # TODO: implement your test here

    def test___imul__(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.__imul__(other))
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # vector3 = Vector3(*args)
        raise SkipTest # TODO: implement your test here

    def test___iter__(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.__iter__())
        raise SkipTest # TODO: implement your test here

    def test___len__(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.__len__())
        raise SkipTest # TODO: implement your test here

    def test___mul__(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.__mul__(other))
        raise SkipTest # TODO: implement your test here

    def test___ne__(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.__ne__(other))
        raise SkipTest # TODO: implement your test here

    def test___neg__(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.__neg__())
        raise SkipTest # TODO: implement your test here

    def test___nonzero__(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.__nonzero__())
        raise SkipTest # TODO: implement your test here

    def test___rdiv__(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.__rdiv__(other))
        raise SkipTest # TODO: implement your test here

    def test___repr__(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.__repr__())
        raise SkipTest # TODO: implement your test here

    def test___rfloordiv__(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.__rfloordiv__(other))
        raise SkipTest # TODO: implement your test here

    def test___rsub__(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.__rsub__(other))
        raise SkipTest # TODO: implement your test here

    def test___rtruediv__(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.__rtruediv__(other))
        raise SkipTest # TODO: implement your test here

    def test___sub__(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.__sub__(other))
        raise SkipTest # TODO: implement your test here

    def test___truediv__(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.__truediv__(other))
        raise SkipTest # TODO: implement your test here

    def test_angle(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.angle(other))
        raise SkipTest # TODO: implement your test here

    def test_cross(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.cross(other))
        raise SkipTest # TODO: implement your test here

    def test_dot(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.dot(other))
        raise SkipTest # TODO: implement your test here

    def test_mag2(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.mag2())
        raise SkipTest # TODO: implement your test here

    def test_normalize(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.normalize())
        raise SkipTest # TODO: implement your test here

    def test_normalized(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.normalized())
        raise SkipTest # TODO: implement your test here

    def test_project(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.project(other))
        raise SkipTest # TODO: implement your test here

    def test_reflect(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.reflect(normal))
        raise SkipTest # TODO: implement your test here

    def test_rotate_around(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.rotate_around(axis, theta))
        raise SkipTest # TODO: implement your test here

    def test_xyz(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.xyz())
        raise SkipTest # TODO: implement your test here

class TestMatrix3:
    def test___call__(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.__call__(other))
        raise SkipTest # TODO: implement your test here

    def test___copy__(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.__copy__())
        raise SkipTest # TODO: implement your test here

    def test___getitem__(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.__getitem__(key))
        raise SkipTest # TODO: implement your test here

    def test___imul__(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.__imul__(other))
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # matrix3 = Matrix3()
        raise SkipTest # TODO: implement your test here

    def test___mul__(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.__mul__(other))
        raise SkipTest # TODO: implement your test here

    def test___repr__(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.__repr__())
        raise SkipTest # TODO: implement your test here

    def test___setitem__(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.__setitem__(key, value))
        raise SkipTest # TODO: implement your test here

    def test_angle(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.angle(angle))
        raise SkipTest # TODO: implement your test here

    def test_determinant(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.determinant())
        raise SkipTest # TODO: implement your test here

    def test_identity(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.identity())
        raise SkipTest # TODO: implement your test here

    def test_inverse(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.inverse())
        raise SkipTest # TODO: implement your test here

    def test_mag(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.mag(v))
        raise SkipTest # TODO: implement your test here

    def test_new_identity(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.new_identity())
        raise SkipTest # TODO: implement your test here

    def test_new_rotate(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.new_rotate(angle))
        raise SkipTest # TODO: implement your test here

    def test_new_scale(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.new_scale(x, y))
        raise SkipTest # TODO: implement your test here

    def test_new_translate(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.new_translate(x, y))
        raise SkipTest # TODO: implement your test here

    def test_offset(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.offset())
        raise SkipTest # TODO: implement your test here

    def test_rotate(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.rotate(angle))
        raise SkipTest # TODO: implement your test here

    def test_scale(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.scale(x, y))
        raise SkipTest # TODO: implement your test here

    def test_translate(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.translate(*args))
        raise SkipTest # TODO: implement your test here

class TestMatrix4:
    def test___call__(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.__call__(other))
        raise SkipTest # TODO: implement your test here

    def test___copy__(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.__copy__())
        raise SkipTest # TODO: implement your test here

    def test___getitem__(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.__getitem__(key))
        raise SkipTest # TODO: implement your test here

    def test___imul__(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.__imul__(other))
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # matrix4 = Matrix4()
        raise SkipTest # TODO: implement your test here

    def test___mul__(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.__mul__(other))
        raise SkipTest # TODO: implement your test here

    def test___repr__(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.__repr__())
        raise SkipTest # TODO: implement your test here

    def test___setitem__(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.__setitem__(key, value))
        raise SkipTest # TODO: implement your test here

    def test_determinant(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.determinant())
        raise SkipTest # TODO: implement your test here

    def test_identity(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.identity())
        raise SkipTest # TODO: implement your test here

    def test_inverse(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.inverse())
        raise SkipTest # TODO: implement your test here

    def test_new(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new(*values))
        raise SkipTest # TODO: implement your test here

    def test_new_identity(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_identity())
        raise SkipTest # TODO: implement your test here

    def test_new_look_at(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_look_at(eye, at, up))
        raise SkipTest # TODO: implement your test here

    def test_new_perspective(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_perspective(fov_y, aspect, near, far))
        raise SkipTest # TODO: implement your test here

    def test_new_rotate_axis(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_rotate_axis(angle, axis))
        raise SkipTest # TODO: implement your test here

    def test_new_rotate_euler(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_rotate_euler(heading, attitude, bank))
        raise SkipTest # TODO: implement your test here

    def test_new_rotate_triple_axis(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_rotate_triple_axis(x, y, z))
        raise SkipTest # TODO: implement your test here

    def test_new_rotatex(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_rotatex(angle))
        raise SkipTest # TODO: implement your test here

    def test_new_rotatey(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_rotatey(angle))
        raise SkipTest # TODO: implement your test here

    def test_new_rotatez(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_rotatez(angle))
        raise SkipTest # TODO: implement your test here

    def test_new_scale(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_scale(x, y, z))
        raise SkipTest # TODO: implement your test here

    def test_new_translate(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.new_translate(x, y, z))
        raise SkipTest # TODO: implement your test here

    def test_rotate_axis(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.rotate_axis(angle, axis))
        raise SkipTest # TODO: implement your test here

    def test_rotate_euler(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.rotate_euler(heading, attitude, bank))
        raise SkipTest # TODO: implement your test here

    def test_rotate_triple_axis(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.rotate_triple_axis(x, y, z))
        raise SkipTest # TODO: implement your test here

    def test_rotatex(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.rotatex(angle))
        raise SkipTest # TODO: implement your test here

    def test_rotatey(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.rotatey(angle))
        raise SkipTest # TODO: implement your test here

    def test_rotatez(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.rotatez(angle))
        raise SkipTest # TODO: implement your test here

    def test_scale(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.scale(x, y, z))
        raise SkipTest # TODO: implement your test here

    def test_transform(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.transform(other))
        raise SkipTest # TODO: implement your test here

    def test_translate(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.translate(x, y, z))
        raise SkipTest # TODO: implement your test here

    def test_transpose(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.transpose())
        raise SkipTest # TODO: implement your test here

    def test_transposed(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.transposed())
        raise SkipTest # TODO: implement your test here

class TestQuaternion:
    def test___abs__(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.__abs__())
        raise SkipTest # TODO: implement your test here

    def test___copy__(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.__copy__())
        raise SkipTest # TODO: implement your test here

    def test___imul__(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.__imul__(other))
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # quaternion = Quaternion(w, x, y, z)
        raise SkipTest # TODO: implement your test here

    def test___mul__(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.__mul__(other))
        raise SkipTest # TODO: implement your test here

    def test___repr__(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.__repr__())
        raise SkipTest # TODO: implement your test here

    def test_conjugated(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.conjugated())
        raise SkipTest # TODO: implement your test here

    def test_get_angle_axis(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.get_angle_axis())
        raise SkipTest # TODO: implement your test here

    def test_get_euler(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.get_euler())
        raise SkipTest # TODO: implement your test here

    def test_get_matrix(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.get_matrix())
        raise SkipTest # TODO: implement your test here

    def test_identity(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.identity())
        raise SkipTest # TODO: implement your test here

    def test_mag2(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.mag2())
        raise SkipTest # TODO: implement your test here

    def test_new_identity(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.new_identity())
        raise SkipTest # TODO: implement your test here

    def test_new_interpolate(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.new_interpolate(q1, q2, t))
        raise SkipTest # TODO: implement your test here

    def test_new_rotate_axis(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.new_rotate_axis(angle, axis))
        raise SkipTest # TODO: implement your test here

    def test_new_rotate_euler(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.new_rotate_euler(heading, attitude, bank))
        raise SkipTest # TODO: implement your test here

    def test_new_rotate_matrix(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.new_rotate_matrix(m))
        raise SkipTest # TODO: implement your test here

    def test_normalize(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.normalize())
        raise SkipTest # TODO: implement your test here

    def test_normalized(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.normalized())
        raise SkipTest # TODO: implement your test here

    def test_rotate_axis(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.rotate_axis(angle, axis))
        raise SkipTest # TODO: implement your test here

    def test_rotate_euler(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.rotate_euler(heading, attitude, bank))
        raise SkipTest # TODO: implement your test here

    def test_rotate_matrix(self):
        # quaternion = Quaternion(w, x, y, z)
        # assert_equal(expected, quaternion.rotate_matrix(m))
        raise SkipTest # TODO: implement your test here

class TestGeometry:
    #tested in derived classes
    @raises(NotImplementedError)
    def test_connect(self):
        Geometry().connect(1)

    @raises(NotImplementedError)
    def test_distance(self):
        Geometry().distance(1)

    @raises(NotImplementedError)
    def test_intersect(self):
        Geometry().intersect(1)
    
    @raises(NotImplementedError)
    def test_point(self):
        Geometry().point(1)

    @raises(NotImplementedError)
    def test_tangent(self):
        Geometry().tangent(1)

    def test_intersect_case_2(self):
        # geometry = Geometry()
        # assert_equal(expected, geometry.intersect(other))
        raise SkipTest # TODO: implement your test here

class TestPoint2:
    
    def setup(self):
        self.p00=Point2(0,0)
        self.p10=Point2(1,0)
        self.p01=Point2(0,1)
        self.p11=Point2(1) #both components are 1 ...
        
    def test___repr__(self):
        assert_equal(repr(self.p10),'Point2(1, 0)')

    def test_connect(self):
        assert_equal(self.p10.connect(self.p01),Segment2(self.p10,self.p01))

    def test_distance(self):
        assert_equal(self.p10.distance(self.p10),0)
        assert_equal(self.p10.distance(self.p00),1)
        assert_equal(self.p10.distance(self.p01),sqrt(2))

    def test_intersect(self):
        # point2 = Point2()
        # assert_equal(expected, point2.intersect(other))
        raise SkipTest # TODO: implement your test here

    def test_dist(self):
        # point2 = Point2()
        # assert_equal(expected, point2.dist(other))
        raise SkipTest # TODO: implement your test here

class TestPolar:
    def test_polar(self):
        # assert_equal(expected, Polar(mag, angle))
        raise SkipTest # TODO: implement your test here

class TestLine2:
    def setup(self):
        self.l1=Line2((1,1),(1,1))
        self.l2=Line2((2,2),Point2(-1,-1)) #parallel to l1
        self.l3=Line2((-1,-1),(-1,1),1) #perpendicular to l1 and l2, normalized
            
    def test___init__(self):
        pass #tested above
        
    def test___copy__(self):
        pass #tested


    def test___repr__(self):
        # line2 = Line2(*args)
        # assert_equal(expected, line2.__repr__())
        raise SkipTest # TODO: implement your test here

    def test_connect(self):
        # line2 = Line2(*args)
        # assert_equal(expected, line2.connect(other))
        raise SkipTest # TODO: implement your test here

    def test_intersect(self):
        # line2 = Line2(*args)
        # assert_equal(expected, line2.intersect(other))
        raise SkipTest # TODO: implement your test here

    def test_point(self):
        # line2 = Line2(*args)
        # assert_equal(expected, line2.point(u))
        raise SkipTest # TODO: implement your test here

    def test_tangent(self):
        # line2 = Line2(*args)
        # assert_equal(expected, line2.tangent(u))
        raise SkipTest # TODO: implement your test here

    def test___eq__(self):
        # line2 = Line2(*args)
        # assert_equal(expected, line2.__eq__(other))
        raise SkipTest # TODO: implement your test here

class TestRay2:
    def test___repr__(self):
        # ray2 = Ray2()
        # assert_equal(expected, ray2.__repr__())
        raise SkipTest # TODO: implement your test here

class TestSegment2:
    def test___abs__(self):
        # segment2 = Segment2()
        # assert_equal(expected, segment2.__abs__())
        raise SkipTest # TODO: implement your test here

    def test___repr__(self):
        # segment2 = Segment2()
        # assert_equal(expected, segment2.__repr__())
        raise SkipTest # TODO: implement your test here

    def test_mag2(self):
        # segment2 = Segment2()
        # assert_equal(expected, segment2.mag2())
        raise SkipTest # TODO: implement your test here

class TestCircle:
    def test___copy__(self):
        # circle = Circle(center, radius)
        # assert_equal(expected, circle.__copy__())
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # circle = Circle(center, radius)
        raise SkipTest # TODO: implement your test here

    def test___repr__(self):
        # circle = Circle(center, radius)
        # assert_equal(expected, circle.__repr__())
        raise SkipTest # TODO: implement your test here

    def test_connect(self):
        # circle = Circle(center, radius)
        # assert_equal(expected, circle.connect(other))
        raise SkipTest # TODO: implement your test here

    def test_intersect(self):
        # circle = Circle(center, radius)
        # assert_equal(expected, circle.intersect(other))
        raise SkipTest # TODO: implement your test here

    def test___abs__(self):
        # circle = Circle(center, radius)
        # assert_equal(expected, circle.__abs__())
        raise SkipTest # TODO: implement your test here
    
    def test___eq__(self):
        # circle = Circle(center, radius)
        # assert_equal(expected, circle.__eq__(other))
        raise SkipTest # TODO: implement your test here

    def test_point(self):
        # circle = Circle(center, radius)
        # assert_equal(expected, circle.point(u))
        raise SkipTest # TODO: implement your test here

    def test_tangent(self):
        # circle = Circle(center, radius)
        # assert_equal(expected, circle.tangent(u))
        raise SkipTest # TODO: implement your test here

class TestArc2:
    def setup(self):
        self.a1=Arc2((0,0),(1,0),(0,1))
        self.a2=Arc2((0,0),0,pi/2.,1) #same as a1
        
    def test___init__(self):
        pass #tested above
        assert_equal(self.a1,self.a2)
        
    def test___eq__(self):
        assert_true(self.a1==self.a2)
        
    def test___abs__(self):
        assert_almost_equal(abs(self.a1),pi/2.)
        raise SkipTest # TODO: implement your test here

    def test___copy__(self):
        # arc2 = Arc2(center, p1, p2, r)
        # assert_equal(expected, arc2.__copy__())
        raise SkipTest # TODO: implement your test here

    def test___repr__(self):
        # arc2 = Arc2(center, p1, p2, r)
        # assert_equal(expected, arc2.__repr__())
        raise SkipTest # TODO: implement your test here

    def test_angle(self):
        # arc2 = Arc2(center, p1, p2, r)
        # assert_equal(expected, arc2.angle())
        raise SkipTest # TODO: implement your test here

    def test_point(self):
        # arc2 = Arc2(center, p1, p2, r)
        # assert_equal(expected, arc2.point(u))
        raise SkipTest # TODO: implement your test here

    def test_tangent(self):
        # arc2 = Arc2(center, p1, p2, r)
        # assert_equal(expected, arc2.tangent(u))
        raise SkipTest # TODO: implement your test here

class TestPoint3:
    def test___repr__(self):
        # point3 = Point3()
        # assert_equal(expected, point3.__repr__())
        raise SkipTest # TODO: implement your test here

    def test_connect(self):
        # point3 = Point3()
        # assert_equal(expected, point3.connect(other))
        raise SkipTest # TODO: implement your test here

    def test_intersect(self):
        # point3 = Point3()
        # assert_equal(expected, point3.intersect(other))
        raise SkipTest # TODO: implement your test here

class TestLine3:
    def test___copy__(self):
        # line3 = Line3(*args)
        # assert_equal(expected, line3.__copy__())
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # line3 = Line3(*args)
        raise SkipTest # TODO: implement your test here

    def test___repr__(self):
        # line3 = Line3(*args)
        # assert_equal(expected, line3.__repr__())
        raise SkipTest # TODO: implement your test here

    def test_connect(self):
        # line3 = Line3(*args)
        # assert_equal(expected, line3.connect(other))
        raise SkipTest # TODO: implement your test here

    def test_intersect(self):
        # line3 = Line3(*args)
        # assert_equal(expected, line3.intersect(other))
        raise SkipTest # TODO: implement your test here

class TestRay3:
    def test___repr__(self):
        # ray3 = Ray3()
        # assert_equal(expected, ray3.__repr__())
        raise SkipTest # TODO: implement your test here

class TestSegment3:
    def test___abs__(self):
        # segment3 = Segment3()
        # assert_equal(expected, segment3.__abs__())
        raise SkipTest # TODO: implement your test here

    def test___repr__(self):
        # segment3 = Segment3()
        # assert_equal(expected, segment3.__repr__())
        raise SkipTest # TODO: implement your test here

    def test_mag2(self):
        # segment3 = Segment3()
        # assert_equal(expected, segment3.mag2())
        raise SkipTest # TODO: implement your test here

class TestSphere:
    def test___copy__(self):
        # sphere = Sphere(center, radius)
        # assert_equal(expected, sphere.__copy__())
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # sphere = Sphere(center, radius)
        raise SkipTest # TODO: implement your test here

    def test___repr__(self):
        # sphere = Sphere(center, radius)
        # assert_equal(expected, sphere.__repr__())
        raise SkipTest # TODO: implement your test here

    def test_connect(self):
        # sphere = Sphere(center, radius)
        # assert_equal(expected, sphere.connect(other))
        raise SkipTest # TODO: implement your test here

    def test_intersect(self):
        # sphere = Sphere(center, radius)
        # assert_equal(expected, sphere.intersect(other))
        raise SkipTest # TODO: implement your test here

class TestPlane:
    def test___copy__(self):
        # plane = Plane(*args)
        # assert_equal(expected, plane.__copy__())
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # plane = Plane(*args)
        raise SkipTest # TODO: implement your test here

    def test___repr__(self):
        # plane = Plane(*args)
        # assert_equal(expected, plane.__repr__())
        raise SkipTest # TODO: implement your test here

    def test_connect(self):
        # plane = Plane(*args)
        # assert_equal(expected, plane.connect(other))
        raise SkipTest # TODO: implement your test here

    def test_intersect(self):
        # plane = Plane(*args)
        # assert_equal(expected, plane.intersect(other))
        raise SkipTest # TODO: implement your test here

if __name__ == "__main__":
    import nose
    nose.runmodule()
