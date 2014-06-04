from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *

from Goulib.geom import *

from math import *

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
        raise SkipTest 
    
class TestPoint2:
    @classmethod
    def setup_class(self):
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
        raise SkipTest 

    def test_dist(self):
        # point2 = Point2()
        # assert_equal(expected, point2.dist(other))
        raise SkipTest 
    
class TestVector2:
    
    @classmethod
    def setup_class(self):
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
        assert_equal(self.v11//2.,Vector2(0))

    def test___rdiv__(self):
        assert_equal(2./self.v11,Vector2(2))

    def test___rfloordiv__(self):
        assert_equal(2.//self.v11,Vector2(2))

    def test___truediv__(self):
        assert_equal(operator.truediv(self.v11,3),Vector2(1/3.))
        
    def test___rtruediv__(self):
        assert_equal(operator.truediv(3,self.v11),Vector2(3.0))

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
        assert_equal(self.v11.reflect(self.v10),Vector2(-1,1))


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
        pass #tested above
    
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
        assert_equal(self.v11.mag2(), 3)
    
    def test___abs__(self):
        assert_equal(abs(self.v11), sqrt(3))
        
    def test_normalized(self):
        assert_true(self.v10.normalized())
        assert_almost_equal(abs(self.v11.normalized()),1)
        
    def test_normalize(self):
        v=self.v11.copy()
        v.normalize()
        assert_almost_equal(abs(v),1)
        
    def test___add__(self):
        assert_equal(self.v10+self.v01, self.v110) # Vector + Vector -> Vector
        assert_equal(self.v10+Point3(0,1,1),Point3(1,1,1))# Vector + Point -> Point
        assert_equal(Point3(1,0,0.5)+Point3(0,1,0.5),self.v11)# Point + Point -> Vector
        
    def test___iadd__(self):
        v=self.v10.copy()
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
        v=self.v10.copy()
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
        assert_almost_equal(self.v10.angle(self.v01),pi/2.)
        assert_almost_equal(self.v11.angle(self.v01),acos(1/sqrt(3)))

    def test_cross(self):
        assert_equal(self.v123.cross(self.v456),Vector3(-3.00, 6.00, -3.00))

    def test_project(self):
        assert_almost_equal(self.v10.project(self.v11),Vector3(1/3.,1/3.,1/3.))

    def test_reflect(self):
        assert_equal(self.v11.reflect(self.v10),Vector3(-1,1,1))

    def test_rotate_around(self):
        # vector3 = Vector3(*args)
        # assert_equal(expected, vector3.rotate_around(axis, theta))
        raise SkipTest 

class TestMatrix3:
    @classmethod
    def setup_class(self):
        self.mat123=Matrix3.new(1,2,3,4,5,6,7,8,9)

    def test___init__(self):
        assert_equal(Matrix3(), Matrix3.new(1,0,0, 0,1,0, 0,0,1))
        
    def test___repr__(self):
        assert_equal(repr(self.mat123), 'Matrix3(1, 4, 7, 2, 5, 8, 3, 6, 9)')
        
    def test_new_identity(self):
        mat=Matrix3.new_identity()
        assert_equal(mat,Matrix3())
        return mat
        
    def test_new_scale(self): 
        mat=Matrix3.new_scale(2,3)
        assert_equal(mat,Matrix3.new(2,0,0, 0,3,0, 0,0,1))
        return mat

    def test_new_rotate(self):
        mat=Matrix3.new_rotate(radians(60))
        s32=sqrt(3)/2
        res=Matrix3.new(0.5,+s32,0, -s32,0.5,0, 0,0,1) #warning : .new takes columnwise elements
        assert_almost_equal(mat, res)
        return mat

    def test_new_translate(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.new_translate(x, y))
        raise SkipTest 
    

    def test___call__(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.__call__(other))
        raise SkipTest 

    def test___copy__(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.__copy__())
        raise SkipTest 

    def test___getitem__(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.__getitem__(key))
        raise SkipTest 

    def test___imul__(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.__imul__(other))
        raise SkipTest 

    def test___mul__(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.__mul__(other))
        raise SkipTest 

    def test___setitem__(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.__setitem__(key, value))
        raise SkipTest 

    def test_angle(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.angle(angle))
        raise SkipTest 

    def test_determinant(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.determinant())
        raise SkipTest 

    def test_identity(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.identity())
        raise SkipTest 

    def test_inverse(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.inverse())
        raise SkipTest 

    def test_mag(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.mag(v))
        raise SkipTest 

    def test_offset(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.offset())
        raise SkipTest 

    def test_rotate(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.rotate(angle))
        raise SkipTest 

    def test_scale(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.scale(x, y))
        raise SkipTest 

    def test_translate(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.translate(*args))
        raise SkipTest 

    def test___abs__(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.__abs__())
        raise SkipTest 

    def test___eq__(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.__eq__(other))
        raise SkipTest 

    def test___sub__(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.__sub__(other))
        raise SkipTest 

    def test_mag2(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.mag2())
        raise SkipTest 

    def test_new(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.new(*values))
        raise SkipTest 

    def test_transpose(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.transpose())
        raise SkipTest 

    def test_transposed(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.transposed())
        raise SkipTest 

class TestMatrix4:
    @classmethod
    def setup_class(self):
        pass
    
    def test___call__(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.__call__(other))
        raise SkipTest 

    def test___copy__(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.__copy__())
        raise SkipTest 

    def test___getitem__(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.__getitem__(key))
        raise SkipTest 

    def test___imul__(self):
        # matrix4 = Matrix4()
        # assert_equal(expected, matrix4.__imul__(other))
        raise SkipTest 

    def test___init__(self):
        # matrix4 = Matrix4()
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

class TestQuaternion:
    @classmethod
    def setup_class(self):
        pass
    
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

    def test___init__(self):
        # quaternion = Quaternion(w, x, y, z)
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

class TestPolar:
    def test_polar(self):
        # assert_equal(expected, Polar(mag, angle))
        raise SkipTest 

class TestLine2:
    @classmethod
    def setup_class(self):   
        self.l1=Line2((1,1),Vector2(1,1))
        self.l2=Line2((2,2),Point2(-1,-1)) #parallel to l1
        assert_equal(self.l1.distance(self.l2),0)
        self.l3=Line2((-1,-1),(-1,1),1) #perpendicular to l1 and l2, normalized
            
    def test___init__(self):
        pass #tested above
        
    def test___copy__(self):
        pass #tested


    def test___repr__(self):
        assert_equal(repr(self.l3),"Line2(Point2(-1, -1),Vector2(0.0, 1.0))")

    def test_connect(self):
        # line2 = Line2(*args)
        # assert_equal(expected, line2.connect(other))
        raise SkipTest 

    def test_intersect(self):
        inter=self.l1.intersect(self.l2)
        assert_equal(self.l1.distance(inter),0)
        assert_equal(self.l2.distance(inter),0)
        inter=self.l2.intersect(self.l3)
        assert_equal(inter,Point2(-1,-1))
        inter=self.l1.intersect(self.l3)
        assert_equal(inter,Point2(-1,-1))

    def test_point(self):
        assert_equal(self.l1.point(0),self.l1.p)
        assert_equal(self.l1.point(1),self.l1.p+self.l1.v)

    def test_tangent(self):
        assert_equal(self.l1.tangent(0),self.l1.v)

    def test___eq__(self):
        # line2 = Line2(*args)
        # assert_equal(expected, line2.__eq__(other))
        raise SkipTest 

class TestRay2:
    def test___repr__(self):
        # ray2 = Ray2()
        # assert_equal(expected, ray2.__repr__())
        raise SkipTest 

class TestSegment2:
    @classmethod
    def setup_class(self):
        self.s1=Segment2((1,1),Vector2(1,1))
        self.s2=Segment2((2,2),Point2(-1,-1)) #parallel to l1
        self.s3=Segment2((-1,-1),(-1,1),1) #perpendicular to l1 and l2, normalized
    
    def test___abs__(self):
        # segment2 = Segment2()
        # assert_equal(expected, segment2.__abs__())
        raise SkipTest 
    
    def test_intersect(self):
        """TODO : implement intersect of colinear segments
        inter=self.s1.intersect(self.s2)
        assert_equal(self.s1.distance(inter),0)
        assert_equal(self.s2.distance(inter),0)
        """
        inter=self.s1.intersect(self.s3)
        assert_equal(inter,None)
        inter=self.s2.intersect(self.s3)
        assert_equal(inter,Point2(-1,-1))

    def test___repr__(self):
        assert_equal(repr(self.s1),"Segment2(Point2(1, 1),Point2(2, 2))")
        assert_equal(repr(self.s2),"Segment2(Point2(2, 2),Point2(-1, -1))")
        assert_equal(repr(self.s3),"Segment2(Point2(-1, -1),Point2(-1.0, 0.0))")
        
    def test_point(self):
        assert_equal(self.s1.point(0),self.s1.p1)
        assert_equal(self.s1.point(1),self.s1.p2)

    def test_tangent(self):
        assert_equal(self.s1.tangent(0),self.s1.v)
        assert_equal(self.s1.tangent(1),self.s1.v)
        assert_equal(self.s1.tangent(1.001),None)

    def test_mag2(self):
        # segment2 = Segment2()
        # assert_equal(expected, segment2.mag2())
        raise SkipTest 

    def test_swap(self):
        # segment2 = Segment2()
        # assert_equal(expected, segment2.swap())
        raise SkipTest 

class TestCircle:
    @classmethod
    def setup_class(self):
        self.c1=Circle(Point2(0,0),1)
        
    def test_point(self):
        assert_equal(self.c1.point(0),(1,0))

    def test_tangent(self):
        assert_equal(self.c1.tangent(0),(0,1))
    
    def test___copy__(self):
        # circle = Circle(center, radius)
        # assert_equal(expected, circle.__copy__())
        raise SkipTest 

    def test___init__(self):
        # circle = Circle(center, radius)
        raise SkipTest 

    def test___repr__(self):
        # circle = Circle(center, radius)
        # assert_equal(expected, circle.__repr__())
        raise SkipTest 

    def test_connect(self):
        # circle = Circle(center, radius)
        # assert_equal(expected, circle.connect(other))
        raise SkipTest 

    def test_intersect(self):
        # circle = Circle(center, radius)
        # assert_equal(expected, circle.intersect(other))
        raise SkipTest 

    def test___abs__(self):
        # circle = Circle(center, radius)
        # assert_equal(expected, circle.__abs__())
        raise SkipTest 
    
    def test___eq__(self):
        # circle = Circle(center, radius)
        # assert_equal(expected, circle.__eq__(other))
        raise SkipTest 

class TestArc2:
    @classmethod
    def setup_class(self):
        self.a1=Arc2((0,0),(1,0),(0,1))
        self.a2=Arc2((0,0),0,pi/2.,1) #same as a1
        self.a3=Arc2((0,0),pi/2.,0,1,dir=-1) #same, inverted
        self.ap=Arc2(center=Point2(454.80692478710336, 69.74749779176005),p1=Point2(74.67492478710335, 86.62949779176006),p2=Point2(74.48092478710339, 58.021497791760055),r=380.506687652)
        self.ap2=Arc2(center=Point2(-454.80607521289664, -9.203502208239968),p1=Point2(-74.67307521289663, -26.08650220823995),p2=Point2(-74.47907521289662, 2.521497791760055),r=380.507731036)
    def test___init__(self):
        pass #tested above
        assert_equal(self.a1,self.a2)
        
    def test___eq__(self):
        assert_true(self.a1==self.a2)
        assert_false(self.a1==self.a3)
        
    def test_angle(self):
        assert_equal(self.a1.angle(),self.a2.angle())
        assert_equal(self.a1.angle(),-self.a3.angle())
        assert_true(self.ap.angle()<radians(10)) # check the angle is not the complementary angle
        assert_true(self.ap2.angle()<radians(10)) # check the angle is not the complementary angle
        
    def test___abs__(self):
        assert_almost_equal(abs(self.a1),pi/2.)
        assert_true(self.ap.length<50) # check the angle is not the complementary angle
        assert_true(self.ap2.length<50) # check the angle is not the complementary angle

    def test___copy__(self):
        # arc2 = Arc2(center, p1, p2, r)
        # assert_equal(expected, arc2.__copy__())
        raise SkipTest 

    def test___repr__(self):
        # arc2 = Arc2(center, p1, p2, r)
        # assert_equal(expected, arc2.__repr__())
        raise SkipTest 

    def test_point(self):
        assert_equal(self.a1.point(0),(1,0))
        assert_equal(self.a1.point(1),(0,1))

    def test_tangent(self):
        assert_equal(self.a1.tangent(0),(0,1))
        assert_equal(self.a1.tangent(1),(-1,0))
        assert_equal(self.a2.tangent(0),(0,1))
        assert_equal(self.a2.tangent(1),(-1,0))
        #a3 direction is inverted wr to a1 and a2
        assert_equal(self.a3.tangent(0),(1,0))
        assert_equal(self.a3.tangent(1),(0,-1))
        
        a=Arc2((1,1),(3,0),(3,2))
        assert_equal(a.tangent(0),(1,2))
        assert_equal(a.tangent(1),(-1,2))

    def test_swap(self):
        # arc2 = Arc2(center, p1, p2, r, dir)
        # assert_equal(expected, arc2.swap())
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
    
class TestArgPair:
    def test_arg_pair(self):
        assert_equal(argPair(1,2),(1,2))
        assert_equal(argPair((1,2)),(1,2))
        assert_equal(argPair([1,2]),(1,2))
        
        assert_equal(argPair(1),(1,1))

if __name__ == "__main__":
    runmodule()

