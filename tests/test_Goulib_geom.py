#!/usr/bin/env python
# coding: utf8
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

    def test___init__(self):
        # geometry = Geometry(*args)
        raise SkipTest 

    def test___contains__(self):
        # geometry = Geometry(*args)
        # assert_equal(expected, geometry.__contains__(pt))
        raise SkipTest 

class TestPoint2:
    @classmethod
    def setup_class(self):
        self.p00=Point2(0,0)
        self.p10=Point2(1,0)
        self.p01=Point2(0,1)
        self.p11=Point2(1,1)
        self.diag=Segment2(self.p10,self.p01)
        self.circle=Circle((1,2),1)

    def test___repr__(self):
        assert_equal(repr(self.p10),'Point2(1, 0)')

    def test_connect(self):
        assert_equal(self.p10.connect(self.p01),self.diag)
        assert_equal(self.p10.connect(self.circle),Segment2(self.p10,self.p11))
        assert_equal(self.circle.connect(self.p10),Segment2(self.p11,self.p10))


    def test_distance(self):
        assert_equal(self.p10.distance(self.p10),0)
        assert_equal(self.p10.distance(self.p00),1)
        assert_equal(self.p10.distance(self.p01),sqrt(2))
        
        assert_equal(self.p00.distance(self.diag),sqrt(2)/2)
        assert_equal(self.p11.distance(self.diag),sqrt(2)/2)

    def test_intersect(self):
        p=self.p11.intersect(self.p11) #Point2 intersects with itself
        assert_equal(p,self.p11)
        assert_false(p is self.p11) #but intersect should return a COPY
        
        assert_equal(self.p11.intersect(self.p10),None)
        
        assert_true(self.p10 in self.diag)
        assert_true(self.p01 in self.diag)
        assert_false(self.p11 in self.diag)
        
        assert_true(self.p11 in self.circle)
        


    def test___contains__(self):
        assert_true(self.p01 in self.p01)
        assert_false(Vector2(0,1) in self.p01) #Vector2 is not a Point2
        assert_false(self.p10 in self.p01)

class TestVector2:

    @classmethod
    def setup_class(self):
        self.v00=Vector2(0,0)
        self.v10=Vector2(1,0)
        self.v01=Vector2(0,1)
        self.v11=Vector2(1,1)

    def test___init__(self):
        #copy constructor
        v10=Vector2(self.v10)
        assert_equal(v10,self.v10)
        assert_false(v10 is self.v10)

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
        
        assert_true(self.v10 == (1,0))
        #more tests embedded below

    def test___ne__(self):
        assert_false(self.v11 != self.v11)
        assert_true(self.v11 != self.v01)

    def test___copy__(self):
        v10 = copy(self.v10)
        assert_true(v10==self.v10)
        assert_false(v10 is self.v10) #make sure copy is a deepcopy

    def test_mag2(self):
        assert_equal(self.v11.mag2(), 2)

    def test___abs__(self):
        assert_equal(abs(self.v11), sqrt(2))

    def test_normalized(self):
        assert_true(self.v10.normalized())
        assert_equal(abs(self.v11.normalized()),1)

    def test_normalize(self):
        v=copy(self.v11)
        v.normalize()
        assert_equal(abs(v),1)

    def test___add__(self):
        assert_equal(self.v10+self.v01, self.v11) # Vector + Vector -> Vector
        assert_equal(self.v10+Point2(0,1),Point2(1,1))# Vector + Point -> Point
        assert_equal(Point2(1,0)+Point2(0,1),self.v11)# Point + Point -> Vector

    def test___iadd__(self):
        v=copy(self.v10)
        v+=self.v01
        assert_equal(v, self.v11)

    def test___neg__(self):
        assert_equal(-self.v11,Vector2(-1,-1))

    def test___sub__(self):
        assert_equal(self.v11-self.v10,self.v01) # Vector - Vector -> Vector
        assert_equal(self.v11-Point2(1,0),Point2(0,1)) # Vector - Point -> Point
        assert_equal(Point2(1,1)-Point2(1,0),self.v01) # Point - Point -> Vector

    def test___rsub__(self):
        assert_equal((1,1)-self.v10,Point2(0,1)) # Point - Vector -> Point

    def test___mul__(self):
        assert_equal(2*self.v11,Vector2(2,2))
        assert_equal(self.v11*2,Vector2(2,2))

    def test___imul__(self):
        v=copy(self.v10)
        v*=2
        assert_equal(v,Vector2(2,0))

    def test___div__(self):
        assert_equal(self.v11/2.,Vector2(.5,.5))

    def test___floordiv__(self):
        assert_equal(self.v11//2.,Vector2(0,0))

    def test___rdiv__(self):
        assert_equal(2./self.v11,Vector2(2,2))

    def test___rfloordiv__(self):
        assert_equal(2.//self.v11,Vector2(2,2))

    def test___truediv__(self):
        assert_equal(operator.truediv(self.v11,3),Vector2(1/3.,1/3.))

    def test___rtruediv__(self):
        assert_equal(operator.truediv(3,self.v11),Vector2(3.,3.))

    def test_dot(self):
        assert_equal(self.v10.dot(self.v01),0)
        assert_equal(self.v11.dot(self.v01),1)

    def test_angle(self):
        assert_equal(self.v10.angle(self.v01),pi/2.)
        assert_equal(self.v11.angle(self.v01),pi/4.)

    def test_cross(self):
        assert_equal(self.v10.cross(),-self.v01)

    def test_project(self):
        assert_equal(self.v10.project(self.v11),Vector2(.5,.5))

    def test_reflect(self):
        assert_equal(self.v11.reflect(self.v10),Vector2(-1,1))

    def test___pos__(self):
        v10 = +self.v10 #copy
        assert_true(v10==self.v10)
        assert_false(v10 is self.v10)

    def test___hash__(self):
        assert Vector2(1,0) in (self.v00, self.v01, self.v10, self.v11)


class TestMatrix3:
    @classmethod
    def setup_class(self):
        self.id3=Matrix3() #identity
        self.mat123=Matrix3(1,2,3,4,5,6,7,8,9) # singular
        self.mat456=Matrix3(4,5,6,9,8,7,3,1,2)

    def test___init__(self):
        #default constructor makes an identity matrix
        assert_equal(self.id3, Matrix3(1,0,0, 0,1,0, 0,0,1))
        #copy constructor
        mat123=Matrix3(self.mat123)
        assert_equal(mat123,self.mat123)
        assert_false(mat123 is self.mat123)

    def test___copy__(self):
        mat123=copy(self.mat123)
        assert_equal(mat123,self.mat123)
        assert_false(mat123 is self.mat123)

    def test___repr__(self):
        assert_equal(repr(self.mat123), 'Matrix3(1, 4, 7, 2, 5, 8, 3, 6, 9)')

    def test_new_identity(self):
        mat=Matrix3.new_identity()
        assert_equal(mat,Matrix3())
        return mat

    def test_new_scale(self):
        mat=Matrix3.new_scale(2,3)
        assert_equal(mat,Matrix3(2,0,0, 0,3,0, 0,0,1))
        return mat

    def test_new_rotate(self):
        mat=Matrix3.new_rotate(radians(60))
        s32=sqrt(3)/2
        res=Matrix3(0.5,+s32,0, -s32,0.5,0, 0,0,1) #warning : .new takes columnwise elements
        assert_equal(mat, res)
        return mat

    def test_new_translate(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.new_translate(x, y))
        raise SkipTest


    def test___call__(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.__call__(other))
        raise SkipTest

    def test___getitem__(self):
        assert_equal(self.mat456.f,8) #central element
        assert_equal(self.mat456[1,1],8)
        assert_equal(self.mat456[4],8)

    def test___imul__(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.__imul__(other))
        raise SkipTest

    def test___mul__(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.__mul__(other))
        raise SkipTest

    def test___setitem__(self):
        pass # used everywhere

    def test_angle(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.angle(angle))
        raise SkipTest

    def test_determinant(self):
        d=self.mat123.determinant()
        assert_equal(d,0)
        d=self.mat456.determinant()
        assert_equal(d,-39)

    def test_identity(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.identity())
        raise SkipTest

    def test_inverse(self):
        inv=self.mat123.inverse()
        assert_equal(inv,self.id3)
        inv=self.mat456.inverse()
        assert_equal(inv[0,0],-0.23076923076923075)

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

    def test_transpose(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.transpose())
        raise SkipTest

    def test_transposed(self):
        # matrix3 = Matrix3()
        # assert_equal(expected, matrix3.transposed())
        raise SkipTest

    def test___iter__(self):
        # matrix3 = Matrix3(*args)
        # assert_equal(expected, matrix3.__iter__())
        raise SkipTest 

    def test_orientation(self):
        # matrix3 = Matrix3(*args)
        # assert_equal(expected, matrix3.orientation())
        raise SkipTest # 



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
         #copy constructor
        l1=Line2(self.l1)
        assert_equal(l1,self.l1)
        assert_false(l1 is self.l1)
        assert_false(l1.p is self.l1.p)
        assert_false(l1.v is self.l1.v)

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

        #same tests as in Notebook
        a=Arc2((0,0),(0,1),(1,0))
        l1=Line2((-2,.5),Vector2(4,0)) #horizontal at y=0.5
        l2=Line2((-2,-.5),Vector2(4,0)) #horizontal at y=-0.5

        #l1 has a single intersection with a
        p1=Point2(-sqrt(3)/2,.5)
        inter=l1.intersect(a)
        assert_equal(inter,p1)
        inter=a.intersect(l1) #same intersection after entity swapping
        assert_equal(inter,p1)

         #l2 has a two intersection with a
        p1=Point2(sqrt(3)/2,-.5)
        p2=Point2(-sqrt(3)/2,-.5)
        inter=l2.intersect(a)
        assert_equal(inter,[p1,p2])
        inter=a.intersect(l2)
        assert_equal(inter,[p1,p2])


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
        self.s1=Segment2((1,1),Vector2(1,1)) # (1,1) -> (2,2)
        self.s2=Segment2((2,2),Point2(-1,-1)) # (2,2) -> (-1,-1) is parallel to s1
        self.s3=Segment2((-1,-1),(-1,1),1) #perpendicular to l1 and l2, normalized

    def test___repr__(self):
        assert_equal(repr(self.s1),"Segment2(Point2(1, 1),Point2(2, 2))")
        assert_equal(repr(self.s2),"Segment2(Point2(2, 2),Point2(-1, -1))")
        assert_equal(repr(self.s3),"Segment2(Point2(-1, -1),Point2(-1.0, 0.0))")

    def test___abs__(self):
        assert_equal(abs(self.s1),sqrt(2))
        
    def test_mag2(self):
        assert_equal(self.s1.mag2(),2)

    def test_intersect(self):
        """TODO: implement intersect of colinear segments
        inter=self.s1.intersect(self.s2)
        assert_equal(self.s1.distance(inter),0)
        assert_equal(self.s2.distance(inter),0)
        """
        inter=self.s1.intersect(self.s3)
        assert_equal(inter,None)
        inter=self.s2.intersect(self.s3)
        assert_equal(inter,Point2(-1,-1))

    def test_distance(self):
        assert_equal(self.s1.distance(self.s2),0)
        assert_equal(self.s2.distance(self.s3),0)
        assert_equal(self.s1.distance(self.s3),2*sqrt(2))


    def test_point(self):
        assert_equal(self.s1.point(0),self.s1.p1)
        assert_equal(self.s1.point(1),self.s1.p2)

    def test_tangent(self):
        assert_equal(self.s1.tangent(0),self.s1.v)
        assert_equal(self.s1.tangent(1),self.s1.v)
        assert_equal(self.s1.tangent(1.001),None)

    def test_swap(self):
        # segment2 = Segment2()
        # assert_equal(expected, segment2.swap())
        raise SkipTest

    def test_bisect(self):
        l=self.s1.bisect()
        assert_equal(l.intersect(self.s1),self.s1.midpoint())
        assert_equal(l.v.dot(self.s1.v),0)

    def test_midpoint(self):
        pass #tested above

class TestCircle:
    @classmethod
    def setup_class(self):
        self.c1=Circle(Point2(0,0),1)
        self.c2=Circle((1,0),1) # intersects c1
        self.c3=Circle((3,0),1) # tangent to c2, disjoint to c1

    def test___init__(self):
        pass #tested above

    def test_point(self):
        assert_equal(self.c1.point(0),(1,0))

    def test_tangent(self):
        assert_equal(self.c1.tangent(0),(0,1))

    def test___copy__(self):
        raise SkipTest

    def test___repr__(self):
        # circle = Circle(center, radius)
        # assert_equal(expected, circle.__repr__())
        raise SkipTest
    
    def test_intersect(self):
        res=self.c1.intersect(self.c2)
        assert_true(isinstance(res,list))
        assert_equal(len(res),2)
        assert_true(isinstance(res[0],Point2))
        assert_true(isinstance(res[0],Point2))
        
        res=self.c1.intersect(self.c3)
        assert_equal(res,None)
        
        res=self.c2.intersect(self.c3)
        assert_true(isinstance(res,Point2))
        
        assert_equal(self.c1.intersect(self.c1),self.c1) #
        
        large=Circle(self.c1.c,1.1*self.c1.r)
        assert_equal(large.intersect(self.c1),self.c1)

    def test_connect(self):
        # circle = Circle(center, radius)
        # assert_equal(expected, circle.connect(other))
        raise SkipTest

    

    def test___abs__(self):
        # circle = Circle(center, radius)
        # assert_equal(expected, circle.__abs__())
        raise SkipTest

    def test___eq__(self):
        # circle = Circle(center, radius)
        # assert_equal(expected, circle.__eq__(other))
        raise SkipTest

    def test___contains__(self):
        # circle = Circle(*args)
        # assert_equal(expected, circle.__contains__(pt))
        raise SkipTest 

    def test_swap(self):
        # circle = Circle(*args)
        # assert_equal(expected, circle.swap())
        raise SkipTest 

class TestArc2:
    @classmethod
    def setup_class(self):
        self.a1=Arc2((0,0),(1,0),(0,1))
        self.a2=Arc2((0,0),0,pi/2.,1) # should be the same as a1
        self.a3=Arc2((0,0),pi/2.,0,1,dir=-1) #same, inverted
        self.ap=Arc2(center=Point2(454.80692478710336, 69.74749779176005),p1=Point2(74.67492478710335, 86.62949779176006),p2=Point2(74.48092478710339, 58.021497791760055),r=380.506687652)
        self.ap2=Arc2(center=Point2(-454.80607521289664, -9.203502208239968),p1=Point2(-74.67307521289663, -26.08650220823995),p2=Point2(-74.47907521289662, 2.521497791760055),r=380.507731036)
        self.c=Arc2((1,1),r=1) #unit circle around (1,1)
        self.cw=Arc2((1,1),r=1,dir=-1) #unit circle around (1,1) clockwise

    def test___init__(self):
        #copy constructor
        a1=Arc2(self.a1)
        assert_equal(a1,self.a1)
        assert_false(a1 is self.a1)
        assert_false(a1.c is self.a1.c)
        assert_false(a1.p is self.a1.p)
        assert_false(a1.p2 is self.a1.p2)

    def test___eq__(self):
        assert_true(self.a1==self.a2)
        assert_true(self.a1==self.a3) # inverted parametrization, but same geometry
        assert_false(self.a1==self.ap)

    def test_angle(self):
        assert_equal(self.a1.angle(),self.a2.angle())
        assert_equal(self.a1.angle(),-self.a3.angle())
        assert_true(self.ap.angle()<radians(10)) # check the angle is not the complementary angle
        assert_true(self.ap2.angle()<radians(10)) # check the angle is not the complementary angle
        assert_equal(self.c.angle(),2*pi)
        assert_equal(self.cw.angle(),-2*pi)

    def test___abs__(self):
        assert_equal(abs(self.a1),pi/2.)
        assert_true(self.ap.length<50) # check the angle is not the complementary angle
        assert_true(self.ap2.length<50) # check the angle is not the complementary angle


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

    def test___contains__(self):
        # arc2 = Arc2(center, p1, p2, r, dir)
        # assert_equal(expected, arc2.__contains__(pt))
        raise SkipTest 

    def test_intersect(self):
        # arc2 = Arc2(center, p1, p2, r, dir)
        # assert_equal(expected, arc2.intersect(other))
        raise SkipTest 



class TestArgPair:
    def test_arg_pair(self):
        assert_equal(argPair(1,2),(1,2))
        assert_equal(argPair((1,2)),(1,2))
        assert_equal(argPair([1,2]),(1,2))

        #assert_equal(argPair(1),(1,1)) # not allowed anymore

class TestCopy:
    def test_copy(self):
        # assert_equal(expected, copy(object))
        raise SkipTest 

class TestCircleFrom3Points:
    def test_circle_from_3_points(self):
        p1=Point2(-1,0) 
        p2=Point2(0,1) 
        p3=Point2(1,0) 
        c=circle_from_3_points(p1,p2,p3)
        assert_equal(c.c.xy,(0,0))
        assert_equal(c.r,1)

class TestArcFrom3Points:
    def test_arc_from_3_points(self):
        p1=Point2(-1,0) 
        p2=Point2(0,1) 
        p3=Point2(1,0) 
        a=arc_from_3_points(p1,p2,p3)
        assert_equal(a.c.xy,(0,0))
        assert_equal(a.r,1)
        assert_equal(a.point(0.5),p2)
        
        p1=Point2(-1,0) 
        p2=Point2(0,-1) 
        p3=Point2(1,0) 
        a=arc_from_3_points(p1,p2,p3)
        assert_equal(a.c.xy,(0,0))
        assert_equal(a.r,1)
        assert_equal(a.point(0.5),p2)
        
        p1=Point2(1,0) 
        p2=Point2(0,1) 
        p3=Point2(-1,0) 
        a=arc_from_3_points(p1,p2,p3)
        assert_equal(a.c.xy,(0,0))
        assert_equal(a.r,1)
        assert_equal(a.point(0.5),p2)
        
        p1=Point2(1,0) 
        p2=Point2(0,-1) 
        p3=Point2(-1,0) 
        a=arc_from_3_points(p1,p2,p3)
        assert_equal(a.c.xy,(0,0))
        assert_equal(a.r,1)
        assert_equal(a.point(0.5),p2)

class TestEllipse:
    @classmethod
    def setup_class(self):
        self.e1=Ellipse((1,2),(3,4))
        self.e2=Ellipse((1,2),2,2,pi/2)     
        self.e3=Ellipse(self.e2)        
        assert_equal(self.e1,self.e2)
        
    def test___init__(self):
        pass

    def test___repr__(self):
        s=str(self.e1)
        assert_equal(s,'Ellipse(Point2(1, 2),2,2)')

    def test___eq__(self):
        # ellipse = Ellipse(*args)
        # assert_equal(expected, ellipse.__eq__(other))
        raise SkipTest # TODO: implement your test here

if __name__ == "__main__":
    runmodule()

