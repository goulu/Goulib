#!/usr/bin/env python
# coding: utf8
from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *


from Goulib.geom import *
from Goulib.drawing import *

import os
path=os.path.dirname(os.path.abspath(__file__))
results=path+'\\results\\drawing\\' #path for results

class TestTrans:
    def test_trans(self):
        assert_equal(Trans(), Matrix3(1,0,0, 0,1,0, 0,0,1))
        assert_equal(Trans(scale=2), Matrix3(2,0,0, 0,2,0, 0,0,1))
        assert_equal(Trans(offset=(2,3)), Matrix3(1,0,0, 0,1,0, 2,3,1))
        s32=sqrt(3)/2
        res=Matrix3(0.5,+s32,0, -s32,0.5,0, 0,0,1) #warning : .new takes columnwise elements
        assert_almost_equal(Trans(rotation=60),res)

class TestBBox:
    @classmethod
    def setup_class(self):
        self.empty=BBox()
        self.unit=BBox((0,1),(1,0))
        self.box=BBox((-1,-2),(3,4))
        
    def test___init__(self):
        pass
    
    def test___repr__(self):
        assert_equal(repr(self.box),"[[-1,3), [-2,4)]")
                     
    def test___call__(self):
        assert_equal(self.box(),[-1, -2, 3, 4])

    def test___iadd__(self):
        self.box+=None # must pass
        
    def test___add__(self):
        b1=self.unit+self.box
        b2=self.box+self.unit
        assert_equal(b1,b2)
        raise SkipTest

    def test_xmin(self):
        assert_equal(self.unit.xmin, 0)
        assert_equal(self.box.xmin, -1)

    def test_xmax(self):
        assert_equal(self.unit.xmax, 1)
        assert_equal(self.box.xmax, 3)

    def test_ymin(self):
        assert_equal(self.unit.ymin, 0)
        assert_equal(self.box.ymin, -2)

    def test_ymax(self):
        assert_equal(self.unit.ymax, 1)
        assert_equal(self.box.ymax, 4)

    def test_xmed(self):
        assert_equal(self.unit.xmed, 0.5)
        assert_equal(self.box.xmed, 1)

    def test_ymed(self):
        assert_equal(self.unit.ymed, 0.5)
        assert_equal(self.box.ymed, 1)
        
    def test_width(self):
        assert_equal(self.unit.width, 1)
        assert_equal(self.box.width, 4)

    def test_height(self):
        assert_equal(self.unit.height, 1)
        assert_equal(self.box.height, 6)
    
    def test_center(self):
        assert_equal(self.box.center(),Point2(1,1))

    def test_size(self):
        assert_equal(self.box.size(),(4,6))

    def test___contains__(self):
        
        p1=(.5,.5)
        p2=Point2(.99,1.01)
        
        assert_true(p1 in self.unit)
        assert_false(p2 in self.unit)
        assert_false(Segment2(p1,p2) in self.unit)
        assert_false([p1,p2] in self.unit)
        
        assert_false(Circle(p1,0.5) in self.unit) #tangent to box is not in box ...
        assert_true(Circle(p1,0.4999999) in self.unit) 
        
        assert_true(self.unit in self.box)
        
    def test_trans(self):
        t=Trans(offset=(-1,-2),scale=(4,6), rotation=0)
        assert_equal(self.unit.trans(t),self.box)

    def test_area(self):
        # b_box = BBox(p1, p2)
        # assert_equal(expected, b_box.area())
        raise SkipTest # TODO: implement your test here

class TestEntity:
    @classmethod
    def setup_class(self):
        self.seg=Segment2((1,0),(2,3))
        self.arc=Arc2((1,1),(0,0),radians(120))
        self.circle=Circle((1,1),2)
        self.point=Point2(0,0)

    def test___init__(self):
        pass

    def test___repr__(self):
        # entity = Entity()
        # assert_equal(expected, entity.__repr__())
        raise SkipTest

    def test_bbox(self):
        # entity = Entity()
        # assert_equal(expected, entity.bbox())
        raise SkipTest

    def test_end(self):
        # entity = Entity()
        # assert_equal(expected, entity.end())
        raise SkipTest

    def test_from_dxf(self):
        # entity = Entity()
        # assert_equal(expected, entity.from_dxf(trans))
        raise SkipTest

    def test_isclosed(self):
        # entity = Entity()
        # assert_equal(expected, entity.isclosed())
        raise SkipTest

    def test_ishorizontal(self):
        # entity = Entity()
        # assert_equal(expected, entity.ishorizontal(tol))
        raise SkipTest

    def test_isline(self):
        # entity = Entity()
        # assert_equal(expected, entity.isline())
        raise SkipTest

    def test_isvertical(self):
        # entity = Entity()
        # assert_equal(expected, entity.isvertical(tol))
        raise SkipTest

    def test_start(self):
        # entity = Entity()
        # assert_equal(expected, entity.start())
        raise SkipTest

    def test_to_dxf(self):
        # entity = Entity()
        # assert_equal(expected, entity.to_dxf(**attr))
        raise SkipTest

    def test_center(self):
        # entity = Entity()
        # assert_equal(expected, entity.center())
        raise SkipTest

    def test_from_svg(self):
        # entity = Entity()
        # assert_equal(expected, entity.from_svg())
        raise SkipTest

    def test_artist(self):
        # entity = Entity()
        # assert_equal(expected, entity.artist(ax, **kwargs))
        raise SkipTest

    def test_from_pdf(self):
        pass #tested elsewhere

    def test_svg_path(self):
        # entity = Entity()
        # assert_equal(expected, entity.svg_path(currentpos))
        raise SkipTest

    def test_patches(self):
        # entity = Entity()
        # assert_equal(expected, entity.patches(**kwargs))
        raise SkipTest

    def test___copy__(self):
        s2=copy(self.seg)
        assert_false(self.seg.p is s2.p) # Segment2.copy is in fact a deepcopy
        s2.layer="something"
        s3=copy(s2)
        assert_equal(s3.layer,"something")

    def test_draw(self):
        # entity = Entity()
        # assert_equal(expected, entity.draw(fig, **kwargs))
        raise SkipTest 

    def test_figure(self):
        # entity = Entity()
        # assert_equal(expected, entity.figure(**kwargs))
        raise SkipTest 

    def test_render(self):
        a=Point2(0,1)
        b=Point2(1,1)
        c=Point2(1,0)
        g=Group([a,b,c])
        g.render('svg')

    def test_setattr(self):
        # entity = Entity()
        # assert_equal(expected, entity.setattr(**kwargs))
        raise SkipTest 

class TestGroup:
    @classmethod
    def setup_class(self):
        #example from notebook
        a=Arc2((0,0),(0,1),(1,0))
        l1=Segment2((-2,.5),Vector2(4,0)) #horizontal at y=0.5
        l2=Segment2((-2,-.5),Vector2(4,0)) #horizontal at y=-0.5
        lines=Group([l1,l2])
        lines.color='blue'
        pts=Group([i[0] for i in lines.intersect(a)]) # list of intersection points
        self.group=Group([lines,a,pts])
        self.group.render('svg')
        
        #second example from notebook
        r1=Rect((0,0),(-1,1))
        r2=Rect((1,-1),(2,2))
        c1=Circle(Point2(4,4),1)
        c2=Circle(Point2(0,2),.5)
        s1=r1.connect(r2)
        s1.color='red'
        s2=r2.connect(c1)
        s2.color='red'
        s3=c1.connect(c2)
        s3.color='red'

        self.group=Group([r1,r2,c1,c2,s1,s2,s3])


        
    def test_distance(self):
        g2=Trans(scale=2, offset=(10,1), rotation=30)*self.group
        g2.color='blue'
        Drawing([self.group,g2]).save(results+'drawing.Group.distance.png')
        assert_equal(self.group.distance(g2),2.026833782163534)

    def test_append(self):
        # group = Group()
        # assert_equal(expected, group.append(entity, **kwargs))
        raise SkipTest 

    def test_extend(self):
        # group = Group()
        # assert_equal(expected, group.extend(entities, **kwargs))
        raise SkipTest 

    def test_from_dxf(self):
        try:
            import dxfgrabber
        except:
            raise SkipTest # optional
        
        dxf= Drawing(path+'/data/Homer_Simpson_by_CyberDrone.dxf')
        self.blocks= dxf.block
        assert_true('hand 1' in self.blocks)

    def test_swap(self):
        # group = Group()
        # assert_equal(expected, group.swap())
        raise SkipTest 
        
    def test_length(self):
        assert_equal(self.group.length,27.22534104051515)

    def test___copy__(self):
        # group = Group()
        # assert_equal(expected, group.__copy__())
        raise SkipTest 

    def test_color(self):
        # group = Group()
        # assert_equal(expected, group.color())
        raise SkipTest 

    def test_color_case_2(self):
        # group = Group()
        # assert_equal(expected, group.color(c))
        raise SkipTest 

    def test_chainify(self):
        # group = Group()
        # assert_equal(expected, group.chainify(mergeable))
        raise SkipTest 

    def test_layer(self):
        # group = Group()
        # assert_equal(expected, group.layer())
        raise SkipTest 

    def test_layer_case_2(self):
        # group = Group()
        # assert_equal(expected, group.layer(l))
        raise SkipTest 

class TestChain:
    @classmethod
    def setup_class(self):
        group=Group()
        group.append(Segment2((0,0),(1,1)))
        group.append(Segment2((2,0),Point2(1,1)))
        self.chain=Chain(group)
        
    def test___init__(self):
        pass #tested above
    
    def test_append(self):
        chain=Chain(self.chain) #copy
        assert_false(chain.isclosed())
        chain.append(Segment2((0,0),(2,0)))
        assert_true(chain.isclosed())
    
    def test___repr__(self):
        assert_equal(repr(self.chain),'Chain(Point2(0, 0),Point2(2, 0),2)')

    def test_bbox(self):
        # chain = Chain()
        # assert_equal(expected, chain.bbox())
        raise SkipTest
    
    def test_start(self):
        assert_equal(self.chain.start,Point2(0,0))

    def test_end(self):
        assert_equal(self.chain.end,Point2(2,0))

    def test_from_dxf(self):
        # chain = Chain()
        # assert_equal(expected, chain.from_dxf(trans))
        raise SkipTest

    def test_to_dxf(self):
        # chain = Chain()
        # assert_equal(expected, chain.to_dxf(split, **attr))
        raise SkipTest

    def test_from_svg(self):
        # chain = Chain(data)
        # assert_equal(expected, chain.from_svg())
        raise SkipTest

    def test_from_pdf(self):
        # chain = Chain(data)
        # assert_equal(expected, chain.from_pdf(color))
        raise SkipTest

    def test_contiguous(self):
        # chain = Chain(data)
        # assert_equal(expected, chain.contiguous(edge, tol, allow_swap))
        raise SkipTest 

class TestDrawing:
    @classmethod
    def setup_class(self):
        self.dxf= Drawing(path+'/data/drawing.dxf')
        self.svg= Drawing(path+'/data/drawing.svg')
        self.pdf= Drawing(path+'/data/drawing.pdf')

        seg=Segment2((1,0),(2,3))
        arc=Arc2((1,1),(0,0),radians(120))
        circle=Circle((1,1),2)
        self.simple=Drawing(data=[seg,arc,circle])

    def test_load(self):
        return
        cube=Drawing(path+'/data/cubeecraft_template.pdf')
        cube.save(results+'cubeecraft.dxf')

    def test_save(self):
        for ext in ['png','svg','pdf','dxf']:
            if self.dxf:
                self.dxf.save(results+'drawing.dxf.%s'%ext)
            self.svg.save(results+'drawing.svg.%s'%ext)
            self.pdf.save(results+'drawing.pdf.%s'%ext)

    def test___init__(self):
        pass # tested above

    def test_bbox(self):
        # drawing = Drawing(filename, options, **kwargs)
        # assert_equal(expected, drawing.bbox(layers, ignore))
        raise SkipTest

    def test_read_dxf(self):
        pass # tested above
    
    def test_read_svg(self):
        pass # tested above

    def test_read_pdf(self):
        pantone=Drawing(path+'/data/Pantone Fan.pdf')
        pantone.save(results+'Pantone_Fan_out.pdf')

    def test_draw(self):
        # drawing = Drawing(filename, **kwargs)
        # assert_equal(expected, drawing.draw(fig, **kwargs))
        raise SkipTest

    def test_figure(self):
        # drawing = Drawing(filename, **kwargs)
        # assert_equal(expected, drawing.figure(**kwargs))
        raise SkipTest

    def test_render(self):
        assert_true(b'!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"' in self.simple.render('svg'))

class TestSpline:
    def test___init__(self):
        # spline = Spline(points)
        raise SkipTest

    def test_bbox(self):
        # spline = Spline(points)
        # assert_equal(expected, spline.bbox())
        raise SkipTest

    def test_end(self):
        # spline = Spline(points)
        # assert_equal(expected, spline.end())
        raise SkipTest

    def test_start(self):
        # spline = Spline(points)
        # assert_equal(expected, spline.start())
        raise SkipTest

    def test_swap(self):
        # spline = Spline(points)
        # assert_equal(expected, spline.swap())
        raise SkipTest

    def test_xy(self):
        # spline = Spline(points)
        # assert_equal(expected, spline.xy())
        raise SkipTest

    def test_length(self):
        # spline = Spline(points)
        # assert_equal(expected, spline.length())
        raise SkipTest

    def test___copy__(self):
        # spline = Spline(points)
        # assert_equal(expected, spline.__copy__())
        raise SkipTest

class TestRect:
    @classmethod
    def setup_class(self):
        # see IPython notebook for graphical examples
        self.r1=Rect((0,0),(-1,1))
        self.r2=Rect((1,-1),(2,2))

    def test___init__(self):
        pass #tested above

    def test_connect(self):
        s1=self.r1.connect(self.r2)
        assert_equal(s1,Segment2(Point2(0,0),Point2(1,0)))
        c1=Circle(Point2(4,1),1)
        s2=self.r2.connect(c1)
        assert_equal(s2,Segment2(Point2(2,1),Point2(3,1)))

    def test_distance(self):
        assert_equal(self.r1.distance(self.r2),1)

    def test___repr__(self):
        # rect = Rect(*args)
        # assert_equal(expected, rect.__repr__())
        raise SkipTest 

    def test_p1(self):
        # rect = Rect(*args)
        # assert_equal(expected, rect.p1())
        raise SkipTest 

    def test_p2(self):
        # rect = Rect(*args)
        # assert_equal(expected, rect.p2())
        raise SkipTest 

class TestText:
    def test___init__(self):
        # text = Text(text, point, size, rotation)
        raise SkipTest 

    def test_bbox(self):
        # text = Text(text, point, size, rotation)
        # assert_equal(expected, text.bbox())
        raise SkipTest 

    def test_patches(self):
        # text = Text(text, point, size, rotation)
        # assert_equal(expected, text.patches(**kwargs))
        raise SkipTest 

    def test_to_dxf(self):
        # text = Text(text, point, size, rotation)
        # assert_equal(expected, text.to_dxf(**attr))
        raise SkipTest 

    def test_intersect(self):
        # text = Text(text, point, size, rotation)
        # assert_equal(expected, text.intersect(other))
        raise SkipTest 

    def test_length(self):
        # text = Text(text, point, size, rotation)
        # assert_equal(expected, text.length())
        raise SkipTest 

class TestInstance:
    def test___init__(self):
        # instance = Instance(group, trans, name)
        raise SkipTest 

    def test___iter__(self):
        # instance = Instance(group, trans, name)
        # assert_equal(expected, instance.__iter__())
        raise SkipTest 

    def test___repr__(self):
        # instance = Instance(group, trans, name)
        # assert_equal(expected, instance.__repr__())
        raise SkipTest 

    def test_from_dxf(self):
        # instance = Instance()
        # assert_equal(expected, instance.from_dxf(blocks, mat3))
        raise SkipTest 

class TestCalcBulge:
    def test_calc_bulge(self):
        # assert_equal(expected, calcBulge(p1, bulge, p2))
        raise SkipTest 

class test__Group:
    def test_bbox(self):
        # __group = _Group()
        # assert_equal(expected, __group.bbox(filter))
        raise SkipTest 

    def test_connect(self):
        # __group = _Group()
        # assert_equal(expected, __group.connect(other))
        raise SkipTest 

    def test_intersect(self):
        # __group = _Group()
        # assert_equal(expected, __group.intersect(other))
        raise SkipTest 

    def test_length(self):
        # __group = _Group()
        # assert_equal(expected, __group.length())
        raise SkipTest 

    def test_patches(self):
        # __group = _Group()
        # assert_equal(expected, __group.patches(**kwargs))
        raise SkipTest 

    def test_to_dxf(self):
        # __group = _Group()
        # assert_equal(expected, __group.to_dxf(**kwargs))
        raise SkipTest 

class TestChains:
    def test_chains(self):
        # assert_equal(expected, chains(group, tol, mergeable))
        raise SkipTest 

class test__Group:
    def test_bbox(self):
        # __group = _Group()
        # assert_equal(expected, __group.bbox(filter))
        raise SkipTest # TODO: implement your test here

    def test_connect(self):
        # __group = _Group()
        # assert_equal(expected, __group.connect(other))
        raise SkipTest # TODO: implement your test here

    def test_intersect(self):
        # __group = _Group()
        # assert_equal(expected, __group.intersect(other))
        raise SkipTest # TODO: implement your test here

    def test_length(self):
        # __group = _Group()
        # assert_equal(expected, __group.length())
        raise SkipTest # TODO: implement your test here

    def test_patches(self):
        # __group = _Group()
        # assert_equal(expected, __group.patches(**kwargs))
        raise SkipTest # TODO: implement your test here

    def test_to_dxf(self):
        # __group = _Group()
        # assert_equal(expected, __group.to_dxf(**kwargs))
        raise SkipTest # TODO: implement your test here

if __name__=="__main__":
    runmodule(level=logging.WARNING)
