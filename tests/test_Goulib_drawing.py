from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *

import os, sys
from Goulib.geom import *
from Goulib.drawing import *

path=os.path.dirname(os.path.abspath(__file__))

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

    def test___add__(self):
        # b_box = BBox(pt1, pt2)
        # assert_equal(expected, b_box.__add__(other))
        raise SkipTest

    def test___call__(self):
        # b_box = BBox(pt1, pt2)
        # assert_equal(expected, b_box.__call__())
        raise SkipTest

    def test___iadd__(self):
        # b_box = BBox(pt1, pt2)
        # assert_equal(expected, b_box.__iadd__(pt))
        raise SkipTest

    def test___init__(self):
        # b_box = BBox(pt1, pt2)
        raise SkipTest

    def test___repr__(self):
        # b_box = BBox(pt1, pt2)
        # assert_equal(expected, b_box.__repr__())
        raise SkipTest

    def test_center(self):
        # b_box = BBox(pt1, pt2)
        # assert_equal(expected, b_box.center())
        raise SkipTest

    def test_size(self):
        # b_box = BBox(pt1, pt2)
        # assert_equal(expected, b_box.size())
        raise SkipTest

    def test_trans(self):
        # b_box = BBox(pt1, pt2)
        # assert_equal(expected, b_box.trans(trans))
        raise SkipTest

class TestRint:
    def test_rint(self):
        assert_equal(rint(0.4),0)
        assert_equal(rint(0.9),1)
        assert_equal(rint(-0.4),0)
        assert_equal(rint(-0.9),-1)

class TestRpoint:
    def test_rpoint(self):
        # assert_equal(expected, rpoint(pt, decimals))
        raise SkipTest

class TestEntity:
    @classmethod
    def setup_class(self):
        self.seg=Segment2((1,0),(2,3))
        self.arc=Arc2((1,1),(0,0),radians(120))
        self.circle=Circle((1,1),2)

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
        # entity = Entity()
        # assert_equal(expected, entity.from_pdf())
        raise SkipTest

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
        raise SkipTest # TODO: implement your test here

    def test_figure(self):
        # entity = Entity()
        # assert_equal(expected, entity.figure(**kwargs))
        raise SkipTest # TODO: implement your test here

    def test_render(self):
        # entity = Entity()
        # assert_equal(expected, entity.render(format, **kwargs))
        raise SkipTest # TODO: implement your test here

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

        self.dxf= Drawing(path+'/Homer_Simpson_by_CyberDrone.dxf')
        self.blocks= self.dxf.block
        assert_true('hand 1' in self.blocks)

    def test_append(self):
        # group = Group()
        # assert_equal(expected, group.append(entity, **kwargs))
        raise SkipTest # TODO: implement your test here

    def test_extend(self):
        # group = Group()
        # assert_equal(expected, group.extend(entities, **kwargs))
        raise SkipTest # TODO: implement your test here

    def test_from_dxf(self):
        # group = Group()
        # assert_equal(expected, group.from_dxf(dxf, layers, only, ignore, trans, flatten))
        raise SkipTest # TODO: implement your test here

    def test_swap(self):
        # group = Group()
        # assert_equal(expected, group.swap())
        raise SkipTest # TODO: implement your test here

    def test___copy__(self):
        # group = Group()
        # assert_equal(expected, group.__copy__())
        raise SkipTest # TODO: implement your test here

    def test_color(self):
        # group = Group()
        # assert_equal(expected, group.color())
        raise SkipTest # TODO: implement your test here

    def test_color_case_2(self):
        # group = Group()
        # assert_equal(expected, group.color(c))
        raise SkipTest # TODO: implement your test here

class TestChain:
    def test___repr__(self):
        # chain = Chain()
        # assert_equal(expected, chain.__repr__())
        raise SkipTest

    def test_append(self):
        # chain = Chain()
        # assert_equal(expected, chain.append(edge, tol))
        raise SkipTest

    def test_bbox(self):
        # chain = Chain()
        # assert_equal(expected, chain.bbox())
        raise SkipTest

    def test_end(self):
        # chain = Chain()
        # assert_equal(expected, chain.end())
        raise SkipTest

    def test_from_dxf(self):
        # chain = Chain()
        # assert_equal(expected, chain.from_dxf(trans))
        raise SkipTest

    def test_start(self):
        # chain = Chain()
        # assert_equal(expected, chain.start())
        raise SkipTest

    def test_to_dxf(self):
        # chain = Chain()
        # assert_equal(expected, chain.to_dxf(split, **attr))
        raise SkipTest

    def test___init__(self):
        # chain = Chain(data)
        raise SkipTest

    def test_from_svg(self):
        # chain = Chain(data)
        # assert_equal(expected, chain.from_svg())
        raise SkipTest

    def test_from_pdf(self):
        # chain = Chain(data)
        # assert_equal(expected, chain.from_pdf(color))
        raise SkipTest

class TestDrawing:
    @classmethod
    def setup_class(self):
        self.dxf= Drawing(path+'/drawing.dxf')
        self.svg= Drawing(path+'/drawing.svg')
        self.pdf= Drawing(path+'/drawing.pdf')

        seg=Segment2((1,0),(2,3))
        arc=Arc2((1,1),(0,0),radians(120))
        circle=Circle((1,1),2)
        self.simple=Drawing(data=[seg,arc,circle])

    def test_load(self):
        return
        cube=Drawing(path+'/cubeecraft_template.pdf')
        cube.save(path+'/cubeecraft.dxf')

    def test_save(self):
        for ext in ['png','svg','pdf','dxf']:
            self.svg.save(path+'/drawing.svg.%s'%ext)
            self.dxf.save(path+'/drawing.dxf.%s'%ext)
            self.pdf.save(path+'/drawing.pdf.%s'%ext)


    def test___init__(self):
        pass # tested above

    def test_bbox(self):
        # drawing = Drawing(filename, options, **kwargs)
        # assert_equal(expected, drawing.bbox(layers, ignore))
        raise SkipTest

    def test_read_dxf(self):
        # drawing = Drawing(filename, options, **kwargs)
        # assert_equal(expected, drawing.read_dxf(filename, options, **kwargs))
        raise SkipTest

    def test_read_svg(self):
        # drawing = Drawing(filename, options, **kwargs)
        # assert_equal(expected, drawing.read_svg(filename, options, **kwargs))
        raise SkipTest

    def test_read_pdf(self):
        # drawing = Drawing(filename, options, **kwargs)
        # assert_equal(expected, drawing.read_pdf(filename, **kwargs))
        raise SkipTest

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

    def test_img(self):
        # drawing = Drawing(filename, **kwargs)
        # assert_equal(expected, drawing.img(size, border, box, layers, ignore, forcelayercolor, antialias, background))
        raise SkipTest

class TestImg2base64:
    def test_img2base64(self):
        # assert_equal(expected, img2base64(img, fmt))
        raise SkipTest

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
        self.r1=r1=Rect((0,0),(-1,1))
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
        raise SkipTest # TODO: implement your test here

class TestText:
    def test___init__(self):
        # text = Text(text, point, size, rotation)
        raise SkipTest # TODO: implement your test here

    def test_bbox(self):
        # text = Text(text, point, size, rotation)
        # assert_equal(expected, text.bbox())
        raise SkipTest # TODO: implement your test here

    def test_patches(self):
        # text = Text(text, point, size, rotation)
        # assert_equal(expected, text.patches(**kwargs))
        raise SkipTest # TODO: implement your test here

    def test_to_dxf(self):
        # text = Text(text, point, size, rotation)
        # assert_equal(expected, text.to_dxf(**attr))
        raise SkipTest # TODO: implement your test here

    def test_intersect(self):
        # text = Text(text, point, size, rotation)
        # assert_equal(expected, text.intersect(other))
        raise SkipTest # TODO: implement your test here

    def test_length(self):
        # text = Text(text, point, size, rotation)
        # assert_equal(expected, text.length())
        raise SkipTest # TODO: implement your test here

class TestInstance:
    def test___init__(self):
        # instance = Instance(group, trans, name)
        raise SkipTest # TODO: implement your test here

    def test___iter__(self):
        # instance = Instance(group, trans, name)
        # assert_equal(expected, instance.__iter__())
        raise SkipTest # TODO: implement your test here

    def test___repr__(self):
        # instance = Instance(group, trans, name)
        # assert_equal(expected, instance.__repr__())
        raise SkipTest # TODO: implement your test here

    def test_from_dxf(self):
        # instance = Instance()
        # assert_equal(expected, instance.from_dxf(blocks, mat3))
        raise SkipTest # TODO: implement your test here

class test__Group:
    def test_bbox(self):
        # __group = _Group()
        # assert_equal(expected, __group.bbox())
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
    import logging
    runmodule(level=logging.WARNING)
