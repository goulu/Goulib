from nose.tools import assert_equal, assert_almost_equal
from nose import SkipTest

import os, math
from Goulib.geom import *
from Goulib.drawing import *

class TestTrans:
    def test_trans(self):
        assert_equal(Trans(), Matrix3.new(1,0,0, 0,1,0, 0,0,1))
        assert_equal(Trans(scale=2), Matrix3.new(2,0,0, 0,2,0, 0,0,1))
        assert_equal(Trans(offset=(2,3)), Matrix3.new(1,0,0, 0,1,0, 2,3,1))
        s32=sqrt(3)/2
        res=Matrix3.new(0.5,+s32,0, -s32,0.5,0, 0,0,1) #warning : .new takes columnwise elements
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
        raise SkipTest # TODO: implement your test here

    def test___call__(self):
        # b_box = BBox(pt1, pt2)
        # assert_equal(expected, b_box.__call__())
        raise SkipTest # TODO: implement your test here

    def test___iadd__(self):
        # b_box = BBox(pt1, pt2)
        # assert_equal(expected, b_box.__iadd__(pt))
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # b_box = BBox(pt1, pt2)
        raise SkipTest # TODO: implement your test here

    def test___repr__(self):
        # b_box = BBox(pt1, pt2)
        # assert_equal(expected, b_box.__repr__())
        raise SkipTest # TODO: implement your test here

    def test_center(self):
        # b_box = BBox(pt1, pt2)
        # assert_equal(expected, b_box.center())
        raise SkipTest # TODO: implement your test here

    def test_size(self):
        # b_box = BBox(pt1, pt2)
        # assert_equal(expected, b_box.size())
        raise SkipTest # TODO: implement your test here

    def test_trans(self):
        # b_box = BBox(pt1, pt2)
        # assert_equal(expected, b_box.trans(trans))
        raise SkipTest # TODO: implement your test here

class TestRint:
    def test_rint(self):
        assert_equal(rint(0.4),0)
        assert_equal(rint(0.9),1)
        assert_equal(rint(-0.4),0)
        assert_equal(rint(-0.9),-1)

class TestRpoint:
    def test_rpoint(self):
        # assert_equal(expected, rpoint(pt, decimals))
        raise SkipTest # TODO: implement your test here

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
        raise SkipTest # TODO: implement your test here

    def test_bbox(self):
        # entity = Entity()
        # assert_equal(expected, entity.bbox())
        raise SkipTest # TODO: implement your test here

    def test_end(self):
        # entity = Entity()
        # assert_equal(expected, entity.end())
        raise SkipTest # TODO: implement your test here

    def test_from_dxf(self):
        # entity = Entity()
        # assert_equal(expected, entity.from_dxf(trans))
        raise SkipTest # TODO: implement your test here

    def test_isclosed(self):
        # entity = Entity()
        # assert_equal(expected, entity.isclosed())
        raise SkipTest # TODO: implement your test here

    def test_ishorizontal(self):
        # entity = Entity()
        # assert_equal(expected, entity.ishorizontal(tol))
        raise SkipTest # TODO: implement your test here

    def test_isline(self):
        # entity = Entity()
        # assert_equal(expected, entity.isline())
        raise SkipTest # TODO: implement your test here

    def test_isvertical(self):
        # entity = Entity()
        # assert_equal(expected, entity.isvertical(tol))
        raise SkipTest # TODO: implement your test here

    def test_start(self):
        # entity = Entity()
        # assert_equal(expected, entity.start())
        raise SkipTest # TODO: implement your test here

    def test_to_dxf(self):
        # entity = Entity()
        # assert_equal(expected, entity.to_dxf(**attr))
        raise SkipTest # TODO: implement your test here

    def test_center(self):
        # entity = Entity()
        # assert_equal(expected, entity.center())
        raise SkipTest # TODO: implement your test here

    def test_from_svg(self):
        # entity = Entity()
        # assert_equal(expected, entity.from_svg())
        raise SkipTest # TODO: implement your test here

    def test_artist(self):
        # entity = Entity()
        # assert_equal(expected, entity.artist(ax, **kwargs))
        raise SkipTest # TODO: implement your test here

    def test_from_pdf(self):
        # entity = Entity()
        # assert_equal(expected, entity.from_pdf())
        raise SkipTest # TODO: implement your test here

    def test_svg_path(self):
        # entity = Entity()
        # assert_equal(expected, entity.svg_path(currentpos))
        raise SkipTest # TODO: implement your test here
    
class TestGroup:
    @classmethod
    def setup_class(self):
        seg=Segment2((1,0),(2,3))
        arc=Arc2((1,1),(0,0),radians(120))
        circle=Circle((1,1),2)
        self.group=Group([seg,arc,circle])
    
    def test___copy__(self):
        # group = Group()
        # assert_equal(expected, group.__copy__())
        raise SkipTest # TODO: implement your test here

    def test_bbox(self):
        # group = Group()
        # assert_equal(expected, group.bbox())
        raise SkipTest # TODO: implement your test here

    def test_length(self):
        # group = Group()
        # assert_equal(expected, group.length())
        raise SkipTest # TODO: implement your test here

    def test_artist(self):
        # group = Group()
        # assert_equal(expected, group.artist(ax, **kwargs))
        raise SkipTest # TODO: implement your test here

    def test_swap(self):
        # group = Group()
        # assert_equal(expected, group.swap())
        raise SkipTest # TODO: implement your test here

    def test_append(self):
        # group = Group()
        # assert_equal(expected, group.append(entity))
        raise SkipTest # TODO: implement your test here

    def test_to_dxf(self):
        # group = Group()
        # assert_equal(expected, group.to_dxf(**attr))
        raise SkipTest # TODO: implement your test here

class TestChain:
    def test___repr__(self):
        # chain = Chain()
        # assert_equal(expected, chain.__repr__())
        raise SkipTest # TODO: implement your test here

    def test_append(self):
        # chain = Chain()
        # assert_equal(expected, chain.append(edge, tol))
        raise SkipTest # TODO: implement your test here

    def test_bbox(self):
        # chain = Chain()
        # assert_equal(expected, chain.bbox())
        raise SkipTest # TODO: implement your test here

    def test_end(self):
        # chain = Chain()
        # assert_equal(expected, chain.end())
        raise SkipTest # TODO: implement your test here

    def test_from_dxf(self):
        # chain = Chain()
        # assert_equal(expected, chain.from_dxf(trans))
        raise SkipTest # TODO: implement your test here

    def test_start(self):
        # chain = Chain()
        # assert_equal(expected, chain.start())
        raise SkipTest # TODO: implement your test here

    def test_to_dxf(self):
        # chain = Chain()
        # assert_equal(expected, chain.to_dxf(split, **attr))
        raise SkipTest # TODO: implement your test here
    
    def test___init__(self):
        # chain = Chain(data)
        raise SkipTest # TODO: implement your test here

    def test_from_svg(self):
        # chain = Chain(data)
        # assert_equal(expected, chain.from_svg())
        raise SkipTest # TODO: implement your test here

    def test_from_pdf(self):
        # chain = Chain(data)
        # assert_equal(expected, chain.from_pdf(color))
        raise SkipTest # TODO: implement your test here

class TestDrawing:
    @classmethod
    def setup_class(self):
        self.path=os.path.dirname(os.path.abspath(__file__))
        self.dxf= Drawing(self.path+'/drawing.dxf')
        self.pdf= Drawing(self.path+'/drawing.pdf')
        self.svg= Drawing(self.path+'/drawing.svg')
        
    def test_load(self):
        pass #tested above
    
    def test_save(self):
        for ext in ['png','svg','pdf','dxf']:
            self.pdf.save(self.path+'/drawing.pdf.%s'%ext)
            self.svg.save(self.path+'/drawing.svg.%s'%ext)
            self.dxf.save(self.path+'/drawing.dxf.%s'%ext)
            

    def test___init__(self):
        # drawing = Drawing(filename, options, **kwargs)
        raise SkipTest # TODO: implement your test here

    def test_bbox(self):
        # drawing = Drawing(filename, options, **kwargs)
        # assert_equal(expected, drawing.bbox(layers, ignore))
        raise SkipTest # TODO: implement your test here

    def test_read_dxf(self):
        # drawing = Drawing(filename, options, **kwargs)
        # assert_equal(expected, drawing.read_dxf(filename, options, **kwargs))
        raise SkipTest # TODO: implement your test here

    def test_read_svg(self):
        # drawing = Drawing(filename, options, **kwargs)
        # assert_equal(expected, drawing.read_svg(filename, options, **kwargs))
        raise SkipTest # TODO: implement your test here

    def test_read_pdf(self):
        # drawing = Drawing(filename, options, **kwargs)
        # assert_equal(expected, drawing.read_pdf(filename, **kwargs))
        raise SkipTest # TODO: implement your test here

    def test_draw(self):
        # drawing = Drawing(filename, **kwargs)
        # assert_equal(expected, drawing.draw(fig, **kwargs))
        raise SkipTest # TODO: implement your test here

    def test_figure(self):
        # drawing = Drawing(filename, **kwargs)
        # assert_equal(expected, drawing.figure(**kwargs))
        raise SkipTest # TODO: implement your test here

    def test_render(self):
        # drawing = Drawing(filename, **kwargs)
        # assert_equal(expected, drawing.render(format, **kwargs))
        raise SkipTest # TODO: implement your test here

    def test_img(self):
        # drawing = Drawing(filename, **kwargs)
        # assert_equal(expected, drawing.img(size, border, box, layers, ignore, forcelayercolor, antialias, background))
        raise SkipTest # TODO: implement your test here

class TestImg2base64:
    def test_img2base64(self):
        # assert_equal(expected, img2base64(img, fmt))
        raise SkipTest # TODO: implement your test here

class TestSpline:
    def test___init__(self):
        # spline = Spline(points)
        raise SkipTest # TODO: implement your test here

    def test_bbox(self):
        # spline = Spline(points)
        # assert_equal(expected, spline.bbox())
        raise SkipTest # TODO: implement your test here

    def test_end(self):
        # spline = Spline(points)
        # assert_equal(expected, spline.end())
        raise SkipTest # TODO: implement your test here

    def test_start(self):
        # spline = Spline(points)
        # assert_equal(expected, spline.start())
        raise SkipTest # TODO: implement your test here

    def test_swap(self):
        # spline = Spline(points)
        # assert_equal(expected, spline.swap())
        raise SkipTest # TODO: implement your test here

    def test_xy(self):
        # spline = Spline(points)
        # assert_equal(expected, spline.xy())
        raise SkipTest # TODO: implement your test here

    def test_length(self):
        # spline = Spline(points)
        # assert_equal(expected, spline.length())
        raise SkipTest # TODO: implement your test here

    def test___copy__(self):
        # spline = Spline(points)
        # assert_equal(expected, spline.__copy__())
        raise SkipTest # TODO: implement your test here

if __name__=="__main__":
    import sys
    import nose
    from cStringIO import StringIO  
    module_name = sys.modules[__name__].__file__

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    result = nose.run(argv=[sys.argv[0], module_name, '-s'])
    sys.stdout = old_stdout
    print mystdout.getvalue()