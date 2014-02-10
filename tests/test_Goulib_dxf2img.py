from nose.tools import assert_equal
from nose import SkipTest

import os

from Goulib.dxf2img import *

class TestTrans:
    def test_trans(self):
        # assert_equal(expected, Trans(scale, offset, rotation))
        raise SkipTest # TODO: implement your test here

class TestRint:
    def test_rint(self):
        # assert_equal(expected, rint(x))
        raise SkipTest # TODO: implement your test here

class TestBBox:
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

    def test___add__(self):
        # b_box = BBox(pt1, pt2)
        # assert_equal(expected, b_box.__add__(other))
        raise SkipTest # TODO: implement your test here

    def test_xmax(self):
        # b_box = BBox(pt1, pt2)
        # assert_equal(expected, b_box.xmax())
        raise SkipTest # TODO: implement your test here

    def test_xmin(self):
        # b_box = BBox(pt1, pt2)
        # assert_equal(expected, b_box.xmin())
        raise SkipTest # TODO: implement your test here

    def test_ymax(self):
        # b_box = BBox(pt1, pt2)
        # assert_equal(expected, b_box.ymax())
        raise SkipTest # TODO: implement your test here

    def test_ymin(self):
        # b_box = BBox(pt1, pt2)
        # assert_equal(expected, b_box.ymin())
        raise SkipTest # TODO: implement your test here

class TestCbox:
    def test_cbox(self):
        # assert_equal(expected, cbox(c, r))
        raise SkipTest # TODO: implement your test here

class TestDXF:
    @classmethod
    def setup_class(self):
        self.path=os.path.dirname(os.path.abspath(__file__))
        self.dxf= DXF(self.path+'/test.dxf')
        pass
        
    def test___init__(self):
        pass #tested in setup

    def test_bbox(self):
        # d_x_f = DXF(filename, options)
        # assert_equal(expected, d_x_f.bbox(layers, ignore))
        raise SkipTest # TODO: implement your test here

    def test_img(self):
        img = self.dxf.img(size=[512, None], border=5,forcelayercolor=False, background='white')
        img.save('test.png') #todo add a test

    def test_iter(self):
        # d_x_f = DXF(filename, options)
        # assert_equal(expected, d_x_f.iter(ent, layers, only, ignore, trans, recurse))
        raise SkipTest # TODO: implement your test here

class TestImg2base64:
    def test_img2base64(self):
        # assert_equal(expected, img2base64(img, fmt))
        raise SkipTest # TODO: implement your test here

class TestFactory:
    def test_factory(self):
        # assert_equal(expected, factory(e, trans))
        raise SkipTest # TODO: implement your test here

if __name__ == "__main__":
    import nose
    nose.runmodule()

