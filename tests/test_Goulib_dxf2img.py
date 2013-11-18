from nose.tools import assert_equal
from nose import SkipTest

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

class TestCbox:
    def test_cbox(self):
        # assert_equal(expected, cbox(c, r))
        raise SkipTest # TODO: implement your test here

class TestDXF:
    def setup(self):
        self.dxf= DXF("test.dxf")
        
    def test___init__(self):
        pass #tested in setup

    def test_bbox(self):
        # d_x_f = DXF(filename, options)
        # assert_equal(expected, d_x_f.bbox(layers, ignore))
        raise SkipTest # TODO: implement your test here

    def test_img(self):
        img = self.dxf.img(size=[512, None], border=5,forcelayercolor=True)
        img.save('test.png') #todo add a test

    def test_iter(self):
        # d_x_f = DXF(filename, options)
        # assert_equal(expected, d_x_f.iter(ent, layers, only, ignore, trans, recurse))
        raise SkipTest # TODO: implement your test here

class TestImg2base64:
    def test_img2base64(self):
        # assert_equal(expected, img2base64(img, fmt))
        raise SkipTest # TODO: implement your test here

if __name__ == "__main__":
    import nose
    nose.runmodule()

