from nose.tools import assert_equal
from nose import SkipTest

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

class TestTrans:
    def test_trans(self):
        # assert_equal(expected, Trans(scale, offset, rotation))
        raise SkipTest # TODO: implement your test here

class TestDXF:
    def test___init__(self):
        # d_x_f = DXF(file, layers, ignore)
        raise SkipTest # TODO: implement your test here

    def test_bbox(self):
        # d_x_f = DXF(file, layers, ignore)
        # assert_equal(expected, d_x_f.bbox())
        raise SkipTest # TODO: implement your test here

    def test_entities(self):
        # d_x_f = DXF(file, layers, ignore)
        # assert_equal(expected, d_x_f.entities(ent))
        raise SkipTest # TODO: implement your test here

    def test_img(self):
        # d_x_f = DXF(file, layers, ignore)
        # assert_equal(expected, d_x_f.img(size, back, pen, border, antialias))
        raise SkipTest # TODO: implement your test here

class TestImg2base64:
    def test_img2base64(self):
        # assert_equal(expected, img2base64(img, fmt))
        raise SkipTest # TODO: implement your test here

