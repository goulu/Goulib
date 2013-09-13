from nose.tools import assert_equal
from nose import SkipTest
from Goulib.colors import *

class TestRgbToHex:
    def test_rgb_to_hex(self):
        assert_equal( rgb_to_hex((0,16,255)),'#0010ff')

class TestHexToRgb:
    def test_hex_to_rgb(self):
        assert_equal(hex_to_rgb('#0010ff'),(0,16,255))
        assert_equal(hex_to_rgb('#0010ff',1./255),(0,16./255,1))

class TestColorRange:
    def test_color_range(self):
        c=color_range(5,'red','blue')
        assert_equal(c[0],color['red'])
        assert_equal(c[1],color['yellow'])
        assert_equal(c[2],color['lime'])
        assert_equal(c[3],color['aqua'])
        assert_equal(c[4],color['blue'])

if __name__ == "__main__":
    import nose
    nose.runmodule()

