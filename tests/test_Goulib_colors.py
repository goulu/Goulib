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

class TestColor:
    def test___init__(self):
        blue1=Color('blue')
        blue2=Color('#0000ff')
        assert_equal(blue1,blue2)
        blue3=Color((0,0,1))
        assert_equal(blue1,blue3)
        blue4=Color((0,0,255))
        assert_equal(blue1,blue4)
    
    def test___add__(self):
        red=Color('red')
        green=Color('lime') # 'green' has hex 80 value, not ff
        blue=Color('blue')
        assert_equal(red+green+blue,'white')



    def test___eq__(self):
        # color = Color(c)
        # assert_equal(expected, color.__eq__(other))
        raise SkipTest # TODO: implement your test here

    def test___repr__(self):
        # color = Color(c)
        # assert_equal(expected, color.__repr__())
        raise SkipTest # TODO: implement your test here

    def test_hex(self):
        # color = Color(c)
        # assert_equal(expected, color.hex())
        raise SkipTest # TODO: implement your test here

if __name__ == "__main__":
    import nose
    nose.runmodule()

