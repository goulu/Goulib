#!/usr/bin/env python
# coding: utf8

from __future__ import division #"true division" everywhere

from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *

from Goulib.colors import *
from Goulib.itertools2 import reshape

import os
path=os.path.dirname(os.path.abspath(__file__))

class TestRgb2Hex:
    def test_rgb2hex(self):
        assert_equal( rgb2hex((0,16/255,1)),'#0010ff')

class TestHex2Rgb:
    def test_hex2rgb(self):
        assert_equal(hex2rgb('#0010ff'),(0,16./255,1))

class TestRgb2Cmyk:
    def test_rgb2cmyk(self):
        assert_equal(rgb2cmyk((0,0,0)),(0,0,0,1))
        assert_equal(rgb2cmyk((.8,.6,.4)),(0,0.25,.5,0.2))

class TestNearestColor:
    def test_nearest_color(self):
        assert_equal(nearest_color('#414142'),color['darkslategray'])
        cmyk=Color((.45,.12,.67,.05),mode='cmyk')
        p=nearest_color(cmyk,pantone)
        assert_equal(p.name,'802C')
        dE=deltaE

class TestAci:
    def test_color_to_aci(self):
        assert_equal(color_to_aci('red'), 1)
        assert_equal(color_to_aci(acadcolors[123]), 123)
        c=color_to_aci('#414142',True)
        assert_equal(acadcolors[c].hex,'#414141')

class TestColorRange:
    def test_color_range(self):
        c=color_range(5,'red','blue')
        assert_equal(c[0],color['red'])
        assert_equal(c[1],color['yellow'])
        assert_equal(c[2],color['lime'])
        assert_equal(c[3],color['cyan'])
        assert_equal(c[4],color['blue'])

class TestColor:
    @classmethod
    def setup_class(self):
        self.blue=Color('blue')
        self.blues=[ #many ways to define the same color
            self.blue,
            Color('#0000ff'),
            Color((0,0,1)),
            Color((0,0,255)),
            Color(self.blue),
        ]
        
        self.red=Color('red')
        self.green=Color('lime') # 'green' has hex 80 value, not ff
        
        self.white=color['white']
        
        self.cmyk=Color((.45,.12,.67,.05),mode='cmyk') # a random cmyk color
        
    def test___init__(self):
        #check all constructors make the same color
        for b in self.blues:
            assert_equal(b,self.blue)

    def test___add__(self):

        assert_equal(self.red+self.green+self.blue,'white')
        assert_equal(self.red+self.green+self.blue,'white')

    def test___sub__(self):
        white=Color('white')
        green=Color('lime') # 'green' has hex 80 value, not ff
        blue=Color('blue')
        assert_equal(white-green-blue,'red')

    def test___eq__(self):
        pass #tested above

    def test___repr__(self):
        assert_equal(repr(Color('blue')),"Color('blue')")

    def test__repr_html_(self):
        assert_equal(Color('blue')._repr_html_(),'<p style="color:#0000ff">blue</p>')

    def test_rgb(self):
        pass #tested above

    def test_hex(self):
        assert_equal(self.cmyk.hex,'#80d447')
        #TODO : find why http://www.ginifab.com/feeds/pms/cmyk_to_pantone.php gives #85d550

    def test_cmyk(self):
        assert_equal(Color('black').cmyk,(0,0,0,1))
        assert_equal(Color('blue').cmyk,(1,1,0,0))
        assert_equal(Color((0,.5,.5)).cmyk,(1,0,0,.5)) #teal
    

class TestColorLookup:
    def test_color_lookup(self):
        c=color['blue']
        c2=color_lookup[c.hex]
        assert_equal(c,c2)

class TestColorToAci:
    def test_color_to_aci(self):
        # assert_equal(expected, color_to_aci(x, nearest))
        raise SkipTest

class TestAciToColor:
    def test_aci_to_color(self):
        # assert_equal(expected, aci_to_color(x, block_color, layer_color))
        raise SkipTest

class TestPantone:
    def test_pantone(self):
        from Goulib.table import Table,Cell
        from Goulib.itertools2 import reshape

        t=[Cell(name,style={'background-color':pantone[name].hex}) for name in sorted(pantone)]
        t=Table(reshape(t,(0,10)))
        with open(path+'\\results\\colors.pantone.html', 'w') as f:
            f.write(t.html())

if __name__ == "__main__":
    runmodule()

