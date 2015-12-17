#!/usr/bin/env python
# coding: utf8
from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *

from Goulib.image import *

import os
path=os.path.dirname(os.path.abspath(__file__))

class TestImage:
    @classmethod
    def setup_class(self):
        self.lena=Image(path+'/data/lena.png')
        self.lena.convert('L').save(path+'/results/lena_bw.png')
        self.lena_bw=Image(path+'/results/lena_bw.png')
        
    def test___init__(self):
        lena2=Image(self.lena)
        assert_equal(self.lena,lena2)
        lena3=Image().open(path+'/data/lena.png')
        assert_equal(self.lena,lena3)
        
    def test___hash__(self):
        h1=hash(self.lena)
        h2=hash(self.lena_bw)
        diff=h1^h2 #XOR
        diff=math2.digsum(diff,2) #number of different pixels
        assert_equal(h1,h2,msg='difference is %d pixels'%diff)

    def test___lt__(self):
        # image = Image(data)
        # assert_equal(expected, image.__lt__(other))
        raise SkipTest 

    def test___repr__(self):
        # image = Image(data)
        # assert_equal(expected, image.__repr__())
        raise SkipTest 

    def test_mode(self):
        # image = Image(data)
        # assert_equal(expected, image.mode())
        raise SkipTest 

    def test_open(self):
        pass #tested above
    
    
    def test_split(self):
        # split the image into individual bands
        source = self.lena.split()
        
    def test_html(self):
        h=self.lena.html()
        assert_true(h)

class TestNormalizeArray(unittest.TestCase):
    def test_normalize_array(self):
        raise SkipTest 

class TestPil2array(unittest.TestCase):
    def test_pil2array(self):
        raise SkipTest 

class TestArray2pil(unittest.TestCase):
    def test_array2pil(self):
        raise SkipTest 

class TestCorrelation(unittest.TestCase):
    def test_correlation(self):
        raise SkipTest 

class TestAlphaToColor(unittest.TestCase):
    def test_alpha_to_color(self):
        raise SkipTest 

class TestAlphaComposite(unittest.TestCase):
    def test_alpha_composite(self):
        raise SkipTest 

class TestAlphaCompositeWithColor(unittest.TestCase):
    def test_alpha_composite_with_color(self):
        raise SkipTest 

class TestPurePilAlphaToColorV1(unittest.TestCase):
    def test_pure_pil_alpha_to_color_v1(self):
        raise SkipTest 

class TestPurePilAlphaToColorV2(unittest.TestCase):
    def test_pure_pil_alpha_to_color_v2(self):
        raise SkipTest 


class TestNormalize:
    def test_normalize(self):
        # assert_equal(expected, normalize(X, norm, axis, copy, positive))
        raise SkipTest 

if __name__=="__main__":
    runmodule()
