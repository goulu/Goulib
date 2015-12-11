#!/usr/bin/env python
# coding: utf8
from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *

import os, sys
from Goulib.image import *

from PIL import Image

path=os.path.dirname(os.path.abspath(__file__))

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

class TestImg2base64:
    def test_img2base64(self):
        # assert_equal(expected, img2base64(img, fmt))
        raise SkipTest
    
class TestAveragehash:
    def test_average_hash(self):
        lena = Image.open(path+'/data/lena.png')
        lena_bw = Image.open(path+'/data/lena_bw.png')
        diff=average_hash(lena)^average_hash(lena_bw)
        diff=math2.digsum(diff,2) #number of different pixels
        assert_true(diff<2)
        

if __name__=="__main__":
    runmodule()
