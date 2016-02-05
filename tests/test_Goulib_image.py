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
        self.lena.grayscale().save(path+'/results/lena_gray.png')
        self.lena_gray=Image(path+'/results/lena_gray.png')
        
    def test___init__(self):
        lena2=Image(self.lena)
        assert_equal(self.lena,lena2)
        lena3=Image().open(path+'/data/lena.png')
        assert_equal(self.lena,lena3)
        
        #from matrix
        a=[[(x-64)*y for x in range(128)] for y in range(128)]
        a=Image(a)
        
    def test___hash__(self):
        h1=hash(self.lena)
        h2=hash(self.lena_gray)
        diff=h1^h2 #XOR
        diff=math2.digsum(diff,2) #number of different pixels
        assert_equal(h1,h2,msg='difference is %d pixels'%diff)
        
    def test___getitem__(self):
        pixel=self.lena_gray[256,256]
        assert_equal(pixel,100)
        pixel=self.lena[256,256]
        assert_equal(pixel,(180,65,72))
        left=self.lena[:,:256]
        right=self.lena[:,256:-1]
        face=self.lena[246:374,225:353]
        face.save(path+"/results/image.lena.face.png")
        face=face.grayscale()
        eye=face[3:35,-35:-3] # negative indexes are handy in some cases
        c=face.correlation(eye)
        c.save(path+"/results/image.correlation.png")

    def test___lt__(self):
        # image = Image(data)
        # assert_equal(expected, image.__lt__(other))
        raise SkipTest 

    def test___repr__(self):
        # image = Image(data)
        # assert_equal(expected, image.__repr__())
        raise SkipTest 

    def test_mode(self):
        pass #useless ? 

    def test_open(self):
        pass #tested above
    
    
    def test_split(self):
        # split the image into individual bands
        source = self.lena.split()
        
    def test_html(self):
        h=self.lena.html()
        assert_true(h)

    def test_average_hash(self):
        # image = Image(data, **kwargs)
        # assert_equal(expected, image.average_hash(hash_size))
        raise SkipTest # TODO: implement your test here

    def test_base64(self):
        # image = Image(data, **kwargs)
        # assert_equal(expected, image.base64(fmt))
        raise SkipTest # TODO: implement your test here

    def test_dist(self):
        # image = Image(data, **kwargs)
        # assert_equal(expected, image.dist(other, hash_size))
        raise SkipTest # TODO: implement your test here

    def test_grayscale(self):
        pass

    def test_invert(self):
        # image = Image(data, **kwargs)
        # assert_equal(expected, image.invert())
        raise SkipTest # TODO: implement your test here

    def test_ndarray(self):
        pass

    def test_to_html(self):
        pass
    
    def test_filter(self):
        if not SKIMAGE: #optional requirement
            raise SkipTest
        from skimage.filters import sobel
        for f in [sobel]:
            assert_true(self.lena.filter(f))
            
    def test_expand(self):
        size=64
        im=self.lena.resize((size,size))
        im=im.expand((size+1,size+1),.5,.5) #can you see that half pixel border ?
        im.save(path+'/results/lena_expand_0.5 border.png')
        
    def test_add(self):
        dot=disk(20)
        im=Image()
        for i in range(3):
            for j in range(3):
                im=im.add(dot,j*38.5,i*41.5,0.5)
        im.save(path+'/results/image_add.png')

class TestCorrelation:
    def test_correlation(self):
        # assert_equal(expected, correlation(input, match))
        raise SkipTest # TODO: implement your test here

class TestAlphaToColor:
    def test_alpha_to_color(self):
        # assert_equal(expected, alpha_to_color(image, color))
        raise SkipTest # TODO: implement your test here

class TestAlphaComposite:
    def test_alpha_composite(self):
        # assert_equal(expected, alpha_composite(front, back))
        raise SkipTest # TODO: implement your test here

class TestAlphaCompositeWithColor:
    def test_alpha_composite_with_color(self):
        # assert_equal(expected, alpha_composite_with_color(image, color))
        raise SkipTest # TODO: implement your test here

class TestPurePilAlphaToColorV1:
    def test_pure_pil_alpha_to_color_v1(self):
        # assert_equal(expected, pure_pil_alpha_to_color_v1(image, color))
        raise SkipTest # TODO: implement your test here

class TestPurePilAlphaToColorV2:
    def test_pure_pil_alpha_to_color_v2(self):
        # assert_equal(expected, pure_pil_alpha_to_color_v2(image, color))
        raise SkipTest # TODO: implement your test here

if __name__=="__main__":
    runmodule()
