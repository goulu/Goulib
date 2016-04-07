#!/usr/bin/env python
# coding: utf8
from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *

from Goulib.image import *
from PIL.ImageFilter import *

import os
path=os.path.dirname(os.path.abspath(__file__))
results=path+'\\results\\image\\' #path for results

class TestImage:
    @classmethod
    def setup_class(self):
        self.lena=Image(path+'/data/lena.png')
        assert_equal(self.lena,self.lena) #make sure image comparizon works
        self.lena.grayscale().save(results+'lena_gray.png')
        self.lena_gray=Image(results+'lena_gray.png')

    def test_pdf(self):
        try:
            import pdfminer
        except:
            return #pass
        Image(path+'/data/lena.pdf').save(results+'lena_pdf_out.png')
        Image(path+'/data/Pantone Fan.pdf').save(results+'Pantone Fan.png')
        #Image(path+'/data/Pantone Fan CMYKOGB Equinox.pdf').save(results+'Pantone Fan CMYKOGB Equinox.png')

    def test___init__(self):
        lena2=Image(self.lena)
        assert_equal(self.lena,lena2)
        lena3=Image().open(path+'/data/lena.png')
        assert_equal(self.lena,lena3)

        #from matrix
        from matplotlib import cm
        a=[[-x*y for x in range(128)] for y in range(128)]
        Image(a).save(results+'generated.png')
        Image(a,colormap=cm.spectral).save(results+'gen_colormap.png')

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
        face.save(path+"/results/image/image.lena.face.png")
        face=face.grayscale()
        eye=face[3:35,-35:-3] # negative indexes are handy in some cases
        c=face.correlation(eye)
        c.save(path+"/results/image/image.correlation.png")

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

    def test_split(self):
        colors=['Cyan','Magenta','Yellow','blacK']
        cmyk=self.lena.split('CMYK')
        cmyk=[im.colorize('white',col) for im,col in zip(cmyk,colors)]
        for im,c in zip(cmyk,'CMYK'):
            im.save(results+'lena_split_%s.png'%c)

    def test_dither(self):
        for k in dithering:
            self.lena_gray.dither(k).save(results+'lena_dither_%s.png'%dithering[k])
        self.lena.dither().save(results+'lena_color_dither.png')


    def test_filter(self):

        for f in [BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE, EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN]:
            self.lena.filter(f).save(results+'lena_%s.png'%f.__name__)

        from skimage.filters import sobel
        for f in [sobel]:
            self.lena.filter(f).save(results+'lena_sk_%s.png'%f.__name__)

    def test_expand(self):
        size=64
        im=self.lena.resize((size,size))
        im=im.expand((size+1,size+1),.5,.5) #can you see that half pixel border ?
        im.save(results+'lena_expand_0.5 border.png')

    def test_add(self):
        dot=disk(20)
        im=Image()
        for i in range(3):
            for j in range(3):
                im=im.add(dot,(j*38.5,i*41.5),0.5)
        im.save(results+'image_add.png')

        cmyk=self.lena.split('CMYK')
        colors=['Cyan','Magenta','Yellow','blacK']
        cmyk=[im.colorize('white',col) for im,col in zip(cmyk,colors)]
        back=sum(cmyk,Image())
        assert_equal(self.lena.dist(back),0)
        
    def test_mul(self):
        mask=disk(self.lena.size[0]/2)
        res=self.lena*mask
        res.save(results+'lena_disk_mul.png')

    def test_shift(self):
        left=self.lena[:,0:256]
        right=self.lena[:,256:]
        blank=Image(size=(513,513),mode='RGB',color='white')
        blank.paste(left,(0,0))
        blank.paste(right.shift(1,1),(256,0))
        blank.save(results+'image_stitched.png')

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
