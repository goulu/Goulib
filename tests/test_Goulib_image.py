#!/usr/bin/env python
# coding: utf8
from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *

from Goulib.image import *

from skimage import data

import os
path=os.path.dirname(os.path.abspath(__file__))
results=path+'\\results\\image\\' #path for results

def assert_image(image,name=None,convert=False):
    """ Checks if an image is present (not black, white, or low contrast
        :param image: Image to check
        :param name: str, optional file name. image is saved is specified
        :param convert: bool specifies if image is converted to RGB for saving
    """
    from skimage.exposure import is_low_contrast
    if name:
        image.save(results+name,autoconvert=convert)
    h=hash(image)
    assert_not_equal(h,0,'image is black')
    assert_not_equal(h,2**64-1,'image is white')
    if is_low_contrast(image.array):
        logging.warning('image %s has low contrast'%name)

class TestImage:
    @classmethod
    def setup_class(self):
        self.lena=Image(path+'/data/lena.png')
        assert_equal(self.lena,self.lena) #make sure image comparizon works
        assert_image(self.lena.grayscale('L'),'grayscale.png') #force to uint8
        self.gray=Image(results+'grayscale.png')
        self.camera=Image(data.camera())

    def test_pdf(self):
        return # for now for speed
        try:
            import pdfminer
        except:
            return #pass
        assert_image(Image(path+'/data/lena.pdf'),'pdf_out.png')
        assert_image(Image(path+'/data/Pantone Fan.pdf'),'Pantone Fan.png')
        #Image(path+'/data/Pantone Fan CMYKOGB Equinox.pdf').save('Pantone Fan CMYKOGB Equinox.png')

    def test___init__(self):
        lena2=Image(self.lena)
        assert_equal(self.lena,lena2)

    def test_generate(self):
        #from matrix
        from matplotlib import cm
        a=[[-x*x+y*y for x in range(128)] for y in range(128)]
        a=normalize(a)
        assert_image(Image(a),'generated.png')
        assert_image(Image(a,colormap=cm.spectral),'gen_colormap.png',True)

    def test___hash__(self):
        h1=hash(self.lena)
        h2=hash(self.gray)
        diff=h1^h2 #XOR
        diff=math2.digsum(diff,2) #number of different pixels
        assert_equal(h1,h2,msg='difference is %d pixels'%diff)

    def test___getitem__(self):
        pixel=self.gray[256,256]
        assert_equal(pixel,90)
        pixel=self.lena[256,256]
        assert_equal(pixel,[ 0.70588235 ,0.25490196 ,0.28235294]) # (180,65,72))
        left=self.lena[:,:256]
        right=self.lena[:,256:-1]
        face=self.lena[246:374,225:353]
        assert_image(face,'lena.face.png')
        face=face.grayscale()
        eye=face[3:35,-35:-3] # negative indexes are handy in some cases
        c=face.correlation(eye)
        # assert_image(c,"correlation.png")

    def test___lt__(self):
        # image = Image(data)
        # assert_equal(expected, image.__lt__(other))
        raise SkipTest

    def test___repr__(self):
        assert_equal(repr(self.lena),'Image(mode=RGB shape=(512, 512, 3) type=float64)')

    def test_mode(self):
        pass #useless ?

    def test_open(self):
        lena3=Image.open(path+'/data/lena.png')
        assert_equal(self.lena,lena3)

    def test_html(self):
        h=self.lena.convert('P').html()
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

    def test_render(self):
        from Goulib.notebook import h
        h(self.lena)

    def test_convert(self):
        for mode in modes:
            im=self.lena.convert(mode)
            assert_image(im,'convert_%s.png'%mode)
            try:
                im2=im.convert('RGB')
                assert_image(im2,'convert_%s_round_trip.png'%mode)
            except Exception as e:
                logging.error('%s round trip conversion failed with %s'%(mode,e))
                im2=im.convert('RGB')


    def test_split(self):
        rgb = self.lena.split()
        for im,c in zip(rgb,'RGB'):
            assert_image(im,'split_%s.png'%c)

        assert_image(Image(rgb),'RGB_merge.png')

        colors=['Cyan','Magenta','Yellow','blacK']
        cmyk=self.lena.split('CMYK')
        cmyk2=[im.colorize(col) for im,col in zip(cmyk,colors)]
        for im,c in zip(cmyk2,'CMYK'):
            assert_image(im,'split_%s.png'%c)

        assert_image(Image(cmyk,mode='CMYK'),'CMYK_merge.png')

        lab=self.lena.split('Lab')
        for im,c in zip(lab,'LAB'):
            assert_image(im,'split_%s.png'%c)

        lab=Image(lab,mode='LAB')
        assert_image(lab,'Lab.png')

    def test_filter(self):
        from PIL.ImageFilter import BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE, EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
        for f in [BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE, EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN]:
            r=self.lena.filter(f)
            assert_image(r,'filter_pil_%s.png'%f.__name__)

        from skimage.filters import sobel, prewitt, scharr, roberts
        from skimage.restoration import denoise_bilateral
        for f in [sobel, prewitt, scharr, roberts, denoise_bilateral,]:
            try:
                assert_image(self.lena.filter(f),'filter_sk_%s.png'%f.__name__)
            except:
                pass

    def test_dither(self):
        for k in dithering:
            im=self.gray.dither(k)*255
            assert_image(im,'dither_%s.png'%dithering[k])
        assert_image(self.lena.dither(),'dither_color_2.png')
        assert_image(self.lena.dither(n=4),'dither_color_4.png')


    def test_resize(self):
        size=128
        im=self.lena.resize((size,size))
        assert_image(im,'resize_%d.png'%size)
        im=self.camera.resize((size,size))
        assert_image(im,'camera_resize_%d.png'%size)

    def test_expand(self):
        size=128
        im=self.lena.resize((size,size))
        im=im.expand((size+1,size+1),.5,.5) #can you see that half pixel border ?
        assert_image(im,'expand_0.5 border.png')

    def test_add(self):
        dot=disk(20)
        im=Image()
        for i in range(3):
            for j in range(3):
                im=im.add(dot,(j*38.5,i*41.5),0.5)
        assert_image(im,'image_add.png')

    def test_colorize(self):
        cmyk=self.lena.split('CMYK')
        colors=['Cyan','Magenta','Yellow','blacK']
        cmyk2=[im.colorize(col) for im,col in zip(cmyk,colors)]
        back=cmyk2[0]-(-cmyk2[1])-(-cmyk2[2])-(-cmyk2[3]) #what a strange syntax ...
        assert_image(back,'image_add_sum_cmyk.png')
        assert_equal(self.lena.dist(back),0)

    def test_mul(self):
        mask=disk(self.lena.size[0]/2)
        res=self.lena*mask
        assert_image(res,'disk_mul.png')

    def test_shift(self):
        left=self.lena[:,0:256]
        right=self.lena[:,256:]
        blank=Image(size=(513,513),mode='RGB',color='white')
        blank.paste(left,(0,0))
        blank.paste(right.shift(1,1),(256,0))
        assert_image(blank,'image_stitched.png')

    def test___abs__(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.__abs__())
        raise SkipTest # TODO: implement your test here

    def test___add__(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.__add__(other))
        raise SkipTest # TODO: implement your test here

    def test___div__(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.__div__(f))
        raise SkipTest # TODO: implement your test here

    def test___mul__(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.__mul__(other))
        raise SkipTest # TODO: implement your test here

    def test___nonzero__(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.__nonzero__())
        raise SkipTest # TODO: implement your test here

    def test___radd__(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.__radd__(other))
        raise SkipTest # TODO: implement your test here

    def test___sub__(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.__sub__(other))
        raise SkipTest # TODO: implement your test here

    def test_compose(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.compose(other, a, b))
        raise SkipTest # TODO: implement your test here

    def test_correlation(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.correlation(other))
        raise SkipTest # TODO: implement your test here

    def test_crop(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.crop(lurb))
        raise SkipTest # TODO: implement your test here

    def test_getdata(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.getdata(dtype))
        raise SkipTest # TODO: implement your test here

    def test_getpixel(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.getpixel(yx))
        raise SkipTest # TODO: implement your test here

    def test_load(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.load(path))
        raise SkipTest # TODO: implement your test here

    def test_nchannels(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.nchannels())
        raise SkipTest # TODO: implement your test here

    def test_normalize(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.normalize(newmax, newmin))
        raise SkipTest # TODO: implement your test here

    def test_npixels(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.npixels())
        raise SkipTest # TODO: implement your test here

    def test_paste(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.paste(image, box, mask))
        raise SkipTest # TODO: implement your test here

    def test_putpixel(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.putpixel(yx, value))
        raise SkipTest # TODO: implement your test here

    def test_quantize(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.quantize(levels))
        raise SkipTest # TODO: implement your test here

    def test_save(self):
        pass #tested everywhere

    def test_scale(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.scale(s))
        raise SkipTest # TODO: implement your test here

    def test_shape(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.shape())
        raise SkipTest # TODO: implement your test here

    def test_size(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.size())
        raise SkipTest # TODO: implement your test here

    def test_threshold(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.threshold(level))
        raise SkipTest # TODO: implement your test here

    def test_getcolors(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.getcolors(maxcolors))
        raise SkipTest # TODO: implement your test here

    def test_getpalette(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.getpalette(maxcolors))
        raise SkipTest # TODO: implement your test here

    def test_setpalette(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.setpalette(p))
        raise SkipTest # TODO: implement your test here

    def test_new(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.new(size, color))
        raise SkipTest # TODO: implement your test here

    def test_pil(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.pil())
        raise SkipTest # TODO: implement your test here

    def test_optimize(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.optimize(maxcolors))
        raise SkipTest # TODO: implement your test here

    def test_replace(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.replace(pairs))
        raise SkipTest # TODO: implement your test here

    def test_sub(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.sub(other, pos, alpha, mode))
        raise SkipTest # TODO: implement your test here

    def test_deltaE(self):
        # image = Image(data, mode, **kwargs)
        # assert_equal(expected, image.deltaE(other))
        raise SkipTest # TODO: implement your test here

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

class TestNchannels:
    def test_nchannels(self):
        # assert_equal(expected, nchannels(arr))
        raise SkipTest # TODO: implement your test here

class TestGuessmode:
    def test_guessmode(self):
        # assert_equal(expected, guessmode(arr))
        raise SkipTest # TODO: implement your test here

class TestAdaptRgb:
    def test_adapt_rgb(self):
        # assert_equal(expected, adapt_rgb(func))
        raise SkipTest # TODO: implement your test here

class TestRgb2rgba:
    def test_rgb2rgba(self):
        pass #tested in test_convert

class TestDisk:
    def test_disk(self):
        # assert_equal(expected, disk(radius, antialias))
        raise SkipTest # TODO: implement your test here

class TestFspecial:
    def test_fspecial(self):
        # assert_equal(expected, fspecial(name, **kwargs))
        raise SkipTest # TODO: implement your test here

class TestReadPdf:
    def test_read_pdf(self):
        # assert_equal(expected, read_pdf(filename, **kwargs))
        raise SkipTest # TODO: implement your test here

class TestFig2img:
    def test_fig2img(self):
        # assert_equal(expected, fig2img(fig))
        raise SkipTest # TODO: implement your test here

class TestQuantize:
    def test_quantize(self):
        # assert_equal(expected, quantize(image, N, L))
        raise SkipTest # TODO: implement your test here

class TestDither:
    def test_dither(self):
        # assert_equal(expected, dither(image, method, N, L))
        raise SkipTest # TODO: implement your test here

class TestGray2rgb:
    def test_gray2rgb(self):
        pass #tested in test_convert

class TestBool2gray:
    def test_bool2gray(self):
        pass #tested in test_convert
        raise SkipTest # TODO: implement your test here

class TestRgba2rgb:
    def test_rgba2rgb(self):
        pass #tested in test_convert

class TestPalette:
    def test_palette(self):
        # assert_equal(expected, palette(im, ncolors))
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # palette = Palette(data, n)
        raise SkipTest # TODO: implement your test here

    def test_index(self):
        # palette = Palette(data, n)
        # assert_equal(expected, palette.index(c, dE))
        raise SkipTest # TODO: implement your test here

    def test_pil(self):
        # palette = Palette(data, n)
        # assert_equal(expected, palette.pil())
        raise SkipTest # TODO: implement your test here

    def test_update(self):
        # palette = Palette(data, n)
        # assert_equal(expected, palette.update(data, n))
        raise SkipTest # TODO: implement your test here

    def test___repr__(self):
        # palette = Palette(data, n)
        # assert_equal(expected, palette.__repr__())
        raise SkipTest # TODO: implement your test here

    def test_patches(self):
        # palette = Palette(data, n)
        # assert_equal(expected, palette.patches(wide, size))
        raise SkipTest # TODO: implement your test here

    def test_sorted(self):
        # palette = Palette(data, n)
        # assert_equal(expected, palette.sorted(key))
        raise SkipTest # TODO: implement your test here

class TestLab2ind:
    def test_lab2ind(self):
        raise SkipTest # TODO: implement your test here

class TestInd2any:
    def test_ind2any(self):
        # assert_equal(expected, ind2any(im, palette, dest))
        raise SkipTest # TODO: implement your test here

class TestInd2rgb:
    def test_ind2rgb(self):
        # assert_equal(expected, ind2rgb(im, palette))
        raise SkipTest # TODO: implement your test here

class TestRandomize:
    def test_randomize(self):
        # assert_equal(expected, randomize(image, N, L))
        raise SkipTest # TODO: implement your test here

class TestDitherer:
    def test___call__(self):
        # ditherer = Ditherer(name, method)
        # assert_equal(expected, ditherer.__call__(image, N))
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # ditherer = Ditherer(name, method)
        raise SkipTest # TODO: implement your test here

    def test___repr__(self):
        # ditherer = Ditherer(name, method)
        # assert_equal(expected, ditherer.__repr__())
        raise SkipTest # TODO: implement your test here

class TestErrorDiffusion:
    def test___call__(self):
        # error_diffusion = ErrorDiffusion(name, positions, weights)
        # assert_equal(expected, error_diffusion.__call__(image, N))
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # error_diffusion = ErrorDiffusion(name, positions, weights)
        raise SkipTest # TODO: implement your test here

class TestFloydSteinberg:
    def test___call__(self):
        # floyd_steinberg = FloydSteinberg()
        # assert_equal(expected, floyd_steinberg.__call__(image, N))
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # floyd_steinberg = FloydSteinberg()
        raise SkipTest # TODO: implement your test here

class TestNormalize:
    def test_normalize(self):
        # assert_equal(expected, normalize(a, newmax, newmin))
        raise SkipTest # TODO: implement your test here

if __name__=="__main__":
    runmodule()
