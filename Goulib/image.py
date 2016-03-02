#!/usr/bin/env python
# coding: utf8
"""
image processing and conversion

:requires:
* `PIL of Pillow <http://pypi.python.org/pypi/pillow/>`_

:optional:
* `pdfminer.six <http://pypi.python.org/pypi/pdfminer.six/>`_ for pdf input
* `scikit-image <http://scikit-image.org/>`_ for advanced filtering
"""
from __future__ import division #"true division" everywhere

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2015, Philippe Guglielmetti"
__credits__ = ['Brad Montgomery http://bradmontgomery.net']
__license__ = "LGPL"

from PIL import Image as PILImage
from PIL import ImagePalette, ImageOps, ImageDraw

# from PIL import ImageCms #disabled for now

try: # http://scikit-image.org/ is optional
    import skimage
    SKIMAGE=True
except:
    SKIMAGE=False

import numpy as np

import six, math, base64, functools, logging

from . import math2
from .drawing import Drawing #to read vector pdf files as images

#dithering methods
PHILIPS=11

dithering={
    PILImage.NEAREST : 'nearest',
    PILImage.ORDERED : 'ordered', # Not yet implemented in Pillow
    PILImage.RASTERIZE : 'rasterize', # Not yet implemented in Pillow
    PILImage.FLOYDSTEINBERG : 'floyd-steinberg',
    PHILIPS: 'philips', #http://www.google.com/patents/WO2002039381A2
}

def adapt_rgb(func):
    """Decorator that adapts to RGB(A) images to a gray-scale filter.
    :param apply_to_rgb: function
        Function that returns a filtered image from an image-filter and RGB
        image. This will only be called if the image is RGB-like.
    """
    # adapted from https://github.com/scikit-image/scikit-image/blob/master/skimage/color/adapt_rgb.py
    @functools.wraps(func)
    def image_filter_adapted(image, *args, **kwargs):
        channels=list(image.split())
        if len(channels)>1:
            for i in range(3): #RGB. If there is an A, it is untouched
                channels[i]=(func)(channels[i], *args, **kwargs)
                if channels[i].mode=='1':
                    channels[i]=channels[i].grayscale()
            return PILImage.merge(image.mode, channels)
        else:
            return func(image, *args, **kwargs)
    return image_filter_adapted

class Image(PILImage.Image):
    def __init__(self, data=None, **kwargs):
        """
        :param data: can be either:
        * `PIL.Image` : makes a copy
        * string : path of image to load
        * None : creates an empty image with kwargs parameters:
        ** size : (y,x) pixel size tuple
        ** mode : 'L' (gray) by default
        ** color: to fill None=black by default
        """
        if data is None:
            if kwargs:
                kwargs.setdefault('mode','L')
                im=PILImage.new(**kwargs)
            else:
                im=PILImage.Image()
            self._initfrom(im)
        elif isinstance(data,PILImage.Image):
            self._initfrom(data)
        elif isinstance(data,six.string_types): #assume a path
            self.open(data,**kwargs)
        else: #assume a np.ndarray
            # http://stackoverflow.com/questions/10965417/how-to-convert-numpy-array-to-pil-image-applying-matplotlib-colormap
            data=np.asarray(data)
            if data.max()>255 or data.min()<0: #if data is out of bounds
                data=_normalize(data,255,0)
            if data.max()<=1:
                data*=255
            colormap=kwargs.pop('colormap',None)
            if colormap:
                data=colormap(data,bytes=True)
            elif data.dtype is not np.uint8:
                data = np.uint8(data)
            self._initfrom(PILImage.fromarray(data))

    def open(self,path):
        self.path=path
        ext=path[-3:].lower()
        if ext=='pdf':
            im=read_pdf(path)
        else:
            im=PILImage.open(path)

        if im is None:
            raise IOError('could not read %s'%path)

        try:
            im.load()
            self.error=None
        except IOError as error: #truncated file ?
            self.error=error
        self._initfrom(im)
        return self

    def save(self,path,format=None, **kwargs):
        try:
            super(Image,self).save(path,format,**kwargs)
            return self
        except IOError as e:
            pass
        try:
            im=self.convert('RGBA')
        except IOError:
            im=self.convert('L') #gray

        return im.save(path,format,**kwargs)

    def _initfrom(self,other):
        #does `PIL.Image.Image._new` "reversed"
        self.im = other.im
        self.mode=other.mode
        self.size=other.size
        if other.palette:
            self.palette = other.palette.copy()
        elif self.im and self.im.mode == "P":
            self.palette = ImagePalette.ImagePalette()
        else:
            self.palette=None

        self.info = other.info.copy()
        #
        #are lines below useful ? they are inited in PIL.Image.__init__
        self.category = other.category
        self.readonly = other.readonly
        self.pyaccess = other.pyaccess

    def _new(self, im):
        #overloads `PIL.Image.Image._new`
        #so that it is called by most usual PIL(low) functions
        #that will therefore return a Goulib.Image instead of a PIL.Image
        new=Image(self)
        new.im = im
        new.mode=im.mode
        new.size=im.size
        return new

    # representations, data extraction and conversions

    def __repr__(self):
        path=getattr(self,'path',None)
        return "<%s path=%s mode=%s size=%dx%d>" % (
            self.__class__.__name__, path,
            self.mode, self.size[0], self.size[1],
            )


    def base64(self, fmt='PNG'):
        """
        :param fmt: string file format ('PNG', 'JPEG', ...
                    see http://pillow.readthedocs.org/en/3.0.x/handbook/image-file-formats.html )
        :result: string base64 encoded image content in specified format
        """
        # http://stackoverflow.com/questions/31826335/how-to-convert-pil-image-image-object-to-base64-string

        buffer = six.BytesIO()
        self.save(buffer, format=fmt)
        return base64.b64encode(buffer.getvalue())

    def to_html(self):
        s=self.base64('PNG').decode('utf-8')
        return r'<img src="data:image/png;base64,{0}">'.format(s)

    def html(self):
        from IPython.display import HTML
        return HTML(self.to_html())

    def _repr_html_(self):
        #this returns exactly the same as _repr_png_, but is Table compatible
        return self.to_html()

    @property
    def npixels(self):
        return math2.mul(self.size)

    def __nonzero__(self):
        return self.npixels >0

    def __lt__(self, other):
        return  self.npixels < other.pixels

    def ndarray(self):
        """ http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.ndarray.html

        :return: `numpy.ndarray` of image
        """
        data = list(self.getdata())
        w,h = self.size
        A = np.zeros((w*h), 'd')
        i=0
        for val in data:
            A[i] = val
            i=i+1
        A=A.reshape(w,h)
        return A

    def __getitem__(self,slice):
        try:
            return self.getpixel(slice)
        except TypeError:
            pass
        left, upper, right, lower=slice[1].start,slice[0].start,slice[1].stop,slice[0].stop
        # calculate box module size so we handle negative coords like in slices
        w,h = self.size
        upper = upper%h if upper else 0
        lower = lower%h if lower else h
        left = left%w if left else 0
        right = right%w if right else w

        im=self.crop((left, upper, right, lower))
        im.load()
        return Image(im)

    # hash and distance

    def average_hash(self, hash_size=8):
        """
        Average Hash computation
        Implementation follows http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

        :param hash_size: int sqrt of the hash size. 8 (64 bits) is perfect for usual photos
        :return: list of hash_size*hash_size bool (=bits)
        """
        # https://github.com/JohannesBuchner/imagehash/blob/master/imagehash/__init__.py
        image = self.convert("L").resize((hash_size, hash_size), PILImage.ANTIALIAS)
        pixels = np.array(image.getdata()).reshape((1,hash_size*hash_size))[0]
        avg = pixels.mean()
        diff=pixels > avg
        return math2.num_from_digits(diff,2)

    def dist(self,other, hash_size=8):
        """ distance between images

        :param hash_size: int sqrt of the hash size. 8 (64 bits) is perfect for usual photos
        :return: float
            =0 if images are equal or very similar (same average_hash)
            =1 if images are completely decorrelated (half of the hash bits are the same by luck)
            =2 if images are inverted
        """
        h1=self.average_hash(hash_size)
        h2=other.average_hash(hash_size)
        # http://stackoverflow.com/questions/9829578/fast-way-of-counting-non-zero-bits-in-python
        diff=bin(h1^h2).count("1") # ^is XOR
        diff=2*diff/(hash_size*hash_size)
        return diff

    def __hash__(self):
        return self.average_hash(8)

    def __abs__(self):
        """:return: float Frobenius norm of image"""
        res= np.linalg.norm(np.array(self,np.float))
        return res

    def invert(self):
        # http://stackoverflow.com/questions/2498875/how-to-invert-colors-of-image-with-pil-python-imaging
        return ImageOps.invert(self)

    __neg__=__inv__=invert #aliases

    def grayscale(self):
        return self.convert("L")

    def colorize(self,color0,color1=None):
        """colorize a grayscale image

        :param color0,color1: 2 colors.
            - If only one is specified, image is colorized from white (for 0) to the specified color (for 1)
            - if 2 colors are specified, image is colorized from color0 (for 0) to color1 (for 1)
        :return: RGB(A) color
        """
        if color1 is None:
            color1=color0
            color0='white'
        return ImageOps.colorize(self,color0,color1)

    @adapt_rgb
    def dither(self,method=PILImage.FLOYDSTEINBERG):
        if method <=PILImage.FLOYDSTEINBERG:
            return self.convert('1',dither=method)
        elif method == PHILIPS:
            width, height = self.size
            mat=self.load()

            new=Image(mode='1',size=(width,height))
            out=new.load()

            def philips(line):
                e=0
                for pixelin in line:
                    e+=pixelin
                    c=int(e>127)
                    e-=c*255
                    yield c

            #col0=[mat[i,0] for i in range(width)]
            for i in range(height):
                row=[mat[i,j] for j in range(width)]
                for j,c in enumerate(philips(row)):
                    out[i,j]=c
            return new
        else:
            raise(NotImplemented)


    def normalize(self,newmax=255,newmin=0):
        #http://stackoverflow.com/questions/7422204/intensity-normalization-of-image-using-pythonpil-speed-issues
        #warning : this normalizes each channel independently, so we don't use @adapt_rgb here
        arr=_normalize(np.array(self),newmax,newmin)
        return Image(arr)

    def split(self, mode=None):
        if mode and mode != self.mode:
            im=self.convert(mode)
        else:
            im=self
        return super(Image,im).split()

    @adapt_rgb
    def filter(self,f):
        try: # scikit-image filter or similar ?
            return Image(f(self))
        except TypeError: #probably a PIL filter then
            return super(Image,self).filter(f)

    def correlation(self, other):
        """Compute the correlation between two, single-channel, grayscale input images.
        The second image must be smaller than the first.
        :param other: the Image we're looking for
        """
        from scipy import signal
        input = self.ndarray()
        match = other.ndarray()
        c=signal.correlate2d(input,match)
        return Image(c)

    def scale(self,s):
        """resize image by factor s

        :param s: (sx, sy) tuple of float scaling factor, or scalar s=sx=sy
        :return: Image scaled
        """
        try:
            s[1]
        except:
            s=[s,s]
        w,h=self.size
        return self.resize((int(w*s[0]+0.5),int(h*s[1]+0.5)))

    @adapt_rgb
    def shift(self,dx,dy,**kwargs):
        from scipy.ndimage.interpolation import shift as shift2
        return Image(shift2(self,(dy,dx),**kwargs))

    def expand(self,size,ox=None,oy=None):
        """
        :return: image in larger canvas size, pasted at ox,oy
        """
        im = Image(mode=self.mode, size=size)
        (w,h)=self.size
        if ox is None:
            ox=(size[0]-w)//2
        elif ox<0:
            ox=size[0]-w+ox
        if oy is None:
            oy=(size[1]-h)//2
        elif oy<0:
            oy=size[1]-h+oy
        if math2.is_integer(ox) and math2.is_integer(oy):
            im.paste(self, tuple(map(math2.rint,(ox,oy,ox+w,oy+h))))
        elif ox>=0 and oy>=0:
            im.paste(self, (0,0,w,h))
            im=im.shift(ox,oy)
        else:
            raise NotImplemented #TODO; something for negative offsets...
        return im

    # @adapt_rgb
    def compose(self,other,a=0.5,b=0.5):
        """compose new image from a*self + b*other
        """
        if self:
            d1=np.array(self,dtype=np.float)
        else:
            d1=None
        if other:
            d2=np.asarray(other,dtype=np.float)
        else:
            d2=None
        if d1 is not None:
            if d2 is not None:
                return Image(a*d1+b*d2)
            else:
                return Image(a*d1)
        else:
            return Image(b*d2)

    def add(self,other,pos=(0,0),alpha=1):
        """ simply adds other image at px,py (subbixel) coordinates
        :warning: result is normalized in case of overflow
        """
        #TOD: use http://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
        px,py=pos
        assert px>=0 and py>=0
        im1,im2=self,other
        size=(max(im1.size[0],int(im2.size[0]+px+0.999)),
              max(im1.size[1],int(im2.size[1]+py+0.999)))
        if not im1.mode: #empty image
            im1.mode=im2.mode
        im1=im1.expand(size,0,0)
        im2=im2.expand(size,px,py)
        return im1.compose(im2,1,alpha)

    def __add__(self,other):
        return self.compose(other,1,1)

    def __sub__(self,other):
        return self.compose(other,1,-1)

    def draw(self,entity):
        from . import drawing, geom
        try: #iterable ?
            for e in entity:
                draw(e)
            return
        except:
            pass

        if isinstance(entity,geom.Circle):
            box=entity.bbox()
            box=(box.xmin, box.ymin, box.xmax, box.ymax)
            ImageDraw.Draw(self).ellipse(box, fill=255)
        else:
            raise NotImplemented
        return self

#from http://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil

def alpha_to_color(image, color=(255, 255, 255)):
    """Set all fully transparent pixels of an RGBA image to the specified color.
    This is a very simple solution that might leave over some ugly edges, due
    to semi-transparent areas. You should use alpha_composite_with color instead.

    Source: http://stackoverflow.com/a/9166671/284318

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    """
    x = np.array(image)
    r, g, b, a = np.rollaxis(x, axis=-1)
    r[a == 0] = color[0]
    g[a == 0] = color[1]
    b[a == 0] = color[2]
    x = np.dstack([r, g, b, a])
    return Image.fromarray(x, 'RGBA')


def alpha_composite(front, back):
    """Alpha composite two RGBA images.

    Source: http://stackoverflow.com/a/9166671/284318

    Keyword Arguments:
    front -- PIL RGBA Image object
    back -- PIL RGBA Image object

    The algorithm comes from http://en.wikipedia.org/wiki/Alpha_compositing

    """
    front = np.asarray(front)
    back = np.asarray(back)
    result = np.empty(front.shape, dtype=np.float)
    alpha = np.index_exp[:, :, 3:]
    rgb = np.index_exp[:, :, :3]
    falpha = front[alpha] / 255.0
    balpha = back[alpha] / 255.0
    result[alpha] = falpha + balpha * (1 - falpha)
    old_setting = np.seterr(invalid='ignore')
    result[rgb] = (front[rgb] * falpha + back[rgb] * balpha * (1 - falpha)) / result[alpha]
    np.seterr(**old_setting)
    result[alpha] *= 255
    np.clip(result, 0, 255)
    # astype('uint8') maps np.nan and np.inf to 0
    result = result.astype('uint8')
    result = Image.fromarray(result, 'RGBA')
    return result


def alpha_composite_with_color(image, color=(255, 255, 255)):
    """Alpha composite an RGBA image with a single color image of the
    specified color and the same size as the original image.

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    """
    back = Image.new('RGBA', size=image.size, color=color + (255,))
    return alpha_composite(image, back)


def pure_pil_alpha_to_color_v1(image, color=(255, 255, 255)):
    """Alpha composite an RGBA Image with a specified color.

    NOTE: This version is much slower than the
    alpha_composite_with_color solution. Use it only if
    numpy is not available.

    Source: http://stackoverflow.com/a/9168169/284318

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    """
    def blend_value(back, front, a):
        return (front * a + back * (255 - a)) / 255

    def blend_rgba(back, front):
        result = [blend_value(back[i], front[i], front[3]) for i in (0, 1, 2)]
        return tuple(result + [255])

    im = image.copy()  # don't edit the reference directly
    p = im.load()  # load pixel array
    for y in range(im.size[1]):
        for x in range(im.size[0]):
            p[x, y] = blend_rgba(color + (255,), p[x, y])

    return im

def pure_pil_alpha_to_color_v2(image, color=(255, 255, 255)):
    """Alpha composite an RGBA Image with a specified color.

    Simpler, faster version than the solutions above.

    Source: http://stackoverflow.com/a/9459208/284318

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    """
    image.load()  # needed for split()
    background = Image.new('RGB', image.size, color)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return background

def disk(radius,antialias=PILImage.BICUBIC):
    size = (2*radius, 2*radius)
    im = Image(size=size)
    ImageDraw.Draw(im).ellipse((0, 0) + size, fill=255)
    return im
    #TODO: http://stackoverflow.com/questions/890051/how-do-i-generate-circular-thumbnails-with-pil

def fspecial(name,**kwargs):
    """mimics the Matlab image toolbox fspecial function
    http://www.mathworks.com/help/images/ref/fspecial.html?refresh=true
    """
    if name=='disk':
        return disk(kwargs.get('radius',5)) # 5 is default in Matlab
    raise NotImplemented

def _normalize(array,newmax=255,newmin=0):
    #http://stackoverflow.com/questions/7422204/intensity-normalization-of-image-using-pythonpil-speed-issues
    #warning : this normalizes each channel independently, so we don't use @adapt_rgb here
    if len(array.shape)==2 : #single channel
        n=1
        minval = array.min()
        maxval = array.max()
        array += newmin-minval
        if maxval is not None and minval != maxval:
            array *= (newmax/(maxval-minval)).astype(array.dtype)
    else:
        n=min(array.shape[2],3) #if RGBA, ignore A channel
        minval = array[:,:,0:n].min()
        maxval = array[:,:,0:n].max()
        for i in range(n):
            array[...,i] += newmin-minval
            if maxval is not None and minval != maxval:
                array[...,i] *= (newmax/(maxval-minval)).astype(array.dtype)
    return array

def read_pdf(filename,**kwargs):
    """ reads a bitmap graphics on a .pdf file
    only the first page is parsed
    """
    from pdfminer.pdfparser import PDFParser
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfinterp import PDFResourceManager
    from pdfminer.pdfinterp import PDFPageInterpreter
    from pdfminer.pdfdevice import PDFDevice
    # PDF's are fairly complex documents organized hierarchically
    # PDFMiner parses them using a stack and calls a "Device" to process entities
    # so here we define a Device that processes only "paths" one by one:


    class _Device(PDFDevice):
        def render_image(self, name, stream):
            import StringIO
            try:
                self.im=PILImage.open(StringIO.StringIO(stream.rawdata))
            except Exception as e:
                logging.error(e)


    #then all we have to do is to launch PDFMiner's parser on the file
    fp = open(filename, 'rb')
    parser = PDFParser(fp)
    document = PDFDocument(parser, fallback=False)
    rsrcmgr = PDFResourceManager()
    device = _Device(rsrcmgr)
    device.im=None #in case we can't find an image in file
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.create_pages(document):
        interpreter.process_page(page)
        break #handle one page only

    im=device.im

    if im is None: #it's maybe a drawing
        fig=Drawing(filename).draw(**kwargs)
        im=fig2img(fig)

    return im

def fig2img ( fig ):
    """
    Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    
    :param fig: matplotlib figure
    :return: PIL image
    """
    #http://www.icare.univ-lille1.fr/wiki/index.php/How_to_convert_a_matplotlib_figure_to_a_numpy_array_or_a_PIL_image

    fig.canvas.draw ( )

    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    w, h, _ = buf.shape
    return PILImage.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )



