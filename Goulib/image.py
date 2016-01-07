#!/usr/bin/env python
# coding: utf8
"""
image processing and conversion
:requires:
* `PIL of Pillow <http://pypi.python.org/pypi/pillow/>`_
"""
from __future__ import division #"true division" everywhere

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2015, Philippe Guglielmetti"
__credits__ = ['Brad Montgomery http://bradmontgomery.net']
__license__ = "LGPL"

from PIL import Image as PILImage
from PIL import ImagePalette, ImageOps

import numpy as np

import six, math, base64

from . import math2

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
            data=np.asarray(data, dtype=np.float32) #force ints to float
            if data.min()<0:
                data -=data.min()
            if data.max()>1:
                data *= 1./data.max() # normalize between 0-1
            colormap=kwargs.pop('colormap',None)
            if colormap:
                data=colormap(data)
            self._initfrom(PILImage.fromarray(np.uint8(data*255)))
            return
        
    def open(self,path):
        self.path=path
        im=PILImage.open(path)
        try:
            im.load()
            self.error=None
        except IOError as error: #truncated file ?
            self.error=error
        self._initfrom(im)
        return self
    
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
        upper,lower=upper%h,lower%h
        left,right=left%w,right%w
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
    
    def __hash__(self):
        return self.average_hash(8)
    
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
    
    def __lt__(self, other):
        return math2.mul(self.size) < math2.mul(other.size)
        
    
    def invert(self):
        # http://stackoverflow.com/questions/2498875/how-to-invert-colors-of-image-with-pil-python-imaging
        return ImageOps.invert(self)
    
    __neg__=__inv__=invert #aliases
    
    def grayscale(self):
        return self.convert("L")
    
    def filter(self,f):
        try: # scikit-image filter or similar ?
            return Image(f(self))
        except ValueError: #maybe because image has channels ? filter each one
            split=self.split()
            split=[Image(f(channel)) for channel in split]
            return PILImage.merge(self.mode, split)
        except:
            pass
        
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
        try:
            s[1]
        except:
            s=[s,s]
        w,h=self.size
        return self.resize((int(w*s[0]+0.5),int(h*s[1]+0.5)))
    
    def shift(self,dx,dy):
        from scipy.ndimage.interpolation import shift as shift2
        try:
            im=Image(shift2(self,(dy,dx)))
        except RuntimeError:
            split=self.split()
            split=[channel.shift(dx,dy) for channel in split]
            im=PILImage.merge(self.mode, split)
        return im
    
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
            im.paste(self, map(math2.rint,(ox,oy,ox+w,oy+h)))
        elif ox>=0 and oy>=0:
            im.paste(self, (0,0,w,h))
            im=im.shift(ox,oy)
        else:
            raise NotImplemented #TODO; something for negative offsets...
        return im
    
    def add(self,other,px=0,py=0,alpha=1):
        """ simply adds other image at px,py (subbixel) coordinates
        :warning: result is normalized in case of overflow
        """
        assert px>=0 and py>=0
        im1,im2=self,other
        size=(max(im1.size[0],int(im2.size[0]+px+0.999)),
              max(im1.size[1],int(im2.size[1]+py+0.999)))
        if not im1.mode: #empty image
            im1.mode=im2.mode            
        im1=im1.expand(size,0,0)
        im2=im2.expand(size,px,py)
        d1=np.asarray(im1)/255
        d2=np.asarray(im2)*alpha/255
        return Image(d1+d2)

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
    result = np.empty(front.shape, dtype='float')
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
    from skimage.morphology import disk as disk2
    return Image(disk2(radius))

def fspecial(name,**kwargs):
    """mimics the Matlab image toolbox fspecial function
    http://www.mathworks.com/help/images/ref/fspecial.html?refresh=true
    """
    if name=='disk':
        return disk(kwargs.get('radius',5)) # 5 is default in Matlab
    raise NotImplemented
    





 