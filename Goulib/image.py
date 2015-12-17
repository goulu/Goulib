#!/usr/bin/env python
# coding: utf8
"""
image processing and conversion
"""
from __future__ import division #"true division" everywhere

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2015, Philippe Guglielmetti"
__credits__ = ['Brad Montgomery http://bradmontgomery.net']
__license__ = "LGPL"

"""
:requires:
* `PIL of Pillow <http://pypi.python.org/pypi/pillow/>`_
"""

from PIL import Image as PILImage
from PIL import ImagePalette, ImageOps

import numpy as np
import six, math

from . import math2

class Image(PILImage.Image):
    def __init__(self, data=None, **kwargs):
        """
        :param data: can be either:
        * None : creates an empty image
        * `PIL.Image` : makes a copy
        * string : path of image to load
        """
        if data is None:
            self._initfrom(PILImage.Image())
        elif isinstance(data,PILImage.Image):
            self._initfrom(data)
        elif isinstance(data,six.string_types): #assume a path
            self.open(data)
        else: #assume a np.ndarray
            # http://stackoverflow.com/questions/10965417/how-to-convert-numpy-array-to-pil-image-applying-matplotlib-colormap
            data = normalize(data) # normalize between 0-1
            colormap=kwargs.pop('colormap',None)
            if colormap:
                data=colormap(data)
            self._initfrom(PILImage.fromarray(np.uint8(data*255)))
            return
            """alternate, more complex solution:
            w,h = data.shape
             self._initfrom(PILImage.new('L', (w,h))) #Only grayscale images (PIL mode 'L') are supported.

            # make sure the array only contains values from 0-255
            # if not... fix them.
            if data.max() > 255 or data.min() < 0: 
                data = normalize(data) # normalize between 0-1
            if data.min() >= 0.0 and data.max() <= 1.0: # values are already between 0-1
                data = data * 255 # shift values to range 0-255
            data = data.flatten()
            array = []
            for val in data:
                if val is np.nan: val = 0 
                array.append(int(val)) # make sure they're all int's
            self.putdata(array)
            """
        
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

        import base64
        import cStringIO
        
        buffer = cStringIO.StringIO()
        self.save(buffer, format=fmt)
        return base64.b64encode(buffer.getvalue())
        
    def to_html(self):
        return r'<img src="data:image/png;base64,{0}">'.format(self.base64('PNG'))
    
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

def normalize(X, norm='max', axis=0, copy=True, positive=True):
    """Scale input vectors individually to unit norm (vector length).
    
    borrowed from http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
    (support for sparse matrices removed, default values changed, positive added)
    
    Parameters
    ----------
    X : array or scipy.sparse matrix with shape [n_samples, n_features]
        The data to normalize, element by element.
    norm : 'l1', 'l2', or 'max', optional ('max' by default)
        The norm to use to normalize each non zero sample (or each non-zero
        feature if axis is 0).
    axis : 0 or 1, optional (0 by default)
        axis used to normalize the data along. If 1, independently normalize
        each sample, otherwise (if 0) normalize each feature.
    copy : boolean, optional, default True
        set to False to perform inplace
    """
        
    if norm == 'l1':
        norms = np.abs(X).sum(axis=axis)
    elif norm == 'l2':
        norms = np.einsum('ij,ij->i', X, X) # http://ajcr.net/Basic-guide-to-einsum/
    elif norm == 'max':
        norms = np.max(X, axis=axis)
    else:
        raise ValueError("'%s' is not a supported norm" % norm)
            
       #norms = _handle_zeros_in_scale(norms)
        X /= norms[:, np.newaxis]

    if axis == 0:
        X = X.T

    return X
    

def correlation(input, match):
    """Compute the correlation between two, single-channel, grayscale input images.
    The second image must be smaller than the first.
    :param input: a PIL Image 
    :para, match: the PIL image we're looking for in input
    """
    input = pil2array(input)
    match = pil2array(match)
    
    assert match.shape < input.shape, "Match Template must be Smaller than the input"
    c = np.zeros(input.shape) # store the coefficients...
    mfmean = match.mean()
    iw, ih = input.shape # get input image width and height
    mw, mh = match.shape # get match image width and height
    

    for i in range(0, iw):
        for j in range(0, ih):

            # find the left, right, top 
            # and bottom of the sub-image
            if i-mw/2 <= 0:
                left = 0
            elif iw - i < mw:
                left = iw - mw
            else:
                left = i
                
            right = left + mw 

            if j - mh/2 <= 0:
                top = 0
            elif ih - j < mh:
                top = ih - mh
            else:
                top = j

            bottom = top + mh

            # take a slice of the input image as a sub image
            sub = input[left:right, top:bottom]
            assert sub.shape == match.shape, "SubImages must be same size!"
            localmean = sub.mean()
            temp = (sub - localmean) * (match - mfmean)
            s1 = temp.sum()
            temp = (sub - localmean) * (sub - localmean)
            s2 = temp.sum()
            temp = (match - mfmean) * (match - mfmean)
            s3 = temp.sum() 
            denom = s2*s3
            if denom == 0: 
                temp = 0
            else: 
                temp = s1 / math.sqrt(denom)
            
            c[i,j] = temp
    return array2pil(c)

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






 