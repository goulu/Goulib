#!/usr/bin/env python
# coding: utf8
"""
color conversion in various colorspaces and palettes
"""

from __future__ import division #"true division" everywhere

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2012-, Philippe Guglielmetti"
__license__ = "LGPL"
__credits__ = ['Colormath https://pypi.python.org/pypi/colormath/',
               'Bruno Nuttens Pantone color table http://blog.brunonuttens.com/206-conversion-couleurs-pantone-lab-rvb-hexa-liste-sql-csv/',
               ]

#get https://pypi.python.org/pypi/colormath/ if you need more

import six, os, sys, logging
import numpy as np

from collections import OrderedDict
from Goulib import math2, itertools2

# color conversion

#redefine some converters in current module to build converters dict below
import skimage.color as skcolor

import matplotlib.colors as mplcolors

def rgb2hex(c,illuminant='ignore'):
    return mplcolors.rgb2hex(c)

def hex2rgb(c,illuminant='ignore'):
    return mplcolors.hex2color(c)

def rgb2cmyk(rgb,**kwargs):
    """
    :param rgb: 3-tuple of floats of red,green,blue in [0..1] range
    :return: 4-tuple of floats (cyan, magenta, yellow, black) in [0..1] range
    """
    # http://stackoverflow.com/questions/14088375/how-can-i-convert-rgb-to-cmyk-and-vice-versa-in-python

    r,g,b=rgb

    c = 1 - r
    m = 1 - g
    y = 1 - b

    k = min(c, m, y)
    if k==1:
        return (0,0,0,1)
    c = (c - k) / (1 - k)
    m = (m - k) / (1 - k)
    y = (y - k) / (1 - k)
    return (c,m,y,k)

def cmyk2rgb(cmyk,**kwargs):
    """
    :param cmyk: 4-tuple of floats (cyan, magenta, yellow, black) in [0..1] range
    :result: 3-tuple of floats (red,green,blue) 
    warning : rgb is out the [0..1] range for some cmyk
    
    """
    c,m,y,k=cmyk
    w=1-k
    return ((1-c)*w, (1-m)*w, (1-y)*w)

def xyz2xyy(xyz,**kwargs):
    """
    Convert from XYZ to xyY
    
    Based on formula from http://brucelindbloom.com/Eqn_XYZ_to_xyY.html
    
    Implementation Notes:
    1. Watch out for black, where X = Y = Z = 0. In that case, x and y are set 
       to the chromaticity coordinates of the reference whitepoint.
    2. The output Y value is in the nominal range [0.0, Y[XYZ]].
    
    """
    s=sum(xyz)
    if s == 0:
        # We can't check for X == Y == Z == 0 because they may actually add up
        # to 0, thus resulting in ZeroDivisionError later
        x, y, _ = xyz2xyy(color['white'].xyz)
        return (x, y, 0.0)
    return (xyz[0]/s, xyz[1]/s, xyz[1])

def xyy2xyz(xyY,**kwargs):
    """
    Convert from xyY to XYZ to
    
    Based on formula from http://brucelindbloom.com/Eqn_xyY_to_XYZ.html
    
    Implementation Notes:
    
    1. Watch out for the case where y = 0.
       In that case, you may want to set X = Y = Z = 0.
    2. The output XYZ values are in the nominal range [0.0, 1.0].
    
    """
    x,y,Y=xyY
    if y==0:
        return (0,0,0)
    X=x*Y/y
    Z=(1-x-y)*Y/y
    return (X,Y,Z)
 
# skimage.color has several useful color conversion routines, but for images
# so here is a generic adapter that allows to call them with colors

def _skadapt(f,**kwargs):
    def adapted(arr,**kwargs):
        arr = np.asanyarray(arr)
        if arr.ndim ==1:
            a=arr.reshape(1,1,arr.shape[-1])
            try:
                res=f(a,**kwargs)
            except TypeError: #unsupported params. retry without
                res=f(a)
            return res.reshape(arr.shape[-1])
        else:
            return f(arr,**kwargs)
    return adapted

#supported colorspaces. need more ? just add it :-)
colorspaces=(
    'CMYK',
    'XYZ',
    'xyY', #for CIE Chromaticity plots
    'Lab',
    'Luv',
    'HSV',
    'RGB',
    'HEX',
    )

#build a graph of available converters
#as in https://github.com/gtaylor/python-colormath

from .graph import DiGraph
converters=DiGraph(multi=False) # a nx.DiGraph() would suffice, but my DiGraph are better
for source in colorspaces:
    for target in colorspaces:
        key=(source.lower(),target.lower())
        if key[0]==key[1]:
            continue
        else:
            convname='%s2%s'%key
            converter = getattr(sys.modules[__name__], convname,None)
            if converter is None:
                converter=getattr(skcolor, convname,None)
                if converter: #adapt it:
                    converter=_skadapt(converter)
        if converter:
            converters.add_edge(key[0],key[1],{'f':converter})

def convert(color,source,target):
    """convert a color between colorspaces,
    eventually using intermediary steps
    """
    source,target=source.lower(),target.lower()
    if source==target: return color
    path=converters.shortest_path(source, target)
    for u,v in itertools2.pairwise(path):
        color=converters[u][v][0]['f'](color)
    return color #isn't it beautiful ?

class Color(object):
    """A color with math operations and conversions
    Color is immutable (._values caches representations)
    """
    def __init__(self, value, space='RGB', name=None, illuminant='D65'):
        """constructor
        :param value: string color name, hex string, or values tuple
        :param space: string defining the color space of value
        :param name: string for color name
        :param illuminant: string in {“A”, “D50”, “D55”, “D65”, “D75”, “E”} 
            * D65 is used by default in skimage, see http://scikit-image.org/docs/dev/api/skimage.color.html
            * D50 is used in Pantone and other graphic arts
        """
        self._name=name
        self.illuminant=illuminant
        space=space.lower() # for easier conversions

        if isinstance(value,Color): #copy constructor
            self._copy_from_(value)
            return
        if isinstance(value, six.string_types):
            if value in pantone:
                self._copy_from_(pantone[value])
                return
            value=value.lower()
            if value in color:
                self._copy_from_(color[value])
                return
            elif len(value)==7 and value[0]=='#':
                if value in color_lookup:
                    self._copy_from_(color_lookup[value])
                    return
                else:
                    space='hex'
            else:
                raise(ValueError("Couldn't create Color(%s,%s)"%(value,space)))

        self.space=space # "native" space in which the color was created

        if space=='rgb':
            if max(value)>1:
                value=tuple(_/255. for _ in value)
            value=math2.sat(value[:3],0,1) # rgb only, not whiter than white...
        if space!='hex': # force to floats
            value=tuple(float(_) for _ in value)
        self._values=OrderedDict() # so native space is always first
        self._values[space]=value 


    def _copy_from_(self,c):
        self.space=c.space
        self.illuminant=c.illuminant
        self._name=c._name
        self._values=c._values

    @property
    def name(self):
        if self._name is None:
            if self.hex in color_lookup:
                self._name=color_lookup[self.hex].name
            else:
                self._name='~'+nearest_color(self).name
        return self._name

    def convert(self, target, **kwargs):
        """ 
        :param target: str of desired colorspace, or none for default
        :return: color in target colorspace
        """
        import networkx as nx
        target=target.lower() if target else self.space
        if target not in self._values:
            try:
                path=converters.shortest_path(self.space, target)
            except nx.exception.NetworkXNoPath:
                raise NotImplementedError(
                    'no conversion between %s and %s color spaces'
                    %(self.space, target)
                )
            kwargs['illuminant']=self.illuminant # to avoid incoherent cached values
            for u,v in itertools2.pairwise(path):
                if v not in self._values:
                    edge=converters[u][v][0]
                    c=edge['f'](self._values[u],**kwargs)

                    if itertools2.isiterable(c): #but not a string
                        c=tuple(map(float,c))
                    self._values[v]=c
        return self._values[target]
    
    def str(self,mode=None):
        res=self.convert(mode)
        if not isinstance(res, six.string_types):
            res=', '.join(map(math2.format,res))
        return res
    
    @property
    def native(self): return self.convert(None)

    @property
    def rgb(self): return self.convert('rgb')

    @property
    def hex(self): return self.convert('hex')

    @property
    def lab(self): return self.convert('lab')
    
    @property
    def luv(self): return self.convert('Luv')

    @property
    def cmyk(self): return self.convert('cmyk')

    @property
    def hsv(self): return self.convert('hsv')
    
    @property
    def xyz(self): return self.convert('xyz')
    
    @property
    def xyY(self): return self.convert('xyY')

    def __hash__(self):
        return hash(self.hex)

    def __repr__(self):
        return "Color('%s')"%(self.name)

    def _repr_html_(self):
        return '<span style="color:%s">%s</span>'%(self.hex,self.name)
    
    def compose(self,other,f,mode='rgb'):
        """ compose colors in given mode
        """
        if not isinstance(other, Color):
            other=Color(other,mode)
        res=f(self.convert(mode),other.convert(mode))
        min=-1 if mode=='lab' else 0
        max=1
        res=[math2.sat(_,min,max) for _ in res]
        return res

    def __add__(self,other):
        from Goulib.image import Image
        if isinstance(other, Image):
            return Image(size=other.size,color=self.native,mode=self.space)+other
        return Color(self.compose(other,math2.vecadd),illuminant=self.illuminant)
        
    def __radd__(self,other):
        """only to allow sum(colors) easily"""
        assert other==0
        return self

    def __sub__(self,other):
        from Goulib.image import Image
        if isinstance(other, Image):
            mode=other.mode
            return Image(size=other.size,color=self.convert(mode),mode=mode)-other
        return Color(self.compose(other,math2.vecsub),illuminant=self.illuminant)
        
    def __mul__(self,factor):
        if factor<0:
            return (-self)*(-factor)
        l,a,b=self.lab
        l*=factor
        res=Color((l,a,b),'lab',illuminant=self.illuminant)
        return res
    
    def __neg__(self):
        """ complementary color"""
        return color['white']-self
    
    def deltaE(self,other):
        """color difference according to CIEDE2000
        https://en.wikipedia.org/wiki/Color_difference
        """
        assert(self.illuminant==other.illuminant)
        return skcolor.deltaE_ciede2000(self.lab, other.lab)
    
    def isclose(self,other,abs_tol=1):
        """
        http://zschuessler.github.io/DeltaE/learn/
        <= 1.0    Not perceptible by human eyes.
        1 - 2    Perceptible through close observation.
        2 - 10    Perceptible at a glance.
        11 - 49    Colors are more similar than opposite
        100    Colors are exact opposite
        """
        dE=self.deltaE(other)
        if dE<=abs_tol:
            return True
        else:
            return False
    
    def __eq__(self,other):
        other=Color(other)
        if self.space==other.space:
            if self.native==other.native:
                return True
        return self.isclose(other,1) #difference not perceptible to human eye

class Palette(OrderedDict):
    """dict of Colors indexed by anything"""
    def __init__(self, data=[], keys=256):
        super(Palette, self).__init__() #mandatory http://stackoverflow.com/questions/11174702/how-to-subclass-an-ordereddict
        if data:
            self.update(data,keys)
        
    def update(self,data,keys=256):
        """updates the dictionary with new colors
        :param data: colors to add
        :param keys: keys to use in dict, or int to discretize the Colormap
        """
        from matplotlib.colors import Colormap
        if isinstance(data, Colormap):
            for i in range(keys):
                self[i]=Color(data(i/(keys-1))) #RGB 
        elif isinstance(keys, six.integer_types): 
            for i,v in itertools2.enumerates(data):
                    self[i]=Color(v) # v.space of RGB
        else:
            for i,v in six.moves.zip(keys,data):
                self[i]=Color(v) # v.space of RGB
                
        return self

        
    def index(self,c,dE=5):
        """
        :return: key of c or nearest color, None if distance is larger than deltaE
        """
        c=Color(c)
        k,v=itertools2.index_min(self,key=lambda c2:deltaE(c,c2))
        if k is None or (dE>0 and deltaE(c,v) > dE):
            return None
        return k
    
    def __repr__(self):
        return '%s of %d colors' % (self.__class__.__name__,len(self))
    
    def _repr_html_(self):
        def tooltip(k):
            c=self[k]
            res='[%s] %s (%s)\n'%(k,c.name,c.illuminant)
            return res+'\n'.join('%s = %s'%(k,c.str(k)) for k in c._values)
        
        mode='inline' if len(self)>256 else 'flex'
    
        labels=(color['black'],color['white']) #possible colors for labels
        res='<div style="display:%s; width:100%%;">'%mode
        style='display:%s-block; min-width: 1px; ' %mode
        style+=' flex-basis: 90%%;'
        style+=' background:%s; color:%s;'
        cell='<div style="'+style+'" title="%s">&nbsp;</div>'
        for k in self:
            c=self[k]
            # c2=nearest_color(c,labels,opt=max) #chose the label color with max difference to pantone color
            res+= cell % (c.hex, c.hex, tooltip(k))
        return res+'</div>'
    
    def patches(self,wide=64,size=(16,16)):
        """Image made of each palette color
        """
        from Goulib.image import Image
        n=len(self)
        data=itertools2.reshape(range(n),(n//wide,wide))
        res=Image(data,'P',palette=self)
        res=res.scale(size)
        return res
    
    @property
    def pil(self):
        """
        :return: a sequence of integers, or a string containing a binary 
        representation of the palette.
        In both cases, the palette contents should be ordered (r, g, b, r, g, b, …). 
        The palette can contain up to 768 entries (3*256). 
        If a shorter palette is given, it is padded with zeros.
        #http://effbot.org/zone/creating-palette-images.htm
        """
        res=[]
        for c in self.values():
            r,g,b=c.rgb
            res.append(math2.rint(r*255))
            res.append(math2.rint(g*255))
            res.append(math2.rint(b*255))
        return res
    
    def sorted(self,key=lambda c:c[1].lab[0]):
        # http://stackoverflow.com/questions/8031418/how-to-sort-ordereddict-of-ordereddict-python
        return Palette(dict(sorted(self.items(), key=key)))
    
def ColorTable(colors,key=None,width=10):
    from Goulib.table import Table, Cell
    from Goulib.itertools2 import reshape
    
    def tooltip(c):
        return '\n'.join('%s = %s'%(k,v) for k,v in c._values.items())

    labels=(color['black'],color['white']) #possible colors for labels
    t=[]
    colors=colors.values()
    if key:
        colors=list(colors)
        colors.sort(key=key)
    for c in colors:
        c2=nearest_color(c,labels,opt=max) #chose the label color with max difference to pantone color
        s='<span title="%s" style="color:%s">%s</span>'%(tooltip(c),c2.hex,c.name)
        t.append(Cell(s,style={'background-color':c.hex}))
    return Table(reshape(t,(0,width)))

# dictionaries of standardized colors

from Goulib.table import Table

path=os.path.dirname(os.path.abspath(__file__))

# http://blog.brunonuttens.com/206-conversion-couleurs-pantone-lab-rvb-hexa-liste-sql-csv/
table=Table(path+'/colors.csv')
table.applyf('hex',lambda x:x.lower())
table=table.groupby('System')

color=Palette() #dict of HTML / matplotlib colors, which seem to be the same
color_lookup=Palette() # reverse color dict indexed by hex
pantone=Palette() #dict of pantone colors

# http://www.w3schools.com/colors/colors_names.asp

for c in table['websafe'].asdict():
    id=c['name'].lower()
    hex=c['hex']
    c=Color(hex,name=id,illuminant='D65')
    color[id]=c
    color_lookup[c.hex]=c

for c in table['Pantone'].asdict():
    id=c['name']
    pantone[id]=Color((c['L'],c['a'],c['b']),space='Lab',name=id,illuminant='D50')
    # pantones are defined with D50 illuminant

acadcolors=[None]*256 #table of Autocad indexed colors
for c in table['autocad'].asdict():
    id=c['name']
    acadcolors[id]=Color(c['hex'],name=id,illuminant='D65') #color name is a 0..255 number


def color_to_aci(x, nearest=True):
    """
    :return: int Autocad Color Index of color x
    """
    if x is None:
        return -1
    x=Color(x)
    try:
        return acadcolors.index(x)
    except ValueError:
        pass
    if nearest:
        return nearest_color(x,acadcolors).name # name = int id
    return -1


def aci_to_color(x, block_color=None, layer_color=None):
    if x==0: return block_color
    if x==256: return layer_color
    return acadcolors[x].hex

def deltaE(c1,c2):
    # http://scikit-image.org/docs/dev/api/skimage.color.html#skimage.color.deltaE_ciede2000
    if not isinstance(c1, Color):
        c1=Color(c1)
    if not isinstance(c2, Color):
        c2=Color(c2)
    return skcolor.deltaE_ciede2000(c1.lab, c2.lab)

def nearest_color(c,l=None, opt=min, comp=deltaE):
    """
    :param x: Color
    :param l: list or dict of Color, color by default
    :param opt: with opt=max you can find the most different color ...
    :return: nearest Color of x in  l
    """
    if not isinstance(c, Color):
        c=Color(c)
    l=l or color
    if isinstance(l,dict):
        l=l.values()
    return opt(l,key=lambda c2:comp(c,c2))

#http://stackoverflow.com/questions/876853/generating-color-ranges-in-python

def color_range(n,start,end,space='hsv'):
    """:param n: int number of colors to generate
    :param start: string hex color or color name
    :param end: string hex color or color name
    :result: list of n Color interpolated between start and end, included
    """

    from .itertools2 import linspace
    start=Color(start).convert(space)
    end=Color(end).convert(space)
    return [Color(v, space=space) for v in linspace(start,end,n)]
