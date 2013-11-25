#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Rasters (simple) .dxf files to bitmap images
:requires: `dxfgrabber <http://pypi.python.org/pypi/dxfgrabber/>`_ and `pil <http://pypi.python.org/pypi/pil/>`_
"""
__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2013, Philippe Guglielmetti"
__credits__ = ['http://effbot.org/imagingbook/imagedraw.htm', 'http://images.autodesk.com/adsk/files/acad_dxf0.pdf']
__license__ = "LGPL"

import io, StringIO, base64, logging, operator
from math import  radians, degrees
import dxfgrabber
import geom

Pt = geom.Point2  # alias

def Trans(scale=1, offset=None, rotation=None):
    """
    :return: :class:Matrix3 of generalized scale+offset+rotation
    """
    res = geom.Matrix3()
    if rotation:
        res = res.rotate(radians(rotation))
    if scale != 1:
        res = res.scale(scale)
    if offset:
        res = res.translate(offset)
    return res

def rint(x):return int(round(x))

class BBox:
    """bounding box"""
    def __init__(self, pt1=None, pt2=None):
        """
        :param pt1: :class:`Pt` first corner (any)
        :param pt2: :class:`Pt` opposite corner (any)
        """
        self._pt1 = None
        self._pt2 = None
        if pt1: self +=pt1
        if pt2: self +=pt2
    
    @property
    def xmin(self): return self._pt1.x
    
    @property
    def ymin(self): return self._pt1.y
    
    @property
    def xmax(self): return self._pt2.x
    
    @property
    def ymax(self): return self._pt2.y
        
    def __iadd__(self, pt):
        """
        enlarge box if required to contain specified point
        :param pt1: :class:`Pt` point to add
        """
        if not pt:
            return self
        if isinstance(pt, BBox):
            self +=pt._pt1
            self +=pt._pt2
        elif isinstance(pt,Pt):
            self+= pt.xy
        else:
            if not self._pt1:
                self._pt1 = Pt(pt)
            else:
                p=map(min, zip(self._pt1.xy, pt))
                self._pt1 = Pt(p)
            if not self._pt2:
                self._pt2 = Pt(pt)
            else:
                p=map(max, zip(self._pt2.xy, pt))
                self._pt2 = Pt(p)
        return self
    
    def __add__(self,other):
        res=BBox()
        res+=self
        res+=other
        return res
        
    def __repr__(self):
        return "%s(%s,%s)" % (self.__class__.__name__,self._pt1, self._pt2)
    
    def __call__(self):
        """:return: list of flatten corners"""
        l = list(self._pt1.xy)
        l.extend(list(self._pt2.xy))
        return l
    
    def size(self):
        """:return: Pt with xy sizes"""
        try:
            return self._pt2 - self._pt1
        except:
            return geom.Vector2(0, 0)
    
    def center(self):
        """:return: Pt center"""
        res = self._pt2 + self._pt1
        return res / 2
        
    def trans(self, trans):
        """
        :param trans: Xform
        :return: BBox = self transformed by trans
        """
        res = BBox(trans(self._pt1), trans(self._pt2))
        # add 2 more corners as they matter if we rotate the box
        res += trans(Pt(self._pt1.x, self._pt2.y))
        res += trans(Pt(self._pt2.x, self._pt1.y))
        return res

def cbox(c, r):
    """ bounding box of a circle
    :param c: Pt center
    :param r: float radius
    :return: BBox
    """
    rr = Pt(r, r)
    return BBox(c + rr, c - rr)

# http://sub-atomic.com/~moses/acadcolors.html
acadcolors = ['black','red','yellow','green','cyan','blue','magenta','white']

from dxfgrabber.drawing import Drawing
class DXF(Drawing):
    def __init__(self, filename, options=None):
        """reads a .dxf file
        :param filename: string path to .dxf file to read
        :param options: passed to :class:`~dxfgrabber.drawing.Drawing`constructor
        """
        # code copied from dxfgrabber.drawing as Drawing takes a stream as input
        def get_encoding():
            with io.open(filename) as fp:
                info = dxfgrabber.tags.dxfinfo(fp)
            return info.encoding
        
        with io.open(filename, encoding=get_encoding(), errors='strict') as fp:
            super(DXF, self).__init__(fp, options)
        self.filename = filename
        
    def iter(self, ent=None, layers=None, only=[], ignore=[], trans=None, recurse=False):
        """iterator over dxf or block entities"""
        if ent is None:
            ent = self.entities
        if not trans:
                trans = Trans()  # identity
        for e in ent:
            if layers and e.layer not in layers:
                continue
            if only:
                if e.dxftype in only:
                    yield e, trans
                else:
                    continue
            elif e.dxftype in ignore:
                continue
            elif recurse and e.dxftype == 'INSERT':
                t2 = trans*Trans(1, e.insert[:2], e.rotation)
                for e2, t3 in self.iter(self.blocks[e.name]._entities, layers=None, ignore=ignore, trans=t2, recurse=recurse):
                    yield e2, t3
            else: 
                yield e, trans
        
    def bbox(self, layers=None, ignore=[]):
        """
        :param layers: list or dictionary of layers to draw. None = all layers
        :param ignore: list of strings of entity types to ignore
        :return: :class:`BBox` bounding box of corresponding entities"""
        box = BBox()
        for e, trans in self.iter(layers=layers, ignore=ignore, recurse=True):
            if e.dxftype == 'LINE':
                box += trans(Pt(e.start[:2]))
                box += trans(Pt(e.end[:2]))
            elif e.dxftype == 'CIRCLE':
                box += cbox(trans(Pt(e.center[:2])), e.radius)
            elif e.dxftype == 'ARC':
                c = Pt(e.center[:2])
                a = e.endangle - e.startangle
                if a > 0:
                    start = e.startangle
                else:  # arc goes clockwise (step will be negative)
                    start = e.endangle
                n = rint(abs(a) / 10.)  # number of points each 10 degrees approximately
                n = max(n, 1)
                step = a / n  # angle between 2 points, might be negative
                for i in range(n + 1):
                    box += trans(c + geom.Polar(e.radius, radians(start + i * step)))
            elif e.dxftype == 'POLYLINE':
                for v in e.vertices:
                    box += trans(Pt(v.location[:2]))
            elif e.dxftype == 'SPLINE':
                for v in e.controlpoints:
                    box += trans(Pt(v.location[:2]))
            elif e.dxftype == 'BLOCK': 
                pass
            elif e.dxftype in ['TEXT', 'INSERT']:
                box += trans(Pt(e.insert[:2]))
            else:
                logging.warning('Unknown entity %s' % e)
        return box
    
    
    
    def img(self, size=[256, 256], border=5, box=None, layers=None, ignore=[], forcelayercolor=False, antialias=1,background='white'):
        """
        :param size: [x,y] max size of image in pixels. if one coord is None, the other one will be enforced
        :param border: int border width in pixels
        :param box: class:`BBox` bounding box. if None, box is calculated to contain all drawn entities
        :param layers: list or dictionary of layers to draw. None = all layers
        :param ignore: list of strings of entity types to ignore
        :result: :class:`PIL:Image` rasterized image
        """
        
        import Image, ImageDraw, ImageFont  # PIL
        
        def _draw(entities):
            for e, trans in entities:
                i = e.color  # color index
                if not i or forcelayercolor:
                    try:
                        i = self.layers[e.layer].color
                    except:
                        pass  # no layer
                pen = acadcolors[i % len(acadcolors)]
                if pen==background: pen=acadcolors[(len(acadcolors)-i) % len(acadcolors)]
                if e.dxftype == 'LINE':
                    b = list((trans * Pt(e.start[:2])).xy)
                    b.extend(list(trans(Pt(e.end[:2])).xy))
                    draw.line(b, fill=pen)
                elif e.dxftype == 'CIRCLE':
                    b = cbox(Pt(e.center[:2]), e.radius)
                    b = b.trans(trans)
                    draw.ellipse(b(), outline=pen)
                elif e.dxftype == 'ARC':
                    c = Pt(e.center[:2])
                    b = cbox(c, e.radius)
                    b = b.trans(trans)
                    b = map(rint, b())
                    startangle = degrees(trans.angle(radians(e.startangle)))
                    endangle = degrees(trans.angle(radians(e.endangle)))
                    startangle, endangle = endangle, startangle  # swap start/end because of Y symmetry
                    draw.arc(b, int(startangle), int(endangle), fill=pen)
                elif e.dxftype == 'POLYLINE':
                    b = []
                    for v in e.vertices:
                        b.extend(list(trans(Pt(v.location[:2])).xy))
                    draw.line(b, fill=pen)
                elif e.dxftype == 'SPLINE':
                    b = []
                    for v in e.controlpoints:
                        b.extend(list(trans(Pt(v[:2])).xy))
                    draw.line(b, fill=pen) # splines are drawn as lines for now...
                elif e.dxftype == 'TEXT':
                    h = e.height * trans.mag()  # [pixels]
                    if h < 4: continue  # too small
                    font = None
                    try:
                        font = ImageFont.truetype("c:/windows/fonts/Courier New.ttf", h)
                        print "font loaded !"
                    except:
                        pass
                    if not font:
                        h = h * 1.4  # magic factor (TODO : calculate DPI of image and conversions...)
                        fh = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 36, 40, 48, 60]
                        i, h = min(enumerate(fh), key=lambda x: abs(x[1] - h))  # http://stackoverflow.com/questions/9706041/finding-index-of-an-item-closest-to-the-value-in-a-list-thats-not-entirely-sort
                        import os
                        path = os.path.realpath(__file__)
                        path = os.path.dirname(path)
                        font = ImageFont.load(path + '\\base_pil\\72\\Courier New_%s_72.pil' % h)
                    pt = Pt(e.insert[0], e.insert[1] + e.height)  # ACAD places texts by top left point...
                    draw.text(trans(pt).xy, e.text, font=font, fill=pen)
                     
                elif e.dxftype == 'INSERT': 
                    t2 = trans * Trans(1, e.insert[:2], e.rotation)
                    _draw(self.iter(self.blocks[e.name]._entities, layers=None, ignore=ignore, trans=t2))
                elif e.dxftype == 'BLOCK': 
                    pass  # block definition is automatically stored in dxf.blocks dictionary
                else:
                    logging.warning('Unknown entity %s' % e)
        # img
        if not box:
            box = self.bbox(layers, ignore)
            
        from Goulib.math2 import product
        if not product(box.size().xy):  # either x or y ==0
            return None
        
        s = map(operator.div, [float(x - border) * antialias if x else 1E9 for x in size ], box.size().xy)
        trans = Trans(scale=min(s))
        size = trans * box.size() + Pt(2 * antialias * border, 2 * antialias * border)  # add borders as an offset
        offset = size / 2 - trans(box.center())  # offset in pixel coordinates
        trans = trans.translate(offset)
        trans = trans.scale(1, -1)  # invert y axis
        trans = trans.translate(0, size.y)  # origin is lower left corner
        
        img = Image.new("RGB", map(rint, size.xy), background)
        draw = ImageDraw.Draw(img)
        _draw(self.iter(layers=layers, ignore=ignore, trans=trans, recurse=False))
        if antialias > 1:
            size = size / antialias
            img = img.resize(map(rint, size.xy), Image.ANTIALIAS)
        return img
    
def factory(e,trans):
    """
    :param e: dxf.entity
    :param trans: geom.Matrix3 transform
    :return: geom entity
    """
    if not e: return
    if e.dxftype=='LINE':
        start=trans(Pt(e.start[:2]))
        end=trans(Pt(e.end[:2]))
        res=geom.Segment2(start,end)
    elif e.dxftype == 'ARC':
        c=Pt(e.center[:2])
        startangle=radians(e.startangle)
        start=c+geom.Polar(e.radius,startangle)
        endangle=radians(e.endangle)
        end=c+geom.Polar(e.radius,endangle)
        res=geom.Arc2(trans(c),trans(start),trans(end))
    elif e.dxftype == 'CIRCLE':
        c=Pt(e.center[:2])
        res=geom.Circle(trans(c), e.radius)
    else:
        logging.warning('unhandled entity type %s'%e.dxftype)
        return None
    res.color=acadcolors[e.color % len(acadcolors)]
    return res

def img2base64(img, fmt='PNG'):
    """
    :param img: :class:`PIL:Image`
    :result: string base64 encoded image content in specified format
    :see: http://stackoverflow.com/questions/14348442/django-how-do-i-display-a-pil-image-object-in-a-template
    """
    output = StringIO.StringIO()
    img.save(output, fmt)
    output.seek(0)
    output_s = output.read()
    return base64.b64encode(output_s)
    
