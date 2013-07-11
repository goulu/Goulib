#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Rasters (simple) .dxf files to bitmap images
:requires: `dxfgrabber <http://pypi.python.org/pypi/dxfgrabber/>`_ and `pil <http://pypi.python.org/pypi/pil/>`_
"""
__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2013, Philippe Guglielmetti"
__credits__ = ['http://effbot.org/imagingbook/imagedraw.htm']
__license__ = "LGPL"

import StringIO, base64, logging, operator

import dxfgrabber
import Image, ImageDraw, ImageFont #PIL

from homcoord import *

def rint(x):return int(round(x))

class BBox:
    """bounding box"""
    def __init__(self,pt1=None,pt2=None):
        self._corner1=None
        self._corner2=None
        if pt1: self+=pt1
        if pt2: self+=pt2
        
    def __iadd__(self,pt):
        if isinstance(pt,BBox):
            self+=pt._corner1
            self+=pt._corner2
        else:
            if not self._corner1:
                self._corner1=pt
            else:
                self._corner1=Pt(map(min,zip(self._corner1.xy,pt.xy)))
            if not self._corner2:
                self._corner2=pt
            else:
                self._corner2=Pt(map(max,zip(self._corner2.xy,pt.xy)))
        return self
        
    def __repr__(self):
        return "BBox(%s,%s)"%(self._corner1,self._corner2)
    
    def __call__(self):
        """:return: list of flatten corners"""
        l=list(self._corner1.xy)
        l.extend(list(self._corner2.xy))
        return l
    
    def size(self):
        """:return: Pt with xy sizes"""
        return self._corner2-self._corner1
    
    def center(self):
        """:return: Pt center"""
        res=self._corner2+self._corner1
        return res/2
        
    def trans(self,trans):
        """
        :param trans: Xform
        :return: BBox = self transformed by trans
        """
        return BBox(trans(self._corner1),trans(self._corner2))

def cbox(c,r):
    """ bounding box of a circle
    :param c: Pt center
    :param r: float radius
    :return: BBox
    """
    rr=Pt(r,r)
    return BBox(c+rr,c-rr)

def Trans(scale=1, offset=[0,0], rotation=0):
    res=Xform([[scale,0,offset[0]],[0,scale,offset[1]],[0,0,1]])
    if rotation:
        res=Xrotate(rotation*pi/180.)*res
    return res

class DXF:
    def __init__(self, file, layers=None, ignore=[]):
        """reads a .dxf file
        :param file: string path to .dxf file to read
        :param layers: list or dictionary of layers to handle. Empty = all layers
        :param ignore: list of strings of entity types to ignore
        """
        self.dxf=dxfgrabber.readfile(file)
        self.layers=layers
        self.ignore=ignore
        
    def entities(self,ent=None):
        """iterator over dxf or block entities"""
        if not ent:
            ent=self.dxf.entities
        for e in ent:
            if self.layers and e.layer not in self.layers:
                continue
            elif e.dxftype in self.ignore:
                continue
            else: 
                yield e
        
    def bbox(self):
        """:return: :class:BBox dwg enclosing bounding box"""
        box=BBox()
        for e in self.entities():
            if e.dxftype=='LINE':
                box+=Pt(e.start[:2])
                box+=Pt(e.end[:2])
            elif e.dxftype == 'CIRCLE':
                box+=cbox(Pt(e.center[:2]),e.radius)
            elif e.dxftype == 'ARC':
                c=Pt(e.center[:2])
                a=e.endangle-e.startangle
                if a>0:
                    start=e.startangle
                else: #arc goes clockwise (step will be negative)
                    start=e.endangle
                n=rint(abs(a)/10.) # number of points each 10° approximately
                n=max(n,1)
                step=a/n #angle between 2 points, might be negative
                for i in range(n+1):
                    box+=c.radial(e.radius,radians(start+i*step))
            elif e.dxftype=='POLYLINE':
                for v in e.vertices:
                    box+=Pt(v.location[:2])
            elif e.dxftype=='BLOCK': 
                pass #TODO ...
            elif e.dxftype in ['TEXT','INSERT']:
                box+=Pt(e.insert[:2])
            else:
                logging.warning('Unknown entity %s'%e)
        return box
    
    def _draw(self,draw,entities,trans,pen="black"):
        for e in entities:
            if e.dxftype=='LINE':
                b=list(trans(Pt(e.start[:2])).xy)
                b.extend(list(trans(Pt(e.end[:2])).xy))
                draw.line(b,fill=pen)
            elif e.dxftype=='CIRCLE':
                b=cbox(Pt(e.center[:2]),e.radius)
                b=b.trans(trans)
                draw.ellipse(b(),outline=pen)
            elif e.dxftype=='ARC':
                c=Pt(e.center[:2])
                b=cbox(c,e.radius)
                b=b.trans(trans)
                b=map(rint,b())
                startangle=degrees(trans.angle(radians(e.startangle)))
                endangle=degrees(trans.angle(radians(e.endangle)))
                startangle,endangle=endangle,startangle #swap start/end because of Y symmetry
                draw.arc(b,int(startangle),int(endangle),fill=pen)
            elif e.dxftype=='POLYLINE':
                b=[]
                for v in e.vertices:
                    b.extend(list(trans(Pt(v.location[:2])).xy))
                draw.line(b,fill=pen)
            elif e.dxftype=='TEXT':
                h=e.height*trans.mag()
                pt=Pt(e.insert[:2])+Pt(0,e.height) #ACAD places texts by top left point...
                font=None
                try:
                    font = ImageFont.truetype("c:/windows/fonts/Courier New.ttf", rint(h))
                except:
                    pass
                if not font:
                    h=h*1.4 #magic factor ...
                    fh=[8,10,12,14,16,18,20,22,24,26,28,30,36,40,48,60]
                    i,h=min(enumerate(fh), key=lambda x: abs(x[1]-h)) #http://stackoverflow.com/questions/9706041/finding-index-of-an-item-closest-to-the-value-in-a-list-thats-not-entirely-sort
                    import os
                    path=os.path.realpath(__file__)
                    path=os.path.dirname(path)
                    font = ImageFont.load(path+'\\base_pil\\72\\Courier New_%s_72.pil'%h)
                draw.text(trans(pt).xy,e.text,font=font,fill=pen) 
            elif e.dxftype=='INSERT': 
                t2=Trans(1,e.insert,e.rotation).compose(trans)
                    
                self._draw(draw,self.entities(self.dxf.blocks[e.name]._entities),t2,pen)
            elif e.dxftype=='BLOCK': 
                pass # block definition is automatically stored in dxf.blocks dictionary
            else:
                logging.warning('Unknown entity %s'%e)
    
    def img(self,size=[256,256],back="white",pen="black",border=5,antialias=1):
        """:result: :class:`PIL:Image` rasterized image"""
        box=self.bbox()
        from Goulib.math2 import product
        if not product(box.size().xy): # either x or y ==0
            return None
        
        s=map(operator.div,[float(x-border)*antialias if x else 1E9 for x in size ],box.size().xy)
        trans=Trans(scale=min(s))
        size=trans(box.size())+Pt(2*antialias*border,2*antialias*border) #add borders as an offset
        offset=size/2-trans(box.center()) #offset in pixel coordinates
        trans=trans*Trans(offset=offset.xy)
        trans=trans*Xscale(1,-1) #invert y axis
        trans=trans*Xlate(0,size.y) #origin is lower left corner
        
        img = Image.new("RGB", map(rint,size.xy), back)
        self._draw(ImageDraw.Draw(img), self.entities(), trans, pen)
        if antialias>1:
            size=size/antialias
            img=img.resize(map(rint,size.xy), Image.ANTIALIAS)
        return img

def img2base64(img,fmt='PNG'):
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
    
if __name__ == '__main__':
    dxf=DXF("..\\tests\\FERRO_01.DXF")
    img=dxf.img(size=[1280,None],border=50)
    print img2base64(img)
    img.save('..\\tests\\out.png')