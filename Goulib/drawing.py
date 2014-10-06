#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
handle vector graphics in .dxf, .svg and .pdf formats

:requires:
* `dxfgrabber <http://pypi.python.org/pypi/dxfgrabber/>`_ for dxf input
* `pdfminer.six <http://pypi.python.org/pypi/pdfminer.six/>`_ for pdf input
* `svg.path <http://pypi.python.org/pypisvg.path/>`_ for svg input
* `matplotlib <http://pypi.python.org/pypi/matplotlib/>`_ for bitmap + svg and pdf output
* `dxfwrite <http://pypi.python.org/pypi/dxfwrite/>`_ for dxf output

"""
from __future__ import division #"true division" everywhere

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2014, Philippe Guglielmetti"
__credits__ = ['http://effbot.org/imagingbook/imagedraw.htm', 'http://images.autodesk.com/adsk/files/acad_dxf0.pdf']
__license__ = "LGPL"

from math import  radians, degrees, tan, atan, pi, copysign
import logging, operator

import matplotlib
import os

if os.getenv('TRAVIS'): # are we running https://travis-ci.org/ automated tests ?
    matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend
# matplotlib.use('pgf') #for high quality pdf, but doesn't work for png, svg ...

import matplotlib.pyplot as plt

from .math2 import rint
from .geom import Point2, Vector2, Line2, Segment2, Arc2, Circle, Polar, Matrix3
from .itertools2 import split

from Goulib.geom import Geometry

# http://sub-atomic.com/~moses/acadcolors.html
# 'aqua' and 'lime' are the names of 'cyan' and 'green' inf goulib.colors
acadcolors = ['black','red','yellow','lime','aqua','blue','magenta','white']

def Trans(scale=1, offset=None, rotation=None):
    """
    :param scale: float or (scalex,scaley) tuple of scale factor
    :param offset: :class:`~geom.Vector3`
    :return: :class:`~geom.Matrix3` of generalized scale+offset+rotation
    """
    res = Matrix3()
    if rotation:
        res = res.rotate(radians(rotation))
    if scale != 1:
        res = res.scale(scale)
    if offset:
        res = res.translate(offset)

    return res

class BBox:
    """bounding box"""
    def __init__(self, pt1=None, pt2=None):
        """
        :param pt1: :class:`~geom.Point2` first corner (any)
        :param pt2: :class:`~geom.Point2` opposite corner (any)
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
        :param pt1: :class:`geom.Point2` point to add
        """
        if not pt:
            return self
        if isinstance(pt, BBox):
            self +=pt._pt1
            self +=pt._pt2
        elif isinstance(pt,Point2):
            self+= pt.xy
        else:
            if not self._pt1:
                self._pt1 = Point2(pt)
            else:
                p=list(map(min, list(zip(self._pt1.xy, pt))))
                self._pt1 = Point2(p)
            if not self._pt2:
                self._pt2 = Point2(pt)
            else:
                p=list(map(max, list(zip(self._pt2.xy, pt))))
                self._pt2 = Point2(p)
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
        l = list(self._pt1.xy)+list(self._pt2.xy)
        return l

    def size(self):
        """:return: :class:`geom.Vector2` with xy sizes"""
        try:
            return self._pt2 - self._pt1
        except:
            return Vector2(0, 0)

    def center(self):
        """:return: Pt center"""
        res = self._pt2 + self._pt1
        return res / 2

    def trans(self, trans):
        """
        :param trans: Xform
        :return: :class:`BBox` = self transformed by trans
        """
        res = BBox(trans(self._pt1), trans(self._pt2))
        # add 2 more corners as they matter if we rotate the box
        res += trans(Point2(self._pt1.x, self._pt2.y))
        res += trans(Point2(self._pt2.x, self._pt1.y))
        return res

def rpoint(pt,decimals=3): # rounds coordinates to number of decimals
    return Point2([round(x,decimals) for x in pt.xy])

class Entity(object):
    """Base class for all drawing entities"""

    color='black' #default color

    @property
    def start(self):
        return self.p

    @property
    def end(self):
        try:
            return self.p2
        except: #probably a Circle
            return self.p

    def __repr__(self):
        return '%s(%s,%s)' % (self.__class__.__name__,self.start,self.end)

    @property
    def center(self):
        return rpoint(self.bbox().center)

    def bbox(self):
        """
        :return: :class:`BBox` bounding box of Entity"""
        if isinstance(self,Segment2):
            return BBox(self.start,self.end)
        elif isinstance(self,Arc2):
            #TODO : improve
            rr = Vector2(self.r, self.r)
            return BBox(self.c - rr, self.c + rr)
        elif isinstance(self,Circle): #must be after Arc2 case since Arc2 is a Circle too
            rr = Vector2(self.r, self.r)
            return BBox(self.c - rr, self.c + rr)

        else:
            raise NotImplementedError()

    def isclosed(self):
        return self.end==self.start

    def isline(self):
        return isinstance(self,Line2) #or derived

    def isvertical(self,tol=0.01):
        return self.isline() and abs(self.start.x-self.end.x)<tol

    def ishorizontal(self,tol=0.01):
        return self.isline() and abs(self.start.y-self.end.y)<tol

    def patches(self, **kwargs):
        """:return: list of (a single) :class:`~matplotlib.patches.Patch` corresponding to entity"""
        import matplotlib.patches as patches
        from matplotlib.path import Path

        kwargs.setdefault('color',self.color)
        if isinstance(self,Segment2):
            path = Path((self.start.xy,self.end.xy),[Path.MOVETO, Path.LINETO])
            return [patches.PathPatch(path, **kwargs)]

        if isinstance(self,Arc2):
            theta1=degrees(self.a)
            theta2=degrees(self.b)
            if self.dir<1 : #swap
                theta1,theta2=theta2,theta1
            d=self.r*2
            return [patches.Arc(self.c.xy,d,d,theta1=theta1,theta2=theta2,**kwargs)]

        #entities below may be filled, so let's handle the color first
        if 'color' in kwargs: # color attribute refers to edgecolor for coherency
            kwargs.setdefault('edgecolor',kwargs.pop('color'))
            kwargs.setdefault('fill',False)
        if isinstance(self,Circle): #must be after isinstance(self,Arc2)
            return [patches.Circle(self.c.xy,self.r,**kwargs)]
        if isinstance(self,Spline):
            path = Path(self.xy, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
            return [patches.PathPatch(path, **kwargs)]
        raise NotImplementedError

    def _dxf_color(self):
        from .colors import color_lookup
        try:
            color=self.color
        except:
            return -1
        # if color is an #rrggb code, find the corresponding color name
        if color in color_lookup:
            color=color_lookup[color]
        # then find the name in acad color
        try:
            return acadcolors.index(color)
        except:
            pass
        return -1 #layer color

    def to_dxf(self,**attr):
        """:return: dxf entity"""
        import dxfwrite.entities as dxf

        color=self._dxf_color()
        if color>=0:
            attr['color']=color

        try:
            layer=self.layer
            attr['layer']=layer
        except:
            pass

        if isinstance(self,Segment2):
            return dxf.Line(start=self.start.xy,end=self.end.xy, **attr)
        elif isinstance(self,Arc2):
            center=self.c.xy
            v=Vector2(Point2(self.start)-self.c)
            startangle=degrees(v.angle())
            v=Vector2(Point2(self.end)-self.c)
            endangle=degrees(v.angle())
            if self.dir<0: #DXF handling of ARC direction is somewhat exotic ...
                startangle,endangle = 180-startangle,180-endangle #start/end were already swapped
                extrusion_direction=(0,0,-1)
                center=(-center[0],center[1]) #negative extrusion on Z causes Y axis symmetry... strange...
            else:
                extrusion_direction=None #default
            return dxf.Arc(radius=self.r, center=center, startangle=startangle, endangle=endangle, extrusion_direction=extrusion_direction, **attr)
        elif isinstance(self,Circle):
            return dxf.Circle(center=self.c.xy, radius=self.r, **attr)
        elif isinstance(self,Spline):
            from dxfwrite import curves as dxf
            spline=dxf.Bezier(**attr)
            spline.start(self.start.xy,(self.p[1]-self.start).xy)
            spline.append(self.end.xy,(self.p[2]-self.end).xy)
            return spline
            return dxf.Spline(self.xy, **attr) # builds a POLYLINE3D (???) with 100 segments ...
        else:
            raise NotImplementedError

    @staticmethod
    def from_svg(path,color):
        """
        :param path: svg path
        :return: Entity of correct subtype
        """
        return Chain.from_svg(path,color)

    @staticmethod
    def from_pdf(path,trans,color):
        """
        :param path: pdf path
        :return: Entity of correct subtype
        """
        return Chain.from_pdf(path,trans,color)

    @staticmethod
    def from_dxf(e,mat3):
        """
        :param e: dxf.entity
        :param mat3: Matrix3 transform
        :return: Entity of correct subtype
        """
        def trans(pt): return rpoint(mat3(Point2(pt)))

        if e.dxftype=='LINE':
            start=trans(Point2(e.start[:2]))
            end=trans(Point2(e.end[:2]))
            res=Segment2(start,end)
        elif e.dxftype == 'ARC':
            c=Point2(e.center[:2])
            startangle=radians(e.startangle)
            start=c+Polar(e.radius,startangle)
            endangle=radians(e.endangle)
            end=c+Polar(e.radius,endangle)
            res=Arc2(trans(c),trans(start),trans(end))
        elif e.dxftype == 'CIRCLE':
            c=Point2(e.center[:2])
            res=Circle(trans(c), e.radius)
        elif e.dxftype == 'POLYLINE':
            res=Chain.from_dxf(e,trans)
        elif e.dxftype == 'LWPOLYLINE':
            res=Chain.from_dxf(e,trans)
        elif e.dxftype == 'SOLID':
            return None #TODO : implement
        elif e.dxftype == 'POINT':
            return None #ignore
        elif e.dxftype == 'TEXT':
            p=trans(Point2(e.insert[:2]))
            res=Text(e.text,p,size=e.height,rotation=e.rotation)
        else:
            logging.warning('unhandled entity type %s'%e.dxftype)
            return None
        res.dxf=e #keep link to source entity
        res.color=acadcolors[e.color % len(acadcolors)]
        res.layer=e.layer
        return res

#Python is FANTASTIC ! here we set Entity as base class of some classes previously defined in geom module !
Segment2.__bases__ += (Entity,)
Circle.__bases__ += (Entity,) # adds it also to Arc2

class Spline(Entity, Geometry):
    """cubic spline segment"""

    def __init__(self, points):
        """:param points: list of (x,y) tuples"""
        super(Spline,self).__init__()
        self.p=[Point2(xy) for xy in points]

    def __copy__(self):
        return self.__class__(self.p)

    copy = __copy__

    @property
    def start(self):
        return self.p[0]

    @property
    def end(self):
        return self.p[-1]

    @property
    def xy(self):
        return [pt.xy for pt in self.p]

    @property
    def length(self):
        """:return: float (very) approximate length"""
        return sum((x.dist(self.p[i - 1]) for i, x in enumerate(self.p) if i>0))

    def bbox(self):
        res=BBox()
        for p in self.p:
            res+=p
        return res

    def swap(self):
        """ swap start and end"""
        self.p.reverse() #reverse in place

    def _apply_transform(self, t):
        self.p=[t*p for p in self.p]

'''
def Spline(pts):
    # uses http://www.charlespetzold.com/blog/2012/12/Bezier-Circles-and-Bezier-Ellipses.html
    # TODO understand http://itc.ktu.lt/itc354/Riskus354.pdf and implement Arc cleanly
    (p0,p1,p2,p3)=pts
    t0=Vector2(Point2(p1)-p0)/0.55
    c=Point2(p0)-t0.cross()
    return Arc2(c,p0,p3)
'''

class Group(list, Geometry):
    """group of Entities
    Notice : a Group is NOT an Entity (I don't remember why...)
    but it is a Geometry since we can intesect, connect and compute distances between Groups
    """

    def append(self,entity,**kwargs):
        if entity is not None:
            if isinstance(entity,Entity):
                for key in kwargs:
                    setattr(entity, key, kwargs[key])
                super(Group,self).append(entity)
            else: #ignore invalid items
                logging.warning('skipped object %s'%entity)
        return self

    def extend(self,entities,**kwargs):
        if not kwargs:
            return super(Group,self).extend(entities)

        for entity in entities:
            self.append(entity,**kwargs)

    def bbox(self):
        """
        :return: :class:`BBox` bounding box of Entity"""
        return sum((entity.bbox() for entity in self), BBox())

    @property
    def length(self):
        return sum((entity.length for entity in self))

    def __copy__(self):
        return self.__class__(self)

    copy = __copy__

    def _apply_transform(self,trans):
        for entity in self:
            entity._apply_transform(trans)

    def intersect(self, other):
        """
        :param other: `geom.Entity`
        :result: generate tuples (Point2,Entity) of intersections between other and each Entity
        """
        try:
            iter(other)
        except:
            other=[other]
        for e in self:
            for o in other:
                inter=e.intersect(o)
                if inter:
                    inter=e.intersect(o)
                    yield (inter,e)
    
    def connect(self, other):
        for (inter, _) in self.intersect(other):
            if isinstance(inter,Point2):
                return Segment2(inter,inter) #segment of null length
            if isinstance(inter,Segment2):
                return Segment2(inter.p,inter.p) #segment of null length
            raise
        try:
            iter(other)
        except:
            other=[other]
        return min((e.connect(o) for e in self for o in other), key=lambda e:e.length)

    def swap(self):
        """ swap start and end"""
        super(Group,self).reverse() #reverse in place
        for e in self:
            e.swap()

    def patches(self, **kwargs):
        """:return: list of :class:`~matplotlib.patches.Patch` corresponding to group"""
        patches=[]
        for e in self:
            patches.extend(e.patches(**kwargs))
        return patches

    def from_dxf(self, dxf, layers=None, only=[], ignore=[], trans=None, recurse=True):
        """
        :param dxf: dxf.entity
        :return: Entity of correct subtype
        """
        if trans is None:
            trans=Trans()

        try:
            self.dxf #already defined ?
        except:
            self.dxf=dxf

        for e in dxf:
            if layers and e.layer not in layers:
                continue
            if only:
                if e.dxftype in only:
                    self.append(Entity.from_dxf(e, trans))
                else:
                    continue
            elif e.dxftype in ignore:
                continue
            elif e.dxftype == 'INSERT':
                t2 = trans*Trans(1, e.insert[:2], e.rotation)
                if recurse:
                    self.from_dxf(self.block[e.name].dxf, layers=None, ignore=ignore, only=None, trans=t2, recurse=recurse)
                else:
                    raise NotImplementedError() #TODO add
            else:
                self.append(Entity.from_dxf(e, trans))
        return self

    def to_dxf(self, **kwargs):
        """:return: flatten list of dxf entities"""
        res=[]
        for e in self:
            r=e.to_dxf(**kwargs)
            if not isinstance(r,list): #flatten
                r=[r]
            res+=r
        return res

class Chain(Group,Entity): #inherit in this order for overloaded methods to work correctly
    """ group of contiguous Entities (Polyline or similar)"""

    def __init__(self,data=[]):
        Group.__init__(self,data)
        Entity.__init__(self)

    @property
    def start(self):
        return self[0].start

    @property
    def end(self):
        return self[-1].end

    def __repr__(self):
        (s,e)=(self.start,self.end) if len(self)>0 else (None,None)
        return '%s(%s,%s,%d)' % (self.__class__.__name__,s,e,len(self))

    def append(self,edge,tol=1E6):
        """
        :return: self, or None if edge is not contiguous
        """

        if len(self)==0:
            return super(Chain,self).append(edge)
        if self.end.dist(edge.start)<=tol:
            return super(Chain,self).append(edge)
        if self.end.dist(edge.end)<=tol:
            edge.swap()
            return super(Chain,self).append(edge)
        if len(self)>1 :
            return None
        #try to reverse the first edge

        if self.start.dist(edge.start)<=tol:
            self[0].swap()
            return super(Chain,self).append(edge)
        if self.start.dist(edge.end)<=tol:
            self[0].swap()
            edge.swap()
            return super(Chain,self).append(edge)
        return None

    @staticmethod
    def from_pdf(path,trans,color):
        """
        :param path: pdf path
        :return: Entity of correct subtype
        :see: http://www.adobe.com/content/dam/Adobe/en/devnet/acrobat/pdfs/PDF32000_2008.pdf p. 132

        """
        def _pt(*args):
            return trans(Point2(*args))

        chain=Chain()
        chain.color=color #TODO handle multicolor chains
        start=None # ensure exception if 'm' is not first
        for code in path:
            if code[0]=='m':
                if start:
                    logging.error("multiple m's in pdf path %s"%path)
                    break # return chain till here, ignore the rest
                home=_pt(code[1:3])
                start=home
                entity=None
                continue
            elif code[0]=='l': #line
                end=_pt(code[1:3])
                entity=Segment2(start,end)
            elif code[0]=='c': #Bezier 2 control points (no arcs in pdf!)
                x1,y1,x2,y2,x3,y3=code[1:]
                end=_pt(x3,y3)
                entity=Spline([start,_pt(x1,y1),_pt(x2,y2),end])
            elif code[0]=='v': #Bezier 1 control point
                x2,y2,x3,y3=code[1:]
                end=_pt(x3,y3)
                entity=Spline([start,start,_pt(x2,y2),end])
            elif code[0]=='y': #Bezier 0 control point
                end=_pt(code[1:3])
                entity=Spline([start,start,end,end])
            elif code[0]=='h': #close to home
                entity=Segment2(start,home)
                entity.color=chain.color
            else:
                logging.warning('unsupported path command %s'%code[0])
                raise NotImplementedError
            entity.color=color
            chain.append(entity)
            start=end
        if len(chain)==1:
            return chain[0] #single entity
        return chain

    @staticmethod
    def from_svg(path,color):
        """
        :param path: svg path
        :return: Entity of correct subtype
        """
        from svg.path import Line, CubicBezier
        chain=Chain()
        chain.color=color
        def _point(svg):
            return Point2(svg.real, svg.imag)

        for seg in path._segments:
            if isinstance(seg,Line):
                entity=Segment2(_point(seg.start),_point(seg.end))
            elif isinstance(seg,CubicBezier):
                entity=Spline([_point(seg.start),_point(seg.control1),_point(seg.control2),_point(seg.end)])
            else:
                logging.error('unknown segment %s'%seg)
                entity=None #will crash below
            entity.color=chain.color
            chain.append(entity)
        return chain

    @staticmethod
    def from_dxf(e,mat3):
        """
        :param e: dxf.entity
        :param mat3: Matrix3 transform
        :return: Entity of correct subtype
        """
        def trans(pt): return rpoint(mat3(Point2(pt)))

        if e.dxftype == 'POLYLINE':
            res=Chain()
            for i in range(1,len(e.vertices)):
                start=e.vertices[i-1].location[:2] #2D only
                end=e.vertices[i].location[:2] #2D only
                bulge=e.vertices[i-1].bulge
                if bulge==0:
                    res.append(Segment2(trans(start),trans(end)))
                else:
                    #formula from http://www.afralisp.net/archive/lisp/Bulges1.htm
                    theta=4*atan(bulge)
                    chord=Segment2(start,end)
                    c=chord.length
                    s=bulge*c/2
                    r=(c*c/4+s*s)/(2*s) #radius (negative if clockwise)
                    gamma=(pi-theta)/2
                    angle=chord.v.angle()+copysign(gamma,bulge)
                    center=start+Polar(r,angle)
                    res.append(Arc2(trans(center),trans(start),trans(end)))
            if e.is_closed:
                res.append(Segment2(trans(e.vertices[-1].location[:2]),trans(e.vertices[0].location[:2])))
        elif e.dxftype == 'LWPOLYLINE':
            res=Chain()
            for i in range(1,len(e.points)):
                start=e.points[i-1]
                end=e.points[i]
                if len(end)==2:
                    res.append(Segment2(trans(start),trans(end)))
                else:
                    bulge=end[2]
                    #formula from http://www.afralisp.net/archive/lisp/Bulges1.htm
                    theta=4*atan(bulge)
                    chord=Segment2(start,end)
                    c=chord.length
                    s=bulge*c/2
                    r=(c*c/4+s*s)/(2*s) #radius (negative if clockwise)
                    gamma=(pi-theta)/2
                    angle=chord.v.angle()+copysign(gamma,bulge)
                    center=start+Polar(r,angle)
                    res.append(Arc2(trans(center),trans(start),trans(end)))
            if e.is_closed:
                res.append(Segment2(trans(e.points[-1]),trans(e.points[0])))
        else:
            logging.warning('unhandled entity type %s'%e.dxftype)
            return None
        res.dxf=e #keep link to source entity
        res.color=acadcolors[e.color % len(acadcolors)]
        res.layer=e.layer
        for edge in res:
            edge.dxf=e
            edge.color=res.color
            edge.layer=res.layer
        return res

    def to_dxf(self, split=False, **attr):
        """:return: polyline or list of entities along the chain"""
        if split: #handle polylines as separate entities
            return super(Chain,self).to_dxf(**attr)

        #assume chain color is the same as the first element's
        color=self[0]._dxf_color()
        from dxfwrite.entities import Polyline
        flags=1 if self.isclosed() else 0
        res=Polyline(color=color, flags=flags, **attr)

        for e in self:
            if isinstance(e,Line2):
                res.add_vertex(e.start.xy)
            elif isinstance(e, Arc2):
                bulge=tan(e.angle()/4)
                res.add_vertex(e.start.xy,bulge=bulge)
            else: #we have a Spline in the chain. Split it for now
                return super(Chain,self).to_dxf(**attr)

        if not self.isclosed():
            res.add_vertex(self.end.xy)
        return res

class Rect(Chain):
    """a rectangle starting at low/left and going trigowise through top/right"""
    def __init__(self,*args):
        if len(args) == 1: #assume it is a Rect of 4 Segment2
            self.p1=Point2(args[0][0].p1)
            self.p2=Point2(args[0][2].p1)
        else:
            v1,v2 = Point2(args[0]),Point2(args[1])
            p1=Point2(min(v1.x,v2.x),min(v1.y,v2.y))
            p2=Point2(max(v1.x,v2.x),max(v1.y,v2.y))
        self.append(Segment2(p1,(p2.x,p1.y)))
        self.append(Segment2((p2.x,p1.y),p2))
        self.append(Segment2(p2,(p1.x,p2.y)))
        self.append(Segment2((p1.x,p2.y),p1))
        self.p1=p1
        self.p2=p2
        
    def __repr__(self):
        return '%s(%s,%s)' % (self.__class__.__name__,self.p1,self.p2)

class Text(Point2, Entity):

    def __init__( self, text, point, size=12, rotation=0):
        """
        :param text: string
        :param point: Point2
        :param size: size in points
        :param rotation: float angle in degrees trigowise
        """
        super(Text,self).__init__(point)
        self.text=text
        self.size=size
        self.rotation=rotation

    def bbox(self):
        return BBox(self,self)

    def to_dxf(self, **attr):
        #TODO: avoir duplicating Entity.to_dxf code
        color=self._dxf_color()
        if color>=0:
            attr['color']=color

        try:
            layer=self.layer
            attr['layer']=layer
        except:
            pass

        from dxfwrite.entities import Text as Text_dxf
        return Text_dxf(insert=self.xy, text=self.text, height=self.size, **attr)

    def patches(self, **kwargs):
        """:return: list of (a single) :class:`~matplotlib.patches.Patch` corresponding to entity"""
        #http://matplotlib.org/api/text_api.html?highlight=text#module-matplotlib.text

                #entities below may be filled, so let's handle the color first
        if 'color' in kwargs: # color attribute refers to edgecolor for coherency
            kwargs.setdefault('edgecolor',kwargs.pop('color'))
            kwargs.setdefault('fill',False)

        kwargs.setdefault('family','sans-serif')

        from matplotlib.text import Annotation
        from matplotlib.text import Text as Text_pdf

        return [Text_pdf(self.x,self.y, self.text, size=self.size, rotation=self.rotation,**kwargs)]

class Drawing(Group):
    """list of Entities representing a vector graphics drawing"""

    def __init__(self, filename=None, data=[], **kwargs):
        Group.__init__(self,data)
        if filename:
            self.load(filename,**kwargs)

    def load(self,filename, **kwargs):
            ext=filename.split('.')[-1].lower()
            if ext=='dxf':
                self.read_dxf(filename, **kwargs)
            elif ext=='svg':
                self.read_svg(filename, **kwargs)
            elif ext=='pdf':
                self.read_pdf(filename, **kwargs)
            else:
                raise IOError("file format .%s not (yet) supported"%ext)

    def read_pdf(self,filename,**kwargs):
        """ reads a vector graphics on a .pdf file
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
        me=self

        class _Device(PDFDevice):
            def paint_path(self, graphicstate, stroke, fill, evenodd, path):
                color=None
                try: color=stroke.color
                except: pass
                if not color:
                    try: color=fill.color
                    except: pass
                t=Matrix3()
                # geom.Matrix 3 has the following format:
                # a b c
                # e f g
                # i j k
                # so we read the components already available in self.ctm:
                t.a,t.b,t.e,t.f,t.c,t.g=tuple(self.ctm)
                for sub in split(path,lambda x:x[0]=='m',True):
                    if not sub: #first sub is empty because 'm' occurs in first place
                        continue
                    e=Entity.from_pdf(sub,t,color)
                    if e:
                        me.append(e)
                    else:
                        logging.warning('pdf path ignored %s'%sub)

        # the PDFPageInterpreter doesn't handle colors yet, so we patch it here:
        class _Interpreter(PDFPageInterpreter):
            # stroke
            def do_S(self):
                self.device.paint_path(self.graphicstate, self.scs, False, False, self.curpath)
                self.curpath = []
                return

            # fill
            def do_f(self):
                self.device.paint_path(self.graphicstate, self.scs, self.ncs, False, self.curpath)
                self.curpath = []
                return

            # setrgb-stroking
            def do_RG(self, r, g, b):
                from pdfminer.pdfcolor import LITERAL_DEVICE_RGB
                self.do_CS(LITERAL_DEVICE_RGB)
                self.scs.color='#%02x%02x%02x' % (r*255,g*255,b*255)

            # setcolor stroking
            def do_sc(self):
                r,g,b=self.pop(self.scs.ncomponents)
                self.scs.color='#%02x%02x%02x' % (r*255,g*255,b*255)

            # setcolor nonstroking
            def do_scn(self):
                try:
                    r,g,b=self.pop(self.ncs.ncomponents)
                    self.ncs.color='#%02x%02x%02x' % (r*255,g*255,b*255)
                except:
                    pass


        #then all we have to do is to launch PDFMiner's parser on the file
        fp = open(filename, 'rb')
        parser = PDFParser(fp)
        document = PDFDocument(parser, fallback=False)
        rsrcmgr = PDFResourceManager()
        device = _Device(rsrcmgr)
        interpreter = _Interpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(document):
            interpreter.process_page(page)
            break #handle one page only
        return

    def read_svg(self,filename, **kwargs):
        #from http://stackoverflow.com/questions/15857818/python-svg-parser
        from xml.dom import minidom
        doc = minidom.parse(filename)  # parseString also exists
        trans=Trans()
        trans.f=-1 #flip y axis
        for path in doc.getElementsByTagName('path'):
            #find the color... dirty, but simply understandable
            color=path.getAttribute('fill') #assign filling color to default stroke color
            alpha=1
            style=path.getAttribute('style')
            for s in style.split(';'):
                item=s.split(':')
                if item[0]=='opacity':
                    alpha=float(item[1])
                elif item[0]=='stroke':
                    color=item[1]
            if not color or alpha==0 : #ignore picture frame
                continue
            # process the path
            d=path.getAttribute('d')
            from svg.path import parse_path
            e=Entity.from_svg(parse_path(d),color)
            e=trans*e
            self.append(e)
        doc.unlink()

    def read_dxf(self, filename, options=None, **kwargs):
        """reads a .dxf file
        :param filename: string path to .dxf file to read
        :param options: passed to :class:`~dxfgrabber.drawing.Drawing`constructor
        :param layers: list of layer names to consider. entities not on these layers are ignored. default=None: all layers are read
        :param only: list of dxf entity types names that are read. default=[]: all are read
        :param ignore: list of dxf entity types names that are ignored. default=[]: none is ignored
        """
        import dxfgrabber
        try:
            self.dxf = dxfgrabber.readfile(filename,options)
        except:
            logging.error('could not read %s'%filename)
            return
        self.name = filename

        #build dictionary of blocks
        self.block={}
        for block in self.dxf.blocks:
            self.block[block.name]=Group().from_dxf(block._entities)

        super(Drawing, self).from_dxf(
            self.dxf.entities,
            layers=kwargs.get('layers',None),
            only=kwargs.get('only',[]),
            ignore=kwargs.get('only',[]),
            trans=Trans(),
            recurse=kwargs.get('recurse',True),
        )

    def img(self, size=[256, 256], border=5, box=None, layers=None, ignore=[], forcelayercolor=False, antialias=1,background='white'):
        """
        :param size: [x,y] max size of image in pixels. if one coord is None, the other one will be enforced
        :param border: int border width in pixels
        :param box: class:`BBox` bounding box. if None, box is calculated to contain all drawn entities
        :param layers: list or dictionary of layers to draw. None = all layers
        :param ignore: list of strings of entity types to ignore
        :result: :class:`PIL:Image` rasterized image
        """

        from PIL import Image, ImageDraw, ImageFont  # PIL or Pillow

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
                    b = list((trans * Point2(e.start[:2])).xy)
                    b+=list(trans(Point2(e.end[:2])).xy)
                    draw.line(b, fill=pen)
                elif e.dxftype == 'CIRCLE':
                    b = cbox(Point2(e.center[:2]), e.radius)
                    b = b.trans(trans)
                    draw.ellipse(b(), outline=pen)
                elif e.dxftype == 'ARC':
                    c = Point2(e.center[:2])
                    b = cbox(c, e.radius)
                    b = b.trans(trans)
                    b = list(map(rint, b()))
                    startangle = degrees(trans.angle(radians(e.startangle)))
                    endangle = degrees(trans.angle(radians(e.endangle)))
                    startangle, endangle = endangle, startangle  # swap start/end because of Y symmetry
                    draw.arc(b, int(startangle), int(endangle), fill=pen)
                elif e.dxftype == 'POLYLINE':
                    b = []
                    for v in e.vertices:
                        b+=list(trans(Point2(v.location[:2])).xy)
                    draw.line(b, fill=pen)
                elif e.dxftype == 'SPLINE':
                    b = []
                    for v in e.controlpoints:
                        b+=list(trans(Point2(v[:2])).xy)
                    draw.line(b, fill=pen) # splines are drawn as lines for now...
                elif e.dxftype == 'TEXT':
                    h = e.height * trans.mag()  # [pixels]
                    if h < 4:
                        continue  # too small
                    font = None
                    try:
                        font = ImageFont.truetype("c:/windows/fonts/Courier New.ttf", h)
                        print("font loaded !")
                    except:
                        pass
                    if not font:
                        h = h * 1.4  # magic factor (TODO : calculate DPI of image and conversions...)
                        fh = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 36, 40, 48, 60]
                        i, h = min(enumerate(fh), key=lambda x: abs(x[1] - h))  # http://stackoverflow.com/questions/9706041/finding-index-of-an-item-closest-to-the-value-in-a-list-thats-not-entirely-sort
                        import os
                        path=os.path.dirname(os.path.abspath(__file__))
                        font = ImageFont.load(path + '/base_pil/72/Courier New_%s_72.pil' % h)
                    pt = Point2(e.insert[0], e.insert[1] + e.height)  # ACAD places texts by top left point...
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

        s = list(map(operator.div, [float(x - border) * antialias if x else 1E9 for x in size ], box.size().xy))
        trans = Trans(scale=min(s))
        size = trans * box.size() + Point2(2 * antialias * border, 2 * antialias * border)  # add borders as an offset
        offset = size / 2 - trans(box.center())  # offset in pixel coordinates
        trans = trans.translate(offset)
        trans = trans.scale(1, -1)  # invert y axis
        trans = trans.translate(0, size.y)  # origin is lower left corner

        img = Image.new("RGB", list(map(rint, size.xy)), background)
        draw = ImageDraw.Draw(img)
        _draw(self.iter(layers=layers, ignore=ignore, trans=trans, recurse=False))
        if antialias > 1:
            size = size / antialias
            img = img.resize(list(map(rint, size.xy)), Image.ANTIALIAS)
        return img

    def figure(self,**kwargs):
        """:return: matplotlib axis suitable for drawing """

        fig=plt.figure(**kwargs)

        box=self.bbox()
        #for some reason we have to plot something in order to size the window (found no other way top do it...)
        try:
            plt.plot((box.xmin,box.xmax),(box.ymin,box.ymax), alpha=0) #draw a transparent diagonal to size everything
        except:
            logging.error('drawing is empty')
            raise

        plt.axis('equal')

        import pylab
        pylab.axis('off') # turn off axis

        return fig

    def draw(self, fig=None, **kwargs):
        """ draw  entities
        :param fig: matplotlib figure where to draw. figure(g) is called if missing
        :return: fig,patch
        """

        if fig is None:
            fig=self.figure()

        p=self.patches() #some of which might be Annotations, which aren't patches but Artists...

        from itertools2 import filter2
        from matplotlib.patches import Patch
        patches,artists=filter2(p,lambda e:isinstance(e,Patch))

        if patches:
            from matplotlib.collections import PatchCollection
            plt.gca().add_collection(PatchCollection(patches,match_original=True))

        if artists:
            for e in artists:
                plt.gca().add_artist(e)
            plt.draw()

        return fig, p

    def render(self,format,**kwargs):
        """ render graph to bitmap stream
        :return: matplotlib figure as a byte stream in specified format
        """

        fig,_=self.draw(**kwargs)

        from io import BytesIO
        output = BytesIO()
        fig.savefig(output, format=format, transparent=kwargs.get('transparent',True))
        plt.close(fig)
        return output.getvalue()

    # for IPython notebooks
    def _repr_png_(self): return self.render('png')
    def _repr_svg_(self): return self.render('svg')

    def save(self,filename,**kwargs):
        """ save graph in various formats"""
        ext=filename.split('.')[-1].lower()
        if ext!='dxf':
            kwargs.setdefault('dpi',600) #force good quality
            open(filename,'wb').write(self.render(ext,**kwargs))
            return
        # save as ASCII DXF V 12
        from dxfwrite import DXFEngine as dxf
        drawing=dxf.drawing(filename)
        #remove some default tables that we don't need
        drawing.tables.layers.clear()
        drawing.tables.styles.clear()
        drawing.tables.viewports.clear()
        # self.drawing.tables.linetypes.clear()

        entities=self.to_dxf(**kwargs)

        for e in entities:
            drawing.add(e)

        """
        for i,layer in enumerate(layers):
            if layer:
                self.drawing.add_layer(layer,color=i,linetype=linetypes[i])
        """
        drawing.save()


def img2base64(img, fmt='PNG'):
    """
    :param img: :class:`PIL:Image`
    :result: string base64 encoded image content in specified format
    :see: http://stackoverflow.com/questions/14348442/django-how-do-i-display-a-pil-image-object-in-a-template
    """
    import io, base64
    output = io.StringIO()
    img.save(output, fmt)
    output.seek(0)
    output_s = output.read()
    return base64.b64encode(output_s)

