#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Read/Write and handle vector graphics in .dxf, .svg and .pdf formats

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

from .math2 import rint, product
from .itertools2 import split, filter2, subdict
from .geom import *
from .colors import color_to_aci, aci_to_color, color_lookup
from .interval import Box

from . import plot #set matplotlib backend
import matplotlib.pyplot as plt # after import .plot

def Trans(scale=1, offset=None, rotation=None):
    """
    :param scale: float or (scalex,scaley) tuple of scale factor
    :param offset: :class:`~geom.Vector3`
    :param rotation: float angle in degrees
    :return: :class:`~geom.Matrix3` of generalized scale+offset+rotation
    """
    res = Matrix3()
    if rotation:
        res = res.rotate(radians(rotation))
    if scale != 1:
        try:
            res = res.scale(scale[0],scale[1])
        except:
            res = res.scale(scale)
    if offset:
        res = res.translate(offset)
    return res

class BBox(Box):
    """bounding box"""
    def __init__(self, p1=None, p2=None):
        """
        :param pt1: :class:`~geom.Point2` first corner (any)
        :param pt2: :class:`~geom.Point2` opposite corner (any)
        """
        super(BBox,self).__init__(2)
        if p1 : self+=p1
        if p2 : self+=p2

    @property
    def xmin(self): return self[0].start

    @property
    def ymin(self): return self[1].start

    @property
    def xmax(self): return self[0].end

    @property
    def ymax(self): return self[1].end
    
    @property
    def xmed(self): return (self.xmin+self.xmax)/2

    @property
    def ymed(self): return (self.ymin+self.ymax)/2
    
    @property
    def width(self): return self[0].size
    
    @property
    def height(self): return self[1].size
    
    def __contains__(self, other):
        """:return: True if other lies in bounding box."""
        if isinstance(other,(Box,tuple)):
            return super(BBox,self).__contains__(other)

        #process simple geom entities without building Box objects
        if isinstance(other,Point2):
            return super(BBox,self).__contains__(other.xy)
        if isinstance(other,Segment2):
            return (super(BBox,self).__contains__(other.p1.xy)
                and super(BBox,self).__contains__(other.p2.xy))
            
        #for more complex entites, get the box
        if isinstance(other,Entity):    
            return super(BBox,self).__contains__(other.bbox())
            
        #if we reached till here, suppose other is an iterable    
        return all(x in i for i,x in zip(self,other))
        

    def __iadd__(self, pt):
        """
        enlarge box if required to contain specified point
        :param pt1: :class:`geom.Point2` point to add
        """
        if pt is None:
            return self
        elif isinstance(pt,Point2):
            self+= pt.xy
        else:
            super(BBox,self).__iadd__(pt)
        return self

    def __call__(self):
        """:return: list of flatten corners"""
        l = list(self.p1.xy)+list(self.p2.xy)
        return l

    def size(self):
        """:return: :class:`geom.Vector2` with xy sizes"""
        return Vector2(super(BBox,self).size)

    def center(self):
        """:return: Pt center"""
        return Point2(super(BBox,self).center)

    def trans(self, trans):
        """
        :param trans: Xform
        :return: :class:`BBox` = self transformed by trans
        """
        res = BBox(trans(self.min), trans(self.max))
        # add 2 more corners as they matter if we rotate the box
        res += trans(self.xmin, self.ymax)
        res += trans(self.xmax, self.ymin)
        return res

def rpoint(pt,decimals=3): # rounds coordinates to number of decimals
    return Point2([round(x,decimals) for x in pt.xy])

def calcBulge(p1,bulge,p2):
    #taken from http://free-cad.sourceforge.net/SrcDocu/de/d1f/importDXF_8py_source.html
    """
    calculates intermediary vertex for curved segments.
    algorithm from http://www.afralisp.net/lisp/Bulges1.htm
    """
    chord = p2-p1
    sagitta = (bulge * chord.length)/2
    startpoint = p1+chord/2
    perp = chord.cross()
    perp.normalize()
    endpoint = perp*sagitta
    return startpoint+endpoint

class Entity(object):
    """Base class for all drawing entities"""
    
    color='black' # by default

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
        if isinstance(self,Point2):
            return BBox(self,self)
        elif isinstance(self,Segment2):
            return BBox(self.start,self.end)
        elif isinstance(self,Arc2):
            #http://stackoverflow.com/questions/1336663/2d-bounding-box-of-a-sector
            res=BBox(self.p, self.p2)
            p=self.c+Vector2(self.r,0)
            if p in self : res+=p
            p=self.c+Vector2(-self.r,0)
            if p in self : res+=p
            p=self.c+Vector2(0,self.r)
            if p in self : res+=p
            p=self.c+Vector2(0,-self.r)
            if p in self : res+=p
            return res
        elif isinstance(self,Circle): #must be after Arc2 case since Arc2 is a Circle too
            rr = Vector2(self.r, self.r)
            return BBox(self.c - rr, self.c + rr)

        raise NotImplementedError()

    def isclosed(self):
        return self.end==self.start

    def isline(self):
        return isinstance(self,Line2) #or derived

    def isvertical(self,tol=0.01):
        return self.isline() and abs(self.start.x-self.end.x)<tol

    def ishorizontal(self,tol=0.01):
        return self.isline() and abs(self.start.y-self.end.y)<tol

    def _dxf_attr(self,attr):
        """
        :return: dict of attributes for dxf.entity
        """
        res={}
        res['color']=color_to_aci(attr.get('color',self.color))
        if 'layer' in attr:
            res['layer']=attr.get('layer')
        else:
            try:
                res['layer']=self.layer
            except:
                pass
        return res
        
    def to_dxf(self,**attr):
        """
        :param attr: dict of attributes passed to the dxf entity, overriding those defined in self
        :return: dxf entity
        """
        import dxfwrite.entities as dxf
        attr=self._dxf_attr(attr)

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
            from dxfwrite import curves
            spline=curves.Bezier(**attr)
            spline.start(self.start.xy,(self.p[1]-self.start).xy)
            spline.append(self.end.xy,(self.p[2]-self.end).xy)
            return spline
            return curves.Spline(self.xy, **attr) # builds a POLYLINE3D (???) with 100 segments ...
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
        res.color=aci_to_color(e.color)
        res.layer=e.layer
        return res

    def patches(self, **kwargs):
        """
        :return: list of (a single) :class:`~matplotlib.patches.Patch` corresponding to entity
        :note: this is the only method that needs to be overridden in descendants for draw, render and IPython _repr_xxx_ to work
        """
        import matplotlib.patches as patches
        from matplotlib.path import Path

        kwargs.setdefault('color',self.color)
        try:
            kwargs.setdefault('linewidth',self.width)
        except:
            pass
        
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
        if isinstance(self,Circle): #must be after isinstance(self,Arc2)
            kwargs.setdefault('fill',False)
            return [patches.Circle(self.c.xy,self.r,**kwargs)]
        if isinstance(self,Point2):
            kwargs.setdefault('fill',True)
            return [patches.Circle(self.xy,0,**kwargs)]
        if isinstance(self,Spline):
            kwargs.setdefault('fill',False)
            path = Path(self.xy, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
            return [patches.PathPatch(path, **kwargs)]
        raise NotImplementedError

    @staticmethod
    def figure(box, **kwargs):
        """
        :param box: :class:`drawing.BBox` bounds and clipping box
        :param kwargs: parameters passed to `~matplotlib.pyplot.figure`
        :return: matplotlib axis suitable for drawing
        """
            
        fig=plt.figure(**kwargs)
        # ax  = fig.add_subplot(111) # unneeded
            
        # TODO: find why this doesn't work:
        # plt.gca().set_position([box.xmin,box.ymin,box.width, box.height])
            
        #for some reason we have to plot something in order to size the window
        #TODO: find a better way... (found no other way top do it...)
        plt.plot((box.xmin,box.xmax),(box.ymin,box.ymax), alpha=0) #draw a transparent diagonal to size everything

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
            if not 'box' in kwargs:
                kwargs['box']=self.bbox()
                
            fig=self.figure(**kwargs)
            
        args=subdict(kwargs,('color','linewidth'))

        p=self.patches(**args) #some of which might be Annotations, which aren't patches but Artists...

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
        res=output.getvalue()
        plt.close(fig)
        return res

    # for IPython notebooks
    def _repr_png_(self): return self.render('png')
    def _repr_svg_(self): return self.render('svg')

#Python is FANTASTIC ! here we set Entity as base class of some classes previously defined in geom module !
Point2.__bases__ += (Entity,)
Segment2.__bases__ += (Entity,)
Circle.__bases__ += (Entity,) # adds it also to Arc2

class Spline(Entity, Geometry):
    """cubic spline segment"""

    def __init__(self, points):
        """:param points: list of (x,y) tuples"""
        super(Spline,self).__init__()
        self.p=[Point2(xy) for xy in points]

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
class _Group(Entity, Geometry):
    """ abstract class for iterable Entities"""
    def bbox(self, filter=None):
        """
        :param filter: optional function(entity):bool returning True if entity should be considered in box
        :return: :class:`BBox` bounding box of Entity
        """
        res=BBox()
        for entity in self: # do not use sum() as it copies Boxes unnecessarily
            if filter is None or filter(entity):
                res+=entity.bbox()
        return res

    @property
    def length(self):
        return sum((entity.length for entity in self))

    def intersect(self, other):
        """
        :param other: `geom.Entity`
        :result: generate tuples (Point2,Entity_self) of intersections between other and each Entity
        """
        try:
            iter(other)
        except:
            other=[other]
        for e in self:
            for o in other:
                inter=e.intersect(o)
                if inter is None: 
                    continue
                if isinstance(inter,Point2) :
                    yield (inter,e)
                elif isinstance(inter,Segment2) :
                    yield (inter.p,e)
                    yield (inter.p2,e)
                elif isinstance(inter,list) : # list of multiple points
                    for i in inter:
                        yield (i,e)
                else:
                    try:
                        for i,e2 in inter:
                            e2.group=e
                            yield (i,e2)
                    except:
                        logging.error('strange intersection %s'%inter)


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

    def patches(self, **kwargs):
        """:return: list of :class:`~matplotlib.patches.Patch` corresponding to group"""
        patches=[]
        for e in self:
            patches.extend(e.patches(**kwargs))
        return patches

    def to_dxf(self, **kwargs):
        """:return: flatten list of dxf entities"""
        res=[]
        for e in self:
            r=e.to_dxf(**kwargs)
            if not isinstance(r,list): #flatten
                r=[r]
            res+=r
        return res

class Group(list, _Group):
    """group of Entities
    but it is a Geometry since we can intersect, connect and compute distances between Groups
    """
    _color='black'
    
    @property
    def color(self):
        return self._color
    
    @color.setter
    def color(self, c):
        self._color=c
        for e in self:
            e.color=c

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

    def __copy__(self):
        #in fact it is a deepcopy...
        #TODO : make this clearer
        res=self.__class__()
        for e in self:
            res.append(copy(e))
        return res

    def _apply_transform(self,trans):
        for entity in self:
            entity._apply_transform(trans)

    def swap(self):
        """ swap start and end"""
        super(Group,self).reverse() #reverse in place
        for e in self:
            e.swap()

    def from_dxf(self, dxf, layers=None, only=[], ignore=[], trans=None, flatten=True):
        #TODO : make it work properly with flatten=False
        """
        :param dxf: dxf.entity
        :param layers: list of layer names to consider. entities not on these layers are ignored. default=None: all layers are read
        :param only: list of dxf entity types names that are read. default=[]: all are read
        :param ignore: list of dxf entity types names that are ignored. default=[]: none is ignored
        :param trans: :class:`Trans` optional transform matrix
        :parm flatten: bool flatten block structure
        :return: :class:`Entity` of correct subtype
        """
        if trans is None:
            trans=Trans()

        self.dxf=dxf

        for e in dxf:
            if layers and e.layer not in layers:
                continue
            elif e.dxftype in ignore:
                continue
            elif only and  e.dxftype not in only:
                continue
            elif e.dxftype == 'INSERT': #TODO : improve insertion on correct layer
                t2 = trans*Trans(1, e.insert[:2], e.rotation)
                if flatten:
                    self.from_dxf(self.block[e.name].dxf, layers=None, ignore=ignore, only=None, trans=t2, flatten=flatten)
                else:
                    self.append(Instance.from_dxf(e, self.block, t2))
            else:
                self.append(Entity.from_dxf(e, trans))
        return self

class Instance(_Group):
    
    def __init__(self, group, p, trans=None):
        """
        :param group: Group
        :param p: Point2 of insertion
        :param trans: optional mat3 of transformation
        """
        self.group=group
        self.trans=trans.translate(p) if trans else Trans().translate(p)

    @staticmethod
    def from_dxf(e, blocks, mat3):
        """
        :param e: dxf.entity
        :param blocks: dict of Groups indexed by name
        :param mat3: Matrix3 transform
        """
        res=Instance(blocks[e.name],Point2(e.insert[:2]),mat3)
        res.name=e.name
        # code below copied from Entity.from_dxf. TODO : merge
        res.dxf=e #keep link to source entity
        res.color=aci_to_color(e.color)
        res.layer=e.layer
        return res

    def __repr__(self):
        return '%s %s' % (self.__class__.__name__, self.group)

    def __iter__(self):
        #TODO : optimize when trans is identity
        for e in self.group:
            res=self.trans*e
            # res.copy_attrs_from(e)
            yield res

    def _apply_transform(self,trans):
        self.trans=trans*self.trans

class Chain(Group): #inherit in this order for overloaded methods to work correctly
    """ group of contiguous Entities (Polyline or similar)"""

    def __init__(self,data=[]):
        Group.__init__(self,data)

    @property
    def start(self):
        return self[0].start

    @property
    def end(self):
        return self[-1].end

    def __repr__(self):
        (s,e)=(self.start,self.end) if len(self)>0 else (None,None)
        return '%s(%s,%s,%d)' % (self.__class__.__name__,s,e,len(self))

    def append(self, edge, tol=1E6, allow_swap=True, **attrs):
        """
        append edge to chain, ensuring contiguity
        :param edge: :class:`Entity` to append
        :param tol: float tolerance on contiguity
        :param allow_swap: if True (default), tries to swap edge or self to find contiguity
        :param attrs: attributes passed to Group.append
        :return: self, or None if edge is not contiguous
        """

        if len(self)==0: #init the chain with edge
            return super(Chain,self).append(edge,**attrs)
        if self.end.distance(edge.start)<=tol:
            return super(Chain,self).append(edge,**attrs)
        if allow_swap and self.end.distance(edge.end)<=tol:
            edge.swap()
            return super(Chain,self).append(edge,**attrs)
        
        if not allow_swap:
            return None
        
        #try to swap the first edge
        
        if len(self)>1 : #but if chain already contains more than 1 edge, game is over
            return None

        if self.start.distance(edge.start)<=tol:
            self[0].swap()
            return super(Chain,self).append(edge,**attrs)
        if self.start.distance(edge.end)<=tol:
            self[0].swap()
            edge.swap()
            return super(Chain,self).append(edge,**attrs)
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

        def arc_from_bulge(start,end,bulge):
            #formula from http://www.afralisp.net/archive/lisp/Bulges1.htm
            theta=4*atan(bulge)
            chord=Segment2(start,end)
            c=chord.length
            s=bulge*c/2
            r=(c*c/4+s*s)/(2*s) #radius (negative if clockwise)
            gamma=(pi-theta)/2
            angle=chord.v.angle()+(gamma if bulge>=0 else -gamma)
            center=start+Polar(r,angle)
            return Arc2(
                trans(center),
                trans(start),
                trans(end),
                # dir=1 if bulge>=0 else -1
            )

        if e.dxftype == 'POLYLINE':
            res=Chain()
            for i in range(1,len(e.vertices)):
                start=e.vertices[i-1].location[:2] #2D only
                end=e.vertices[i].location[:2] #2D only
                bulge=e.vertices[i-1].bulge
                if bulge==0:
                    res.append(Segment2(trans(start),trans(end)))
                else:
                    res.append(arc_from_bulge(start,end,bulge))
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
                    res.append(arc_from_bulge(start,end,bulge=end[2]))
            if e.is_closed:
                res.append(Segment2(trans(e.points[-1]),trans(e.points[0])))
        else:
            logging.warning('unhandled entity type %s'%e.dxftype)
            return None
        res.dxf=e #keep link to source entity
        res.color=aci_to_color(e.color)
        res.layer=e.layer
        for edge in res:
            edge.dxf=e
            edge.color=res.color
            edge.layer=res.layer
        return res

    def to_dxf(self, split=True, **attr):
        """:return: polyline or list of entities along the chain"""
        if split: #handle polylines as separate entities
            return super(Chain,self).to_dxf(**attr)

        #if no color specified assume chain color is the same as the first element's
        color=attr.get('color', self.color or self[0].color)
        attr['color']=color_to_aci(color)
        from dxfwrite.entities import Polyline
        attr['flags']=1 if self.isclosed() else 0
        res=Polyline(**attr)

        for e in self:
            if isinstance(e,Line2):
                res.add_vertex(e.start.xy)
            elif isinstance(e, Arc2):
                bulge=tan(e.angle()/4)
                res.add_vertex(e.start.xy,bulge=bulge)
            else: 
                if attr.pop('R12',True): #R12 doesn't handle splines.
                    attr['color']=color #otherwise it causes trouble
                    del attr['flags']
                    return super(Chain,self).to_dxf(**attr)

        if not self.isclosed():
            res.add_vertex(self.end.xy)
        return res

class Rect(Chain):
    """a rectangle starting at low/left and going trigowise through top/right"""
    def __init__(self,*args):
        if isinstance(args[0], Rect): #copy constructor
            p1=Point2(args[0].p1)
            p2=Point2(args[0].p2)
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

dtp= 25.4/72 # one dtp point in mm https://en.wikipedia.org/wiki/Point_(typography)#Current_DTP_point_system

class Text(Entity):

    def __init__( self, text, point, size=12, rotation=0):
        """
        :param text: string
        :param point: Point2
        :param size: size in points
        :param rotation: float angle in degrees trigowise
        """
        self.p=Point2(point)
        self.text=text
        self.size=size # unit is dtp
        self.rotation=rotation
        
    def _apply_transform(self, t):
        self.p=t*self.p
        self.rotation+=degrees(t.angle())

    def bbox(self): #TODO : improve this very rough approximation
        return BBox(self.p,self.p+Vector2(0.8*len(self.text)*self.size*dtp,self.size*dtp))
    
    @property
    def length(self): #TODO : improve this very rough approximation
        """:return: float length of the text contour in mm"""
        return len(self.text)*3.6*self.size*dtp
    
    def intersect(self,other):
        return None #TODO implement

    def to_dxf(self, **attr):
        from dxfwrite.entities import Text as Text_dxf
        return Text_dxf(insert=self.p.xy, text=self.text, height=self.size, **self._dxf_attr(attr))

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

        return [Text_pdf(self.p.x,self.p.y, self.text, size=self.size, rotation=self.rotation,**kwargs)]

class Drawing(Group):
    """list of Entities representing a vector graphics drawing"""

    def __init__(self, data=[], **kwargs):
        if isinstance(data,six.string_types): #filename
            self.load(data,**kwargs)
        else:
            Group.__init__(self,data)

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
        :param kwargs: dict of optional parameters passed to :method:`Group.from_dxf`
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
            self.block[block.name]=Group().from_dxf(block._entities, **kwargs)

        super(Drawing, self).from_dxf(self.dxf.entities, **kwargs)

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
                pen = aci_to_color(i)
                if pen==background:
                    pen=aci_to_color(255-i) #TODO: Check
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
        _draw(self.iter(layers=layers, ignore=ignore, trans=trans, flatten=False))
        if antialias > 1:
            size = size / antialias
            img = img.resize(list(map(rint, size.xy)), Image.ANTIALIAS)
        return img

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

