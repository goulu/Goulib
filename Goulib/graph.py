#!/usr/bin/env python
# coding: utf8
"""
efficient Euclidian Graphs for :mod:`networkx` and related algorithms

:requires:
* `networkx <http://networkx.github.io/>`_
* `matplotlib <http://pypi.python.org/pypi/matplotlib/>`_

:optional:
* `scipy <http://www.scipy.org/>`_ for delauney triangulation
* `rtree <http://toblerity.org/rtree/>`_ for faster GeoGraph algorithms
"""

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2014, Philippe Guglielmetti"
__credits__ = []
__license__ = "LGPL"

import logging, math, six, json

import networkx as nx # http://networkx.github.io/

from . import plot #set matplotlib backend
import matplotlib.pyplot as plt # after import .plot

try:
    import numpy, scipy.spatial
    SCIPY=True
except Exception:
    logging.warning('scipy not available, delauney triangulation is not supported')
    SCIPY=False

from . import math2
from . import itertools2

"""
finding the nearest neighbor in a large GeoGraph is much faster with the
RTree package, but it's not generally available, so a pure python fallback
is provided
"""

try:
    from rtree import index # http://toblerity.org/rtree/
    RTREE=True
except Exception: #fallback, especially because I couldn't manage to install rtree on travis-ci
    RTREE=False

if not RTREE:
    logging.warning('rtree not available')

    class index: #mimics rtree.index module

        class Property:
            def set_dimension(self,n):
                self.n=n

        class Index(dict):
            """fallback for rtree.index"""
            def __init__(self,properties):
                self.prop=properties
                self.bounds=[None]*properties.n*2 # minx,miny,maxx,maxy, ...
            def count(self,ignored):
                return len(self)
            def insert(self,k,p,_):
                self[k]=p
                for i,c in enumerate(p):
                    v=self.bounds[i]
                    self.bounds[i]=min(c if v is None else v,c)
                    v=self.bounds[self.prop.n+i]
                    self.bounds[self.prop.n+i]=max(c if v is None else v,c)
            def delete(self,k,_):
                del self[k]
            def nearest(self,p, num_results, objects='raw'):
                """ very inefficient, but remember it's a fallback..."""
                return itertools2.best(list(self.values()),key=lambda q:math2.dist(p,q),n=num_results)

_nk=0 # node key

try: # pygraphviz is optional
    from pygraphviz import AGraph # http://pygraphviz.github.io/
except Exception:
    class AGraph(): pass #dummy class to let _Geo.__init__ work nevertheless


def to_networkx_graph(data,create_using=None,multigraph_input=False):
    """Make a NetworkX graph from a known data structure.
    enhances `networkx.convert.to_networkx_graph`
    :param data: any type handled by `convert.to_networkx_graph`, plus:
    * :class:`scipy.spatial.qhull.Delaunay` to enable building a graph from a delauney triangulation

    If create_using is a :class:`GeoGraph`and data is a Graph where nodes have a 'pos' attribute,
    then this attribute will be used to rename nodes as (x,y,...) tuples suitable for GeoGraph.

    """
    if SCIPY and isinstance(data,scipy.spatial.qhull.Delaunay):
        create_using.delauney=data
        triangles=data.points[data.simplices]
        for point in triangles:
            p=list(map(tuple,point)) #convert from numpy to regular list
            create_using.add_edge(p[0],p[1])
            create_using.add_edge(p[1],p[2])
            create_using.add_edge(p[2],p[0])
        return create_using
    elif isinstance(data,nx.Graph):
        if isinstance(create_using,_Geo):
            tol=create_using.tol
            create_using.tol=0 #zero tolerance when copying
            for k in data.node: #create nodes
                create_using.add_node(k,**data.node[k])
            for u,v,d in data.edges(data=True):
                create_using.add_edge(u,v,**d)
            create_using.tol=tol # revert original tolerance
            return create_using

        # pass only the adjacency matrix to ensure node keys aren't trashed in to_networkx_graph
        return nx.convert.to_networkx_graph(data.adj,create_using,data.is_multigraph())
    else:
        return nx.convert.to_networkx_graph(data,create_using,multigraph_input)



class _Geo(plot.Plot):
    """base class for graph with nodes at specified positions.
    if edges have a "length" attribute, it is used to compute distances,
    otherwise euclidian distance between nodes is used by default
    """
    def __init__(self, parent, data=None, nodes=None, **kwargs):
        """
        :param parent: type of the graph. nx.Graph, nx.DiGraph or nx.MultiGraph
        :param data: see :meth:`to_networkx_graph` for valid types
        """
        self.parent=parent
        self.idx=None # index is created at first call to add_node

        self._map={} #map from original node name to position for AGraph and other graphs were 'pos' is a node attribute

        if data:
            if isinstance(data,six.string_types): # suppose data is a filename
                ext=data.split('.')[-1].lower()
                if ext=='dot':
                    data=nx.nx_pydot.read_dot(data) #https://github.com/artiste-qb-net/quantum-fog/issues/9
                else:
                    raise(Exception('unknown file format'))
            elif isinstance(data,AGraph):
                if not getattr(data,'has_layout',False):
                    data.layout()

            to_networkx_graph(data,self)
        elif nodes:
            for node in nodes:
                self.add_node(node)

        self.render_args={}


    def copy(self):
        """
        :return: copy of self graph
        """
        # does not use deepcopy because the rtree index must be rebuilt
        return self.__class__(self,**self.graph)

    def __eq__(self,other):
        """
        :return: True if self and other are equal
        """
        def eq(a,b): return a==b
        return nx.is_isomorphic(self,other, node_match=eq, edge_match=eq)

    def __nonzero__(self):
        """
        :return: True if graph has at least one node
        """
        return self.number_of_nodes()>0

    __bool__ = __nonzero__

    '''
    def __getattr__(self,name):
        """called if name attribute is not found in self"""
        return self.graph.get(name)
    '''

    def clear(self):
        #saves some graph attributes cleared by convert._prep_create_using
        t,m=self.tol,self.multi
        self.parent.clear(self)
        self.multi=m
        self.graph['tol']=t

    @property
    def tol(self):
        return self.graph['tol']

    @tol.setter
    def tol(self, tol):
        self.graph['tol']=tol

    @property
    def multi(self):
        return self.graph.get('multi',True)

    @multi.setter
    def multi(self, s):
        self.graph['multi']=s

    def is_multigraph(self):
        """used internally in constructor"""
        try:
            return self.multi
        except AttributeError:
            return True

    def pos(self,nodes=None):
        """
        :param nodes: a single node, an iterator of all nodes if None
        :return: the position of node(s)
        """
        try:
            return self.node[nodes]['pos']
        except KeyError:
            pass
        if isinstance(nodes,tuple):
            return nodes
        if nodes is None:
            nodes=self.nodes()
        return (self.pos(n) for n in nodes)


    def dist(self,u,v):
        """
        :return: float distance between nodes u and v
        """

        try:
            return self[u][v]['length']
        except Exception:
            return math2.dist(self.pos(u),self.pos(v))

    def length(self,edges=None):
        """
        :param edges: iterator over edges either as (u,v,data) or (u,v,key,data). If None, all edges are taken
        :returns: sum of 'length' attributes of edges
        """

        if edges is None: # take all edges
            return self.size(weight="length")

        res=0
        for edge in edges:
            data=edge[-1]
            (u,v)=edge[:2]
            try:
                l=data['length']
            except:
                l=self.dist(u,v)
                data['length']=l
            res+=l
        return res

    def box(self):
        """:return: nodes bounding box as (xmin,ymin,...),(xmax,ymax,...)"""
        if self.idx is None: #empty graph
            return (0,0)
        minmax=self.idx.bounds
        n=len(minmax)//2
        return tuple(minmax[:n]),tuple(minmax[n:])

    def box_size(self):
        """:return: (x,y) size"""
        a,b=self.box()
        try:
            a[0],b[0] #at least one dimension ?
            return tuple(math2.vecsub(b,a))
        except:
            return (0,0)

    def contiguity(self,pts):
        """
        :return: int number of points from pts already in graph
        """
        res=0
        for pt in pts:
            if pt in self:
                res+=1
        return res

    def closest_nodes(self,p,n=1,skip=False):
        """
        nodes closest to a given position
        :param p: (x,y) position tuple
        :param skip: optional bool to skip n itself
        :return: list of nodes, minimal distance
        """
        if skip: n+=1
        if n==1:
            if p in self.node:
                return [p],0

        res,d=[],None
        for p2 in self.idx.nearest(p, num_results=n, objects='raw'):
            d=self.dist(p, p2)
            if skip and d<=self.tol:
                continue
            res.append(p2)
        return res, d

    def closest_edges(self,p,data=False):
        """:return: container of edges close to p and distance"""
        edges=[]
        n=0
        while not edges and n<10:
            n+=1 #handle more and more nodes until we find one with edges
            nbunch,d=self.closest_nodes(p,n)
            edges=self.edges(nbunch=nbunch,data=data)

        return edges,d

    def add_node(self, p, attr_dict=None, **attr):
        """add a node or return one already very close
        :return (x,y,...) node id
        """
        if p in self.node: #point already exists
            return p

        a={}
        if attr_dict :
            a.update(attr_dict)
        if attr :
            a.update(**attr)

        id=p #save the node id as the position might differ

        if type(p) is not tuple:
            # try to find a position tuple somewhere
            if p in self._map:
                return self._map[p]

            p=a.get('pos',id)
            if isinstance(p,six.string_types):
                p=p.split(',')
            try:
                p=tuple(float(x) for x in p)
            except: # assign a random position, but keep node id
                from random import random
                a['pos']=p=tuple((random(),random()))


        if self.idx is None: #now we know the dimension, so we can create the index
            n=len(p)
            prop=index.Property()
            prop.set_dimension(n)
            self.idx = index.Index(properties=prop)

        close,d=self.closest_nodes(p) #search for those within tolerance
        if close and d<=self.tol:
            return close[0]
        else: # point doesn't exist yet : create it
            #RTREE uses unique int node identifiers
            global _nk
            _nk+=1
            self.idx.insert(_nk,p,p)
            # networkX uses any id type, but the RTREE key is kept
            a['key']=_nk
            self.parent.add_node(self,id, **a)
            #._map links both id
            self._map[id]=p
        return id

    def add_nodes_from(self, nodes, **attr):
        """must be here because Graph.add_nodes_from doesn't call add_node cleanly as it should..."""
        for node in nodes:
            try:
                nn,d=node
                attr.update(d)
            except:
                nn=node
            self.add_node(nn,**attr)

    def remove_node(self,n):
        """
        :param n: node tuple
        remove node from graph and rtree
        """
        n1=self.number_of_nodes()
        n2=self.idx.count(self.idx.bounds)
        if n1!=n2:
            logging.error('GeoGraph has %d!=%d'%(n1,n2))
            raise RuntimeError('Nodes/Rtree mismatch')
        nk=self.node[n]['key']
        self.parent.remove_node(self,n)
        self.idx.delete(nk,n) #in fact n is ignored, the nk key is used here

    def _last(self,u,v):
        """since multigraphs can have several edges from u to v,
        :return: dict of the edge added last
        """
        try:
            edge=self[u][v]
        except: # no edge between u and v yet
            return None #not {}, to make a diff with existing empty dict
        if len(edge)==1: #quick
            return edge[0]
        else:
            return edge[max(edge.keys())]

    def add_edge(self, u, v, k=None, attr_dict=None, **attrs):
        """add an edge to graph
        :return: edge data from created or existing edge
        """

        if type(k) is dict: #old syntax : previous versions of NetworkX didn't require a key
            attr_dict,k=k,None

        #adjust to existing nodes within tolerance and keep track of actual precision
        u=self.add_node(u)
        v=self.add_node(v)

        #attr and kwargs will be merged and copied here.
        #this is important because we want to handle the
        #length parameter separately for each edge
        a={}
        if attr_dict :
            a.update(**attr_dict)
        if attrs :
            a.update(**attrs)

        if not self.is_multigraph():
            try:
                data=self[u][v][0] # 0 because there is only one entry if not multi
            except:
                data=None
            if data:
                data.update(a)
                return data #return existing edge data
            else:
                pass #and add the edge normally below


        #if no length is explicitly defined, we calculate it and set it as parameter
        #therefore all edges in a GeoGraph have a length attribute
        a.setdefault('length',self.dist(u,v))
        if k is None: #try to guess the key
            try:
                keys=set(self[u][v].keys()) #existing keys
                k=0 if len(keys)==0 else None #if k=None, it will be determined below
            except:
                keys=set()
                k=0

        res=self.parent.add_edge(self,u, v, k, **a)

        #Goulib returns the data dict, becasue that's what the most useful
        if isinstance(res,dict):
            return res
        # but NetworkX 1.x returns None and 2.0 returns the ink key...
        # so we have to search the dict object that was just created

        if k is None:
            k=next(iter(set(self[u][v].keys())-keys))
        return self[u][v][k]

    def remove_edge(self,u,v=None,key=None,clean=False):
        """
        :param u: Node or Edge (Nodes tuple)
        :param v: Node if u is a single Node
        :param clean: bool removes disconnected nodes. must be False for certain nx algos to work
        :result: return attributes of removed edge
        remove edge from graph. NetworkX graphs do not remove unused nodes
        """

        if v is None:
            u,v=u[0],u[1]

        if self.is_multigraph():
            if key is None:
                key=self.edge[u][v]
                key=itertools2.first(key)
            data=self.edge[u][v][key]
        else:
            data=self.edge[u][v]
        self.parent.remove_edge(self,u,v,key)

        if clean:
            if self.degree(u)==0:
                self.remove_node(u)
            if self.degree(v)==0:
                self.remove_node(v)
        return data

    def number_of_nodes(self,doublecheck=False):
        #check that both structures are coherent at the same time
        n1=self.parent.number_of_nodes(self)
        if doublecheck:
            n2=self.idx.count(self.idx.bounds) if n1 else 0
            if n1!=n2:
                raise KeyError('%s %s number of nodes %d!=%d'%(self.__class__.__name__, self.name, n1,n2))
        return n1

    def shortest_path(self,source=None,target=None):
        return nx.shortest_path(self,source,target) #TODO : add distance weights

    def stats(self):
        """:return: dict of graph data to use in __repr__ or usable otherwise"""
        res={}
        res['name']=self.name
        res['bbox']=self.box()
        res['size']=self.box_size()
        res['nodes']=self.number_of_nodes()
        res['edges']=self.number_of_edges()
        res['components']=nx.number_connected_components(self)
        res['length']=self.length()
        return res

    def __str__(self):
        """:returns: string representation, used mainly for logging and debugging"""
        return str(self.stats())

    def draw(self,**kwargs):
        """ draw graph with default params"""
        kwargs.setdefault('with_labels',False)
        kwargs.setdefault('node_size',0)

        return draw_networkx(self, **kwargs)

        """ for debug and compatibility check:
        kwargs.pop('edge_color',None)
        return nx.draw_networkx(self, **kwargs)
        """

    def render(self,fmt='svg',**kwargs):
        """ render graph to bitmap stream
        :param fmt: string defining the format. 'svg' by default for INotepads
        :return: matplotlib figure as a byte stream in specified format
        """

        self.render_args.update(kwargs)
        fig=self.draw(**self.render_args)

        from io import BytesIO
        output = BytesIO()
        fig.savefig(output, format=fmt, transparent=kwargs.get('transparent',True))
        res=output.getvalue()
        plt.close(fig)
        return res

    def save(self,filename,**kwargs):
        """ save graph in various formats"""
        ext=filename.split('.')[-1].lower()
        if ext=='dxf':
            write_dxf(self,filename)
        elif ext=='dot':
            write_dot(self, filename)
        elif ext=='json':
            write_json(self,filename,**kwargs)
        else:
            open(filename,'wb').write(self.render(ext,**kwargs))
        return self

class GeoGraph(_Geo, nx.MultiGraph):
    """ Undirected graph with nodes positions
    can be set to non multiedges anytime with attribute multi=False
    """
    def __init__(self, data=None, nodes=None, **kwargs):
        """
        :param data: see :meth:`to_networkx_graph` for valid types
        :param kwargs: other parameters will be copied as attributes, especially:

        """
        properties=kwargs
        try:
            properties.update(data.graph)
        except:
            pass
        properties.setdefault('tol',0.010) #default tolerance on node positions

        nx.MultiGraph.__init__(self,None, **properties)
        _Geo.__init__(self,nx.MultiGraph,data,nodes)

class DiGraph(_Geo, nx.MultiDiGraph):
    """ directed graph with nodes positions
    can be set to non multiedges anytime with attribute multi=False
    """
    def __init__(self, data=None, nodes=None, **kwargs):
        """
        :param data: see :meth:`to_networkx_graph` for valid types
        :param kwargs: other parameters will be copied as attributes, especially:

        """
        properties=kwargs
        try:
            properties.update(data.graph)
        except:
            pass
        properties.setdefault('tol',0.010) #default tolerance on node positions

        nx.MultiDiGraph.__init__(self,None, **properties)
        _Geo.__init__(self,nx.MultiDiGraph,data,nodes)


def figure(g, box=None,**kwargs):
    """
    :param g: _Geo derived Graph
    :param box: optional interval.Box if g has no box
    :return: matplotlib axis suitable for drawing graph g
    """
    fig=plt.figure(**kwargs)
    try:
        box=g.box()
    except:
        pass #use 2nd parameter

    plt.plot(box[0],box[1],alpha=0) #draw a transparent diagonal to size everything
    plt.axis('equal')
    import pylab
    pylab.axis('off') # turn off axis

    return fig

def draw_networkx(g, pos=None, **kwargs):
    """ improves nx.draw_networkx
    :param g: NetworkX Graph
    :param pos: can be either :

    - optional dictionary of (x,y) node positions
    - function of the form lambda node:(x,y) that maps node positions.
    - None. in this case, nodes are directly used as positions if graph is a GeoGraph, otherwise nx.draw_shell is used

    :param **kwargs: passed to nx.draw method as described in http://networkx.lanl.gov/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html with one tweak:

    - if edge_color is a function of the form lambda data:color string, it is mapped over all edges

    """

    #build node positions

    if six.callable(pos): #mapping function
        pos=dict(((node,pos(node)) for node in g.nodes()))

    if pos is None:
        try:
            pos=dict((node,data['pos'][:2]) for node,data in g.nodes(True))
        except KeyError:
            pass

    if pos is None:
        try:
            pos=dict(((node,node[:2]) for node in g.nodes()))
        except TypeError:
            pass

    if pos is None:
        pos=nx.spring_layout(g) # default to spring layout


    try: #convert ndarray to python lists
        for k in pos:
            pos[k]=pos[k].tolist()
    except AttributeError:
        pass

    edgelist=list(kwargs.pop('edgelist',g.edges(data=True)))

    edge_color=kwargs.get('edge_color',None)
    if edge_color is None:
        # build edge_colors
        default=None
        try: # get default edge color
            default=g.color
        except AttributeError:
            pass
        if default is None:
            default='k' #black

        def edge_color(data): #function to color edges, applied below
            c=data.get('color',default)
            return c if c else default

    if six.callable(edge_color): #mapping function ?
        edge_color=list(map(edge_color,(data for u,v,data in edgelist)))

    if edge_color: #not empty
        kwargs['edge_color']=edge_color

    fig=kwargs.pop('fig',None)

    if not fig:
        try: # we need a bounding box
            box=g.box()
        except:
            box=(math2.minimum(pos.values()),math2.maximum(pos.values()))
        fig=figure(g,box=box,figsize=kwargs.get('figsize',None))

    if kwargs.get('node_size',300)>0:
        nx.draw_networkx_nodes(g, pos, **kwargs)

    nx.draw_networkx_edges(g, pos, edgelist, **kwargs)

    labels=kwargs.pop('labels',False) # True in nx.draw_networkx they're
    if labels:
        if labels==True: labels=None #will be set automatically
        if six.callable(labels): #mapping function ?
            labels=list(labels(u) for u in g.nodes(data=True))
        if isinstance(labels,list): # a dict is expected:
            labels=dict(zip(g.nodes(),labels))
        nx.draw_networkx_labels(g, pos, labels, **kwargs)

    return fig

def to_drawing(g, d=None, edges=[]):
    """
    draws Graph to a `Drawing`
    :param g: :class:`Graph`
    :param d: existing :class:`Drawing` to draw onto, or None to create a new Drawing
    :param edges: iterable of edges (with data) that will be added, in the same order. By default all edges are drawn
    :return: :class:`Drawing`

    Graph edges with an 'entity' property
    """
    from . import drawing, geom
    if d is None: d=drawing.Drawing()
    if not edges:
        edges=g.edges(data=True)
    for edge in edges:
        u,v,data=edge[0],edge[1],edge[-1]
        try:
            e=data['entity']
        except KeyError:
            u=g.pos(u)
            v=g.pos(v)
            e=geom.Segment2(u,v)
        d.append(e)
    return d

def write_dxf(g,filename):
    """writes :class:`networkx.Graph` in .dxf format"""
    to_drawing(g).save(filename)

def write_dot(g,filename):
    try:
        import pygraphviz
        from networkx.drawing.nx_agraph import write_dot as _write_dot
        logging.info("using package pygraphviz")
    except ImportError:
        try:
            import pydot
            from networkx.drawing.nx_pydot import write_dot as _write_dot
            logging.info("using package pydot")
        except ImportError:
            logging.error("Both pygraphviz and pydot were not found. See \
                http://networkx.github.io/documentation/latest/reference/drawing.html \
                for info")
            raise

    if isinstance(g, _Geo) or any(map(lambda n:hasattr(n,'pos'),g.nodes())):
        g=g.copy()

    for n in g.nodes(data=True):
        try:
            pos=g.pos(n[0])
        except AttributeError:
            pos=n[1].get('pos',None)
        if pos is not None: # format pos as neato wants it : "x,y"
            n[1]['pos']='"%s,%s"'%pos

    _write_dot(g,filename)

def to_json(g, **kwargs):
    """
    :return: string JSON representation of a graph
    """
    from datetime import datetime, date, time, timedelta
    def json_serial(obj):
        """JSON serializer for objects not serializable by default json code"""
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        try:
            return str(obj)
        except Exception:
            pass
        raise TypeError ("Type %s not serializable"%(type(obj)))
    from networkx.readwrite import json_graph
    g=json_graph.node_link_data(g)
    for node in g['nodes']:
        pos=node.pop('pos',None)
        if pos:
            node['x']=pos[0]
            node['y']=pos[1]
            #node['fixed']=True
    kwargs.setdefault('default',json_serial)
    return json.dumps(g, **kwargs)

def write_json(g,filename, **kwargs):
    """write a JSON file, suitable for D*.js representation
    """
    with open(filename, 'w') as file:
        file.write(to_json(g,**kwargs))


def delauney_triangulation(nodes, qhull_options='', incremental=False, **kwargs):
    """
    https://en.wikipedia.org/wiki/Delaunay_triangulation
    :param nodes: _Geo graph or list of (x,y) or (x,y,z) node positions
    :param qhull_options: string passed to :meth:`scipy.spatial.Delaunay`,
    which passes it to Qhull ( http://www.qhull.org/ )
    *'Qt' ensures all points are connected
    *'Qz' required when nodes lie on a sphere
    *'QJ' solves some singularity situations

    :param kwargs: passed to the :class:`GeoGraph` constructor
    :return: :class:`GeoGraph` with delauney triangulation between nodes
    """
    if isinstance(nodes,_Geo):
        nodes=list(nodes.pos())
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
    points=numpy.array(nodes)
    tri = scipy.spatial.Delaunay(points, qhull_options=qhull_options, incremental=incremental)
    kwargs['multi']=False #to avoid duplicating triangle edges below
    g=GeoGraph(tri,dimension=tri.ndim,**kwargs)
    return g

def euclidean_minimum_spanning_tree(nodes,**kwargs):
    """
    :param nodes: list of (x,y) nodes positions
    :return: :class:`GeoGraph` with minimum spanning tree between nodes

    see https://en.wikipedia.org/wiki/Euclidean_minimum_spanning_tree
    """
    g=GeoGraph(None,**kwargs)
    d=delauney_triangulation(nodes,**kwargs)
    for edge in nx.minimum_spanning_edges(d, weight='length'):
        g.add_edge(*edge)
    return g

# Function to distribute N points on the surface of a sphere
# (source: http://www.softimageblog.com/archives/115)
def points_on_sphere(N):
    pts = []
    inc = math.pi * (3 - math.sqrt(5))
    off = 2 / float(N)
    for k in range(0, int(N)):
        y = k * off - 1 + (off / 2)
        r = math.sqrt(1 - y*y)
        phi = k * inc
        pts.append((math.cos(phi)*r, y, math.sin(phi)*r))
    return pts




