#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
efficient Euclidian Graphs for NetworkX and related algorithms
"""

from __future__ import division #"true division" everywhere

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2014, Philippe Guglielmetti"
__credits__ = []
__license__ = "LGPL"

import logging
import math

import networkx as nx # http://networkx.github.io/
from rtree import index # http://toblerity.org/rtree/
import numpy, scipy.spatial
import matplotlib.pyplot as plt

import Goulib.math2 as math2

_nk=0 # node key

class GeoGraph(nx.MultiGraph):
    """a Graph with nodes at specified positions.
    if edges have a "length" attribute, it is used to compute distances,
    otherwise euclidian distance between nodes is used by default
    """
    def __init__(self, data=None, **kwargs):
        """
        :param multi: boolean to control if multiple edges between nodes are allowed (True) or not (False)
        :param **kwargs: other parameters will be copied as attributes
        """
        super(GeoGraph,self).__init__(None, **kwargs)
        
        #create a rtree and store nodes in it
        n=kwargs.get('dimension',None)
        if n is None:
            try: #guess dimension from first node
                n=len(data.node.iterkeys().next())
            except:
                n=2
        prop=index.Property()
        prop.set_dimension(n)
        self.idx = index.Index(properties=prop)
        
        if data:
            if isinstance(data,scipy.spatial.qhull.Delaunay):
                self.delauney=data
                triangles=data.points[data.simplices]
                for point in triangles:
                    p=map(tuple,point) #convert from numpy to regular list
                    self.add_edge(p[0],p[1])
                    self.add_edge(p[1],p[2])
                    self.add_edge(p[2],p[0])
            elif isinstance(data,nx.Graph): # pass only the adjacency matrix to ensure node keys aren't trashed in to_networkx_graph
                nx.convert.to_networkx_graph(data.adj,create_using=self,multigraph_input=data.is_multigraph())
            else:
                nx.convert.to_networkx_graph(data,create_using=self)
        pass
        
            
    def copy(self):
        """does not use deepcopy because the rtree index must be rebuilt"""
        return self.__class__(self,**self.graph)
                
    def __nonzero__(self):
        """:return: True if graph has at least one node
        """
        return self.number_of_nodes()>0
    
    def clear(self): 
        #saves some graph attributes cleared by convert._prep_create_using
        t,m=self.tol,self.multi
        super(GeoGraph,self).clear()
        self.multi=m
        self.graph['tol']=t
    
    @property
    def tol(self):
        return self.graph.get('tol',0.010)
    
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
        except:
            return True
            
        
    def dist(self,u,v):
        """
        :return: float distance between nodes u and v
        """
        
        try:
            return self[u][v]['length'] 
        except: 
            return math2.dist(u,v)
                
    def length(self,edges=None):
        """:returns: sum of 'length' attributes of edges"""
        
        if edges is None: # take all edges
            return self.size(weight="length")
            
        res=0
        for edge in edges:
            try:
                l=edge[2]['length']
            except:
                l=self.dist(edge[0],edge[1])
                edge[2]['length']=l
            res+=l
        return res
    
    def box(self):
        """:return: nodes bounding box as (xmin,ymin,...),(xmax,ymax,...)"""
        minmax=self.idx.bounds
        n=len(minmax)//2
        return tuple(minmax[:n]),tuple(minmax[n:])
        
    def box_size(self):
        """:return: (x,y) size"""
        min,max=self.box()
        try:
            min[0],max[0] #at least one dimension ?
            return tuple(math2.vecsub(max,min))
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
            try: # does it already exist ?
                self.node[p]
                return [p],0
            except:
                pass
        
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
        """
        try: #point already exists ?
            self.node[p]
            return p
        except:
            pass
        
        close,d=self.closest_nodes(p) #search for those within tolerance
        if close and d<=self.tol:
            return close[0]
        else: # point doesn't exist yet : create it
            global _nk
            _nk+=1
            attr['key']=_nk
            super(GeoGraph,self).add_node(p, attr_dict, **attr)
            self.idx.insert(_nk,p,p)
        return p
    
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
        super(GeoGraph,self).remove_node(n)  
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

    def add_edge(self, u, v, attr_dict={}, **attrs):
        """add an edge to graph
        :return: edge data from created or existing edge
        side effect: updates largest merge in self.max_merge
        """
        
        #attr and kwargs will be merged and copied here. 
        #this is important because we want to handle the 
        #length parameter separately for each edge
        a={} 
        if attr_dict : 
            a.update(**attr_dict)
        if attrs :
            a.update(**attrs)
        
        if not self.is_multigraph():
            data=self._last(u,v)
            if data:
                data.update(a)
                return data #return existing edge data
            else:
                pass #and add the edge normally below
            
        #adjust to existing nodes within tolerance and keep track of actual precision
        u=self.add_node(u)
        v=self.add_node(v)
        
        #if no length is explicitely defined, we calculate it and set it as parameter
        #therefore all edges in a GeoGraph have a length attribute
        a.setdefault('length',self.dist(u,v))
        super(GeoGraph,self).add_edge(u, v, attr_dict=a) #doesn't return created data... what a pity ...
        return self._last(u,v)
        
    def remove_edge(self,u,v=None,key=None,clean=False):
        """
        :param u: Node or Edge (Nodes tuple)
        :param v: Node if u is a single Node
        :param clean: bool removes disconnected nodes. must be False for certain nx algos to work
        remove edge from graph. NetworkX graphs do not remove unused nodes
        """
        
        if v is None:
            u,v=u[0],u[1]
        
        super(GeoGraph,self).remove_edge(u,v,key)   
        
        if clean:
            if self.degree(u)==0:
                self.remove_node(u)
            if self.degree(v)==0:
                self.remove_node(v)
                
    def number_of_nodes(self):
        #check that both structures are coherent at the same time
        n1=super(GeoGraph,self).number_of_nodes()   
        n2=self.idx.count(self.idx.bounds) if n1 else 0
        if n1!=n2:
            raise KeyError('%s %s number of nodes %d!=%d'%(self.__class__.__name__, self.name, n1,n2))
        return n1
    
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
    
    def render(self,format,**kwargs):
        """ render graph to bitmap stream
        :return: matplotlib figure as a byte stream in specified format
        """
        
        fig=self.draw(**kwargs)['fig']
        
        from io import BytesIO
        output = BytesIO()
        fig.savefig(output, format=format, transparent=kwargs.get('transparent',True))
        plt.close(fig)
        return output.getvalue()
    
    # for IPython notebooks
    def _repr_svg_(self): return self.render('svg')
    def _repr_png_(self): return self.render('png')
    
    def save(self,filename,**kwargs):
        """ save graph in various formats"""
        ext=filename.split('.')[-1].lower()
        open(filename,'wb').write(self.render(ext,**kwargs))
    
def figure(g,**kwargs):
    """:return: matplotlib axis suitable for drawing graph g"""
    fig=plt.figure(**kwargs)
    min,max=g.box()
    plt.plot((min[0],max[0]),(min[1],max[1]),alpha=0) #draw a transparent diagonal to size everything
    plt.axis('equal')
    
    import pylab
    pylab.axis('off') # turn off axis
    
    return fig

def draw_networkx(g, **kwargs):
    """ improves draw_networkx 
    :param g: NetworkX Graph
    :param pos: can be either :
    
    - optional dictionary of (x,y) node positions
    - function of the form lambda node:(x,y) that maps node positions.
    - None. in this case, nodes are directly used as positions if graph is a GeoGraph, otherwise nx.draw_shell is used
    
    :param fig: matplotlib figure. If None, figure(g) is used to obtain one
    :param **kwargs: passed to nx.draw method with one tweak:
    
    - if edge_color is a function of the form lambda data:color string, it is mapped over all edges
    """
    
    #build node positions
    pos=kwargs.get('pos',None)
    if pos is None and isinstance(g,GeoGraph):
        pos={} #dict of nodes positions
        for node in g.nodes_iter():
            pos[node]=node
    elif pos and callable(pos): #mapping function ?
            pos=dict(((node,pos(node)) for node in g.nodes_iter()))
            
    if not pos:
        pos=nx.spring_layout(g) # default to spring layout
        
    edgelist=kwargs.setdefault('edgelist',g.edges(data=True))
    
    # build edge_colors
    edge_color=kwargs.get('edge_color',None)
    if edge_color:
        if callable(edge_color): #mapping function ?
            edge_color=map(edge_color,(data for u,v,data in edgelist))
    else: #try to color the graph
        edge_color=[]
        for u,v,data in edgelist:
            if 'color' in data:
                edge_color.append(data['color'])
            else:
                try:
                    edge_color.append(g.color)
                except:
                    pass
    if edge_color: #not empty
        kwargs['edge_color']=edge_color
    
    if not 'fig' in kwargs:
        kwargs['fig']=figure(g)
    
    if kwargs.get('node_size',300)>0:
        nx.draw_networkx_nodes(g, pos, **kwargs)
        
    nx.draw_networkx_edges(g, pos, **kwargs)
    if kwargs.get('with_labels',False):
        nx.draw_networkx_labels(g, pos, **kwargs)
        
    return kwargs
    
def delauney_triangulation(nodes, qhull_options='', incremental=False, **kwargs):
    """
    :param nodes: list of (x,y) or (x,y,z) node positions
    :param qhull_options: string passed to scipy.spatial.Delaunay, which passes it to Qhull ( http://www.qhull.org/ )
        * 'Qt' ensures all points are connected
        * 'Qz' required when nodes lie on a sphere
        * 'QJ' solves some singularity situations
    :param **kwargs: passed to the GeoGraph constructor
    :return: GeoGraph with delauney triangulation between nodes
    see https://en.wikipedia.org/wiki/Delaunay_triangulation
    """
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
    points=numpy.array(nodes)
    tri = scipy.spatial.Delaunay(points, qhull_options=qhull_options, incremental=incremental) 
    kwargs['multi']=False #to avoid duplicating triangle edges below
    g=GeoGraph(tri,dimension=tri.ndim,**kwargs)
    return g

def euclidean_minimum_spanning_tree(nodes,**kwargs):
    """
    :param nodes: list of (x,y) nodes positions
    :return: geograph with minimum spanning tree between  nodes
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
        
        
    
    
