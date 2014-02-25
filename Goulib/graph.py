#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Euclidian Graphs and additional NetworkX algosrithms
"""

from __future__ import division #"true division" everywhere

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2014, Philippe Guglielmetti"
__credits__ = []
__license__ = "LGPL"

import logging

import networkx as nx # http://networkx.github.io/
from rtree import index # http://toblerity.org/rtree/

import Goulib.math2 as math2
import Goulib.itertools2 as itertools2

_nk=0 # node key

class GeoGraph(nx.MultiGraph):
    """a Graph with nodes at specified positions.
    if edges have a "length" attribute, it is used to compute distances,
    otherwise euclidian distance between nodes is used by default
    """
    def __init__(self, G=None, **kwargs):
        """
        :param multi: boolean to control if multiple edges between nodes are allowed (True) or not (False)
        :param **kwargs: other parameters will be copied as attributes
        """
        
        super(GeoGraph,self).__init__(None,**kwargs)
        prop=index.Property()
        try: #guess dimension from first node
            n=len(G.node.iterkeys().next())
        except:
            n=2
        prop.set_dimension(n)
        self.idx = index.Index(properties=prop)
        if G:
            self.add_edges_from(G.edges(data=True))
            
    def copy(self):
        """does not use deepcopy because the rtree index must be rebuilt"""
        return self.__class__(self,**self.graph)
                
    def __nonzero__(self):
        """:return: True if graph has at least one node
        """
        return self.number_of_nodes()>0
    
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
        n=self.nodes()
        return tuple(math2.minimum(n)), tuple(math2.maximum(n))
        
    def box_size(self):
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
        :param skip: optional bool to skip n
        :return: list of nodes
        """
        if skip: n+=1
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
            n+=1 #handle more and more nodes untif we find one with edges
            nbunch,d=self.closest_nodes(p,n)
            edges=self.edges(nbunch=nbunch,data=data)
            
        return edges,d
        
    def add_node(self, n, attr_dict=None, **attr):
        """add a node or return one already very close
        """
        close,d=self.closest_nodes(n)
        try:
            node=close[0]
        except: # point doesn't exist yet
            node=None
            
        if node and d<=self.tol:
            self.max_merge=max(d,self.__dict__.get('max_merge',0))
        else:
            global _nk
            _nk+=1
            attr['key']=_nk
            super(GeoGraph,self).add_node(n, attr_dict, **attr)
            node=n
            #p=_smallbox(n,self.tol)
            self.idx.insert(_nk,node,node)
        return node
    
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
        self.idx.delete(nk,n) 
        
        self.number_of_nodes() #debug check

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
        
    def remove_edge(self,u,v=None,clean=False):
        """
        :param u: Node or Edge (Nodes tuple)
        :param v: Node if u is a single Node
        :param clean: bool removes disconnected nodes. must be False for certain nx algos to work
        remove edge from graph. NetworkX graphs do not remove unused nodes
        """
        
        if v is None:
            u,v=u[0],u[1]
        
        super(GeoGraph,self).remove_edge(u,v)   
        
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
    
    def render(self,format='png',**kwargs):
        kwargs.setdefault('with_labels',False)
        kwargs.setdefault('node_size',0)
        pos={} #dict of nodes positions
        for node in self.nodes_iter():
            pos[node]=node
        return render(self,pos,format=format,**kwargs)
    
    # for IPython notebooks
    def _repr_svg_(self): return self.render('svg')
    def _repr_png_(self): return self.render('png')
    
def render(g, pos=None,format='svg',**kwargs):
    """return a matplotlib figure generated from a graph"""
    import matplotlib.pyplot as plt
    from io import BytesIO
    size=g.box_size()

    fig=plt.figure(figsize=(24,16))
    ax = fig.add_subplot(111)
    if not pos:
        kwargs['ax']=ax
        nx.draw_shell(g, **kwargs)
    else:
        nx.draw_networkx(g, ax=ax, pos=pos, **kwargs)
    
    import pylab
    pylab.axis('off') # turn of axis

    output = BytesIO()
    fig.savefig(output, format=format, transparent=kwargs.get('transparent',True))
    plt.close(fig)
    return output.getvalue()
    
def delauney_triangulation(nodes,**kwargs):
    """
    :param nodes: list of (x,y) nodes positions
    :return: geograph with minimum spanning tree between  nodes
    see https://en.wikipedia.org/wiki/Delaunay_triangulation
    """
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
    import numpy as np
    from scipy.spatial import Delaunay
    points=np.array(nodes)
    tri = Delaunay(points, qhull_options='Qt') #option ensures all points are connected
    kwargs['multi']=False #to avoid duplicating triangle edges below
    g=GeoGraph(None,**kwargs)
    triangles=points[tri.simplices]
    for point in triangles:
        p=map(tuple,point) #convert from numpy to regular list
        g.add_edge(p[0],p[1])
        g.add_edge(p[1],p[2])
        g.add_edge(p[2],p[0])
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
        
        
    
    
