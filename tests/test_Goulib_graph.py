#!/usr/bin/python
# -*- coding: utf-8 -*-

from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *

from Goulib.graph import *

import logging

import os
path=os.path.dirname(os.path.abspath(__file__))

class TestGeoGraph:
    @classmethod
    def setup_class(self):
        self.empty=GeoGraph()
        self.cube=GeoGraph(nx.hypercube_graph(3),multi=False)
        self.geo=GeoGraph(nx.random_geometric_graph(50,.25))
        
        nodes=points_on_sphere(50)
        self.sphere=GeoGraph(nodes=nodes) #test if we can construct from nodes only
        self.sphere=delauney_triangulation(nodes,'Qz',tol=0) #'Qz' required for spheres
        
        try:
            from pygraphviz import AGraph
        except:
            self.crazy=None
            return #skip AGraph tests as pygraphviz is an optional requirement
        a=AGraph(path+'/crazy.dot')
        self.crazy=GeoGraph(a)
                    
    def test_save(self):
        
        #complex AGraph
        if self.crazy:
            self.crazy.save(path+'/crazy.png', transparent=False)
        
        import matplotlib.pyplot as plt
        #define a function that maps edge data to a color
        m=plt.get_cmap('Blues')
        def edge_color(data): #make longer links darker
            return m(data['length']/.25)
        self.geo.save(path+'/graph.png', transparent=False, edge_color=edge_color, node_size=50)
        
        #3D graph
        self.sphere.save(path+'/sphere.png', transparent=False)
        
        
    def test_render(self):
        pass #tested above
    
    def test_is_multigraph(self):
        assert_false(self.cube.is_multigraph())
        
    def test___init__(self):
        assert_equal(self.cube.number_of_nodes(),8)
        assert_equal(self.cube.number_of_edges(),12)

    def test___nonzero__(self):
        assert_true(bool(self.cube))
        assert_false(bool(self.empty))
        
    def test_length(self):
        assert_equal(self.cube.length(),12)
        
    def test_multi(self):
        pass #tested below

    def test_multi_case_2(self):
        pass #tested below
        
    def test_add_edge(self):
        g=self.cube.copy()
        assert_equal(g.length(),12)
        edge=g.add_edge((0,0,0),(1,1,1),length=3)
        assert_equal(g.number_of_edges(),13)
        assert_equal(g.length(),15)
        #try recreating the same edge when multi is False
        g.multi=False
        assert_equal(g.number_of_edges(),13)
        edge=g.add_edge((0,0,0),(1,1,1),length=2) #it should only change the attribute
        
        assert_equal(g.number_of_edges(),13)
        assert_equal(g.length(),14)
        #try recreating the same edge when multi is False
        g.multi=True
        edge=g.add_edge((0,0,0),(1,1,1),length=3) #now this one should be added
        assert_equal(edge['length'],3)
        assert_equal(g.number_of_edges(),14)
        assert_equal(g.length(),17)
        
    def test_remove_edge(self):
        g=self.cube.copy()
        assert_equal(g.number_of_nodes(),8)
        assert_equal(g.number_of_edges(),12)
        
        for edge in g.edges((0,0,0)):
            g.remove_edge(edge,clean=True)
            
        assert_equal(g.number_of_nodes(),7)
        assert_equal(g.number_of_edges(),9)
        
    def test_closest_nodes(self):
        close,d=self.cube.closest_nodes((0,0,0))
        assert_equal(d,0)
        assert_equal(close,[(0,0,0)])
        close,d=self.cube.closest_nodes((0,0,0),skip=True)
        assert_equal(d,1)
        assert_equal(len(close),3)
        close,d=self.cube.closest_nodes((0.5,0.5,0))
        assert_equal(len(close),4)
        
    def test_remove_node(self):
        g=self.cube.copy()
        assert_equal(g.number_of_nodes(),8)
        assert_equal(g.number_of_edges(),12)
        
        g.remove_node((0,0,0))
        
        assert_equal(g.number_of_nodes(),7)
        assert_equal(g.number_of_edges(),9)
        return g 
        
    def test_closest_edges(self):
        close,d=self.cube.closest_edges((0,0,0))
        assert_equal(d,0)
        assert_equal(len(close),3)
        
        g=self.test_remove_node()
        close,d=g.closest_edges((0,0,0))
        assert_equal(d,1)
        assert_equal(len(close),6)

    def test_box(self):
        assert_equal(self.cube.box(),((0,0,0),(1,1,1)))

    def test_box_size(self):
        assert_equal(self.cube.box_size(),(1,1,1))

    def test_stats(self):
        stats=self.cube.stats()
        assert_equal(stats['nodes'],8)
        assert_equal(stats['edges'],12)
    
    def test_str(self):
        s=str(self.empty)
        assert_true("'nodes': 8" in str(self.cube))

    def test_dist(self):
        import math
        assert_equal(self.cube.dist((0,0,0), (1,1,1)),math.sqrt(3))

    def test_contiguity(self):
        # geo_graph = GeoGraph(G, multi, **kwargs)
        # assert_equal(expected, geo_graph.contiguity(pt1, pt2))
        raise SkipTest 

    def test_tol(self):
        # geo_graph = GeoGraph(G, **kwargs)
        # assert_equal(expected, geo_graph.tol())
        raise SkipTest 

    def test___str__(self):
        # geo_graph = GeoGraph(G, **kwargs)
        # assert_equal(expected, geo_graph.__str__())
        raise SkipTest 

    def test_add_node(self):
        # geo_graph = GeoGraph(G, **kwargs)
        # assert_equal(expected, geo_graph.add_node(n, attr_dict, **attr))
        raise SkipTest 

    def test_add_nodes_from(self):
        # geo_graph = GeoGraph(G, **kwargs)
        # assert_equal(expected, geo_graph.add_nodes_from(nodes, **attr))
        raise SkipTest 

    def test_copy(self):
        # geo_graph = GeoGraph(G, **kwargs)
        # assert_equal(expected, geo_graph.copy())
        raise SkipTest 

    def test_number_of_nodes(self):
        # geo_graph = GeoGraph(G, **kwargs)
        # assert_equal(expected, geo_graph.number_of_nodes())
        raise SkipTest 

    def test_draw(self):
        # geo_graph = GeoGraph(G, **kwargs)
        # assert_equal(expected, geo_graph.draw(**kwargs))
        raise SkipTest 

    def test_clear(self):
        # geo_graph = GeoGraph(data, **kwargs)
        # assert_equal(expected, geo_graph.clear())
        raise SkipTest 

    def test___bool__(self):
        # geo_graph = GeoGraph(data, nodes, **kwargs)
        # assert_equal(expected, geo_graph.__bool__())
        raise SkipTest 

class TestRender:
    def test_render(self):
        pass # tested in test_save TODO : more tests with attributes

class TestDelauneyTriangulation:
    def test_delauney_triangulation(self):
        import time
        n=1000 if RTREE else 100
        from random import random
        start=time.clock()
        nodes=[(random(),random()) for _ in range(n)]
        graph=delauney_triangulation(nodes, tol=0)
        logging.info('Delauney %d : %f'%(n,time.clock()-start))
        assert_equal(graph.number_of_nodes(),n)
        assert_true(nx.is_connected(graph))
        graph.save(path+'/delauney.png')
        start=time.clock()
        graph=euclidean_minimum_spanning_tree(nodes)
        logging.info('Spanning tree %d : %f'%(n,time.clock()-start))
        graph.save(path+'/emst.png')

class TestEuclideanMinimumSpanningTree:
    def test_euclidean_minimum_spanning_tree(self):
        pass #tested together with Delauney triangulation
    
  
class TestFigure:
    def test_figure(self):
        pass #tested above

class TestDraw:
    def test_draw(self):
        pass #tested above


class TestDrawNetworkx:
    def test_draw_networkx(self):
        g=nx.gn_graph(10) #generate a DiGraph
        draw_networkx(g)

class TestPointsOnSphere:
    def test_points_on_sphere(self):
        pass

class TestToDrawing:
    def test_to_drawing(self):
        # assert_equal(expected, to_drawing(g, d, edges))
        raise SkipTest 

class TestWriteDxf:
    def test_write_dxf(self):
        # assert_equal(expected, write_dxf(g, filename))
        raise SkipTest 

class TestToNetworkxGraph:
    def test_to_networkx_graph(self):
        # assert_equal(expected, to_networkx_graph(data, create_using, multigraph_input))
        raise SkipTest 

class test__Geo:
    def test___getattr__(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.__getattr__(name))
        raise SkipTest 

    def test___init__(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        raise SkipTest 

    def test___nonzero__(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.__nonzero__())
        raise SkipTest 

    def test___str__(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.__str__())
        raise SkipTest 

    def test_add_edge(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.add_edge(u, v, attr_dict, **attrs))
        raise SkipTest 

    def test_add_node(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.add_node(p, attr_dict, **attr))
        raise SkipTest 

    def test_add_nodes_from(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.add_nodes_from(nodes, **attr))
        raise SkipTest 

    def test_box(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.box())
        raise SkipTest 

    def test_box_size(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.box_size())
        raise SkipTest 

    def test_clear(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.clear())
        raise SkipTest 

    def test_closest_edges(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.closest_edges(p, data))
        raise SkipTest 

    def test_closest_nodes(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.closest_nodes(p, n, skip))
        raise SkipTest 

    def test_contiguity(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.contiguity(pts))
        raise SkipTest 

    def test_copy(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.copy())
        raise SkipTest 

    def test_dist(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.dist(u, v))
        raise SkipTest 

    def test_draw(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.draw(**kwargs))
        raise SkipTest 

    def test_is_multigraph(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.is_multigraph())
        raise SkipTest 

    def test_length(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.length(edges))
        raise SkipTest 

    def test_multi(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.multi())
        raise SkipTest 

    def test_multi_case_2(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.multi(s))
        raise SkipTest 

    def test_number_of_nodes(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.number_of_nodes())
        raise SkipTest 

    def test_remove_edge(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.remove_edge(u, v, key, clean))
        raise SkipTest 

    def test_remove_node(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.remove_node(n))
        raise SkipTest 

    def test_render(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.render(format, **kwargs))
        raise SkipTest 

    def test_save(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.save(filename, **kwargs))
        raise SkipTest 

    def test_stats(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.stats())
        raise SkipTest 

    def test_tol(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.tol())
        raise SkipTest 

class test__Geo:
    def test___getattr__(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.__getattr__(name))
        raise SkipTest 

    def test___init__(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        raise SkipTest 

    def test___nonzero__(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.__nonzero__())
        raise SkipTest 

    def test___str__(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.__str__())
        raise SkipTest 

    def test_add_edge(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.add_edge(u, v, attr_dict, **attrs))
        raise SkipTest 

    def test_add_node(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.add_node(p, attr_dict, **attr))
        raise SkipTest 

    def test_add_nodes_from(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.add_nodes_from(nodes, **attr))
        raise SkipTest 

    def test_box(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.box())
        raise SkipTest 

    def test_box_size(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.box_size())
        raise SkipTest 

    def test_clear(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.clear())
        raise SkipTest 

    def test_closest_edges(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.closest_edges(p, data))
        raise SkipTest 

    def test_closest_nodes(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.closest_nodes(p, n, skip))
        raise SkipTest 

    def test_contiguity(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.contiguity(pts))
        raise SkipTest 

    def test_copy(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.copy())
        raise SkipTest 

    def test_dist(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.dist(u, v))
        raise SkipTest 

    def test_draw(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.draw(**kwargs))
        raise SkipTest 

    def test_is_multigraph(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.is_multigraph())
        raise SkipTest 

    def test_length(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.length(edges))
        raise SkipTest 

    def test_multi(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.multi())
        raise SkipTest 

    def test_multi_case_2(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.multi(s))
        raise SkipTest 

    def test_number_of_nodes(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.number_of_nodes())
        raise SkipTest 

    def test_remove_edge(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.remove_edge(u, v, key, clean))
        raise SkipTest 

    def test_remove_node(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.remove_node(n))
        raise SkipTest 

    def test_render(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.render(format, **kwargs))
        raise SkipTest 

    def test_save(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.save(filename, **kwargs))
        raise SkipTest 

    def test_stats(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.stats())
        raise SkipTest 

    def test_tol(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.tol())
        raise SkipTest 

class test__Geo:
    def test___getattr__(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.__getattr__(name))
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        raise SkipTest # TODO: implement your test here

    def test___nonzero__(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.__nonzero__())
        raise SkipTest # TODO: implement your test here

    def test___str__(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.__str__())
        raise SkipTest # TODO: implement your test here

    def test_add_edge(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.add_edge(u, v, attr_dict, **attrs))
        raise SkipTest # TODO: implement your test here

    def test_add_node(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.add_node(p, attr_dict, **attr))
        raise SkipTest # TODO: implement your test here

    def test_add_nodes_from(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.add_nodes_from(nodes, **attr))
        raise SkipTest # TODO: implement your test here

    def test_box(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.box())
        raise SkipTest # TODO: implement your test here

    def test_box_size(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.box_size())
        raise SkipTest # TODO: implement your test here

    def test_clear(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.clear())
        raise SkipTest # TODO: implement your test here

    def test_closest_edges(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.closest_edges(p, data))
        raise SkipTest # TODO: implement your test here

    def test_closest_nodes(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.closest_nodes(p, n, skip))
        raise SkipTest # TODO: implement your test here

    def test_contiguity(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.contiguity(pts))
        raise SkipTest # TODO: implement your test here

    def test_copy(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.copy())
        raise SkipTest # TODO: implement your test here

    def test_dist(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.dist(u, v))
        raise SkipTest # TODO: implement your test here

    def test_draw(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.draw(**kwargs))
        raise SkipTest # TODO: implement your test here

    def test_is_multigraph(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.is_multigraph())
        raise SkipTest # TODO: implement your test here

    def test_length(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.length(edges))
        raise SkipTest # TODO: implement your test here

    def test_multi(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.multi())
        raise SkipTest # TODO: implement your test here

    def test_multi_case_2(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.multi(s))
        raise SkipTest # TODO: implement your test here

    def test_number_of_nodes(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.number_of_nodes(doublecheck))
        raise SkipTest # TODO: implement your test here

    def test_remove_edge(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.remove_edge(u, v, key, clean))
        raise SkipTest # TODO: implement your test here

    def test_remove_node(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.remove_node(n))
        raise SkipTest # TODO: implement your test here

    def test_render(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.render(fmt, **kwargs))
        raise SkipTest # TODO: implement your test here

    def test_save(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.save(filename, **kwargs))
        raise SkipTest # TODO: implement your test here

    def test_stats(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.stats())
        raise SkipTest # TODO: implement your test here

    def test_tol(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.tol())
        raise SkipTest # TODO: implement your test here

class test__Geo:
    def test___getattr__(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.__getattr__(name))
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        raise SkipTest # TODO: implement your test here

    def test___nonzero__(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.__nonzero__())
        raise SkipTest # TODO: implement your test here

    def test___str__(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.__str__())
        raise SkipTest # TODO: implement your test here

    def test_add_edge(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.add_edge(u, v, attr_dict, **attrs))
        raise SkipTest # TODO: implement your test here

    def test_add_node(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.add_node(p, attr_dict, **attr))
        raise SkipTest # TODO: implement your test here

    def test_add_nodes_from(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.add_nodes_from(nodes, **attr))
        raise SkipTest # TODO: implement your test here

    def test_box(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.box())
        raise SkipTest # TODO: implement your test here

    def test_box_size(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.box_size())
        raise SkipTest # TODO: implement your test here

    def test_clear(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.clear())
        raise SkipTest # TODO: implement your test here

    def test_closest_edges(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.closest_edges(p, data))
        raise SkipTest # TODO: implement your test here

    def test_closest_nodes(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.closest_nodes(p, n, skip))
        raise SkipTest # TODO: implement your test here

    def test_contiguity(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.contiguity(pts))
        raise SkipTest # TODO: implement your test here

    def test_copy(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.copy())
        raise SkipTest # TODO: implement your test here

    def test_dist(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.dist(u, v))
        raise SkipTest # TODO: implement your test here

    def test_draw(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.draw(**kwargs))
        raise SkipTest # TODO: implement your test here

    def test_is_multigraph(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.is_multigraph())
        raise SkipTest # TODO: implement your test here

    def test_length(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.length(edges))
        raise SkipTest # TODO: implement your test here

    def test_multi(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.multi())
        raise SkipTest # TODO: implement your test here

    def test_multi_case_2(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.multi(s))
        raise SkipTest # TODO: implement your test here

    def test_number_of_nodes(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.number_of_nodes(doublecheck))
        raise SkipTest # TODO: implement your test here

    def test_remove_edge(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.remove_edge(u, v, key, clean))
        raise SkipTest # TODO: implement your test here

    def test_remove_node(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.remove_node(n))
        raise SkipTest # TODO: implement your test here

    def test_render(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.render(fmt, **kwargs))
        raise SkipTest # TODO: implement your test here

    def test_save(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.save(filename, **kwargs))
        raise SkipTest # TODO: implement your test here

    def test_stats(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.stats())
        raise SkipTest # TODO: implement your test here

    def test_tol(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.tol())
        raise SkipTest # TODO: implement your test here

class test__Geo:
    def test___getattr__(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.__getattr__(name))
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        raise SkipTest # TODO: implement your test here

    def test___nonzero__(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.__nonzero__())
        raise SkipTest # TODO: implement your test here

    def test___str__(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.__str__())
        raise SkipTest # TODO: implement your test here

    def test_add_edge(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.add_edge(u, v, attr_dict, **attrs))
        raise SkipTest # TODO: implement your test here

    def test_add_node(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.add_node(p, attr_dict, **attr))
        raise SkipTest # TODO: implement your test here

    def test_add_nodes_from(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.add_nodes_from(nodes, **attr))
        raise SkipTest # TODO: implement your test here

    def test_box(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.box())
        raise SkipTest # TODO: implement your test here

    def test_box_size(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.box_size())
        raise SkipTest # TODO: implement your test here

    def test_clear(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.clear())
        raise SkipTest # TODO: implement your test here

    def test_closest_edges(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.closest_edges(p, data))
        raise SkipTest # TODO: implement your test here

    def test_closest_nodes(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.closest_nodes(p, n, skip))
        raise SkipTest # TODO: implement your test here

    def test_contiguity(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.contiguity(pts))
        raise SkipTest # TODO: implement your test here

    def test_copy(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.copy())
        raise SkipTest # TODO: implement your test here

    def test_dist(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.dist(u, v))
        raise SkipTest # TODO: implement your test here

    def test_draw(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.draw(**kwargs))
        raise SkipTest # TODO: implement your test here

    def test_is_multigraph(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.is_multigraph())
        raise SkipTest # TODO: implement your test here

    def test_length(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.length(edges))
        raise SkipTest # TODO: implement your test here

    def test_multi(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.multi())
        raise SkipTest # TODO: implement your test here

    def test_multi_case_2(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.multi(s))
        raise SkipTest # TODO: implement your test here

    def test_number_of_nodes(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.number_of_nodes(doublecheck))
        raise SkipTest # TODO: implement your test here

    def test_remove_edge(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.remove_edge(u, v, key, clean))
        raise SkipTest # TODO: implement your test here

    def test_remove_node(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.remove_node(n))
        raise SkipTest # TODO: implement your test here

    def test_render(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.render(fmt, **kwargs))
        raise SkipTest # TODO: implement your test here

    def test_save(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.save(filename, **kwargs))
        raise SkipTest # TODO: implement your test here

    def test_stats(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.stats())
        raise SkipTest # TODO: implement your test here

    def test_tol(self):
        # __geo = _Geo(parent, data, nodes, **kwargs)
        # assert_equal(expected, __geo.tol())
        raise SkipTest # TODO: implement your test here

if __name__=="__main__":
    runmodule()