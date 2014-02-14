from nose.tools import assert_equal, assert_true, assert_false, assert_almost_equal, raises
from nose import SkipTest

from Goulib.graph import *

class TestGeoGraph:
    @classmethod
    def setup_class(self):
        self.empty=GeoGraph()
        self.cube=GeoGraph(nx.hypercube_graph(3),multi=False)

        self.logi=GeoGraph(multi=False)
        nodes={}
        from Goulib.table import Table
        for line in Table('localisation.csv'):
            name=line[0]
            pos=tuple([line[3],-line[4]]) #flip Y to keep North North ...
            nodes[name]=pos

        mat=Table('matrice.csv')
        for line in mat:
            end=nodes[line[0]]
            for i,v in enumerate(line[2:]):
                if v:
                    start=nodes[mat.titles[i+2]]
                    try:
                        edge=self.logi[start][end]
                        w=edge[0]['weight']
                        pass
                    except:
                        w=0
                    self.logi.add_edge(start,end,weight=v+w)
                    
    def test_render(self):
        from Goulib.colors import hex_to_rgb
        import matplotlib.pyplot as plt
        edge_color=[]
        colors={'red':'#C3001E','green':'#1DC300','blue':'#265896','lightblue':'#6598CA','gray':'#585858'}
        map=plt.get_cmap('Greys')
        for u,v,data in self.logi.edges(data=True):
            c=data['weight']
            if c>100:
                c=hex_to_rgb(colors['red'],scale=1./255)
            else:
                c=map(float(c)/100)
            edge_color.append(c)
            
        
        open('logi.png','wb').write(self.logi.render(transparent=False, edge_color=edge_color, node_size=100, node_color=colors['lightblue']))
        
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
        for edge in g.edges((0,0,0)):
            g.remove_edge(edge,clean=True)
        assert_equal(g.number_of_nodes(),7)

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
        assert_true("'bbox': ((), ())" in s)
        assert_true("'nodes': 8" in str(self.cube))

    def test_dist(self):
        import math
        assert_equal(self.cube.dist((0,0,0), (1,1,1)),math.sqrt(3))

    def test_closest_nodes(self):
        close,d=self.cube.closest_nodes((0,0,0))
        assert_equal(d,0)
        assert_equal(close,[(0,0,0)])
        close,d=self.cube.closest_nodes((0,0,0),skip=True)
        assert_equal(d,1)
        assert_equal(len(close),3)
        close,d=self.cube.closest_nodes((0.5,0.5,0))
        assert_equal(len(close),4)
        
    def test_closest_edges(self):
        close,d=self.cube.closest_edges((0,0,0))
        assert_equal(d,0)
        assert_equal(len(close),3)

    def test_contiguity(self):
        # geo_graph = GeoGraph(G, multi, **kwargs)
        # assert_equal(expected, geo_graph.contiguity(pt1, pt2))
        raise SkipTest # TODO: implement your test here

    def test_tol(self):
        # geo_graph = GeoGraph(G, **kwargs)
        # assert_equal(expected, geo_graph.tol())
        raise SkipTest # TODO: implement your test here

    def test___str__(self):
        # geo_graph = GeoGraph(G, **kwargs)
        # assert_equal(expected, geo_graph.__str__())
        raise SkipTest # TODO: implement your test here

class TestDelauney:
    def test_delauney_triangulation(self):
        n=25
        from random import random
        nodes=[(random(),random()) for _ in range(n)]
        graph=delauney_triangulation(nodes)
        assert_equal(graph.number_of_nodes(),n)
        assert_true(nx.is_connected(graph))
        open('delauney.png','wb').write(graph.render())
        
        graph=euclidean_minimum_spanning_tree(nodes)
        open('emst.png','wb').write(graph.render())

class TestRender:
    def test_render(self):
        # assert_equal(expected, render(g, pos, format, **kwargs))
        raise SkipTest # TODO: implement your test here

class TestDelauneyTriangulation:
    def test_delauney_triangulation(self):
        # assert_equal(expected, delauney_triangulation(nodes, **kwargs))
        raise SkipTest # TODO: implement your test here

class TestEuclideanMinimumSpanningTree:
    def test_euclidean_minimum_spanning_tree(self):
        # assert_equal(expected, euclidean_minimum_spanning_tree(nodes, **kwargs))
        raise SkipTest # TODO: implement your test here
    
if __name__ == "__main__":
    import nose
    nose.runmodule()