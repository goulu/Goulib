#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
generate JavaScript charts using http://nvd3.org/
outputs strings to inline in HTML : no fancy JSON or server dependent stuff
"""
__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2012, Philippe Guglielmetti"
__credits__ = ["http://d3js.org/","http://nvd3.org/"]
__license__ = "LGPL"

import datetime
import colors

_count=0 #Chart counter for unique ids

def generate(y,x=None,length=100):
    """return a list of values y(x), where x and y can be
    iterable, functions or generators"""
    try: #y already iterable ?
        return list(y[:])
    except: pass
    if x:
        x=generate(x,None,length)
    else: 
        x=range(length)
    try: # y is f(x) ?
        return map(y,x)
    except: pass
    try:
        return [y[i] for i in x]
    except: pass
    return [y() for i in x]
    
from math import *
import unittest
class TestCase(unittest.TestCase):
    def runTest(self):
        y=[1,2,3]
        self.assertEqual(generate(y),y)
        self.assertEqual(generate(lambda x:x,y),y)
        generate(sin,cos)

class Chart:
    """base class. should not be used directly"""
    def __init__(self,model, name=None,**kwargs):
        self.model=model
        if not name:
            global _count
            _count+=1
            name="chart%d"%(_count)
        self.name=name
        
        self.args=kwargs
        self.data=[]
        self.axes={}
        try:
            self.x=generate(kwargs['x'])
            del kwargs['x']
        except: #no data
            self.x=None
            
        try:
            """date format : see https://github.com/mbostock/d3/wiki/Time-Formatting"""
            self.dateformat=kwargs['dateformat']
            del kwargs['dateformat']
        except:
            self.dateformat='%x'
            
        try: 
            self.colors=kwargs['colors']
            del kwargs['colors']
        except: 
            self.colors=colors.color_range(6)
        
        try:
            y=kwargs['y']
            del kwargs['y']
            for v in y:
                self.add(y=v,**kwargs)
        except: #no data
            pass
            
    def add(self, y, name=None, x=None, **kwargs):
        """adds a curve"""
        if not x:
            x=self.x
        x=generate(x,None,100)
        y=generate(y,x)
        if len(x)<len(y):
            x=range(len(y))
            
        if not name : name="Stream %d"%(len(self.data)+1)
        
        try: color=kwargs['color']
        except: color=self.colors[len(self.data)%len(self.colors)]
        
        if isinstance(x[0],(datetime.date,datetime.time,datetime.datetime)):
                self.axes['xAxis']["tickFormat"]="function(d) { return d3.time.format('%s')(new Date(d)) }\n"%self.dateformat 
                x=[d.isoformat() for d in x]   
            
        y=[{"x":x[i],"y":y} for i,y in enumerate(y)]
        data={"values":y,"key":name,"color":color}
        #multiChart
        try: data["type"]=kwargs["type"]
        except: pass
        
        try: data["yAxis"]=kwargs["axis"]
        except: data["yAxis"]="1"
        
        try : 
            if kwargs["bar"]: data["bar"]='true'
        except: pass
        
        try: data["disabled"]=kwargs["disabled"]
        except : pass
        
        self.data.append(data)
        
    def axis(self,name,label=None,format=".2f"):
        axe={}
        axe["tickFormat"]="d3.format(',%s')"%format
        if label:
            axe["axisLabel"]=label
        self.axes[name]=axe
        
    def __str__(self):
        #generate HTML div
        style=''
        try:
            style+=' width:%spx;'%self.args["width"]
        except: pass
        try:
            style+=' height:%spx;'%self.args["height"]
        except: pass
        if style:
            style=' style="%s"'%style
        out='<div id="%s"><svg%s></svg></div>\n'%(self.name,style)
        
        #generate Javascript
        out+="""<script type="text/javascript">
            nv.addGraph(function() {
                var chart = nv.models.%s();\n"""%self.model
                
        try:
            if self.args["stacked"]:
                out+="chart.stacked(true);"
        except:
            pass
                               
        for k,a in self.axes.iteritems():
            out+="chart.%s\n"%k
            for attr,value in a.iteritems():
                out+="    .%s(%s)\n"%(attr,value)
        out+="""d3.select('#%s svg')
                .datum(data_%s)
                .transition().duration(500).call(chart);
        """%(self.name,self.name) #for some reason the data get mixed...
        try:
            resize=self.args["resize"]
        except:
            resize=True
        if resize: out+="nv.utils.windowResize(chart.update);\n"
        out+="return chart;\n});\n"
        import json
        out+="""data_%s=%s;\n</script>"""%(self.name,json.dumps(self.data))
        return out

"""the following classes correspond to those defined in nv.d3.js"""

class LineChart(Chart):
    def __init__(self,**kwargs):
        Chart.__init__(self,'lineChart',**kwargs)
        self.axis('xAxis')
        self.axis('yAxis')
        
class ScatterChart(Chart):
    def __init__(self,**kwargs):
        Chart.__init__(self,'scatterChart',**kwargs)
        self.axis('xAxis')
        self.axis('yAxis')
        
class LineWithFocusChart(Chart):
    def __init__(self,**kwargs):
        try: # must have a specified height, otherwise it superimposes both chars
            kwargs['height']
        except:
            kwargs['height']=250
        Chart.__init__(self,'lineWithFocusChart',**kwargs)
        self.axis('xAxis')
        self.axis('yAxis')
        self.axis('y2Axis')

        
class MultiBarChart(Chart):
    def __init__(self,**kwargs):
        Chart.__init__(self,'multiBarChart',**kwargs)
        self.axis('xAxis')
        self.axis('yAxis')
        
class MultiBarHorizontalChart(Chart):
    def __init__(self,**kwargs):
        Chart.__init__(self,'multiBarHorizontalChart',**kwargs)
        self.axis('xAxis')
        self.axis('yAxis')
        
class CumulativeLineChart(Chart):
    def __init__(self,**kwargs):
        Chart.__init__(self,'cumulativeLineChart',**kwargs)
        self.axis('xAxis')
        self.axis('yAxis')
        
class StackedAreaChart(Chart):
    def __init__(self,**kwargs):
        Chart.__init__(self,'stackedAreaChart',**kwargs)
        self.axis('xAxis')
        self.axis('yAxis')
        
class LinePlusBarChart(Chart):
    def __init__(self,**kwargs):
        Chart.__init__(self,'linePlusBarChart',**kwargs)
        self.axis('xAxis')
        self.axis('y1Axis')
        self.axis('y2Axis')
        
class MultiChart(Chart):
    def __init__(self,**kwargs):
        Chart.__init__(self,'multiChart',**kwargs)
        self.axis('xAxis')
        self.axis('yAxis1')
        self.axis('yAxis2')
    
        
"""additional useful classes"""
class Pareto(MultiChart):
    def __init__(self,values,norm=None,**kwargs):
        MultiChart.__init__(self,**kwargs)
        values=generate(values)
        values.sort(reverse=True)
        from math2 import cumsum
        if not norm:norm=sum(values)
        self.add([x/norm for x in cumsum(values)],type="line",name="CumSum", axis=2)
        self.add(values,type="bar",name="Histo") # second, only for nicer colors
        
def hist(values, bins=None):
    values.sort()
    if not bins:
        bins=range(int(values[0]),int(values[-1])+1)
    sbins=list(bins)
    hist=[]
    while values:
        hist.append(0)
        while values and values[0]<=bins[0]: 
            hist[-1]+=1
            del values[0]
        del bins[0]
    return hist,sbins

class Histogram(MultiChart):
    def __init__(self,values,**kwargs):
        MultiChart.__init__(self,**kwargs)
        try:
            bins=generate(kwargs['bins'])
            del kwargs['bins']
        except: #no data
            bins=None
        values=generate(values)
        values,bins=hist(values,bins)
        self.add(values,x=bins,type="bar")
        
"""
Test data generators inspired by Lee Byron's used in http://leebyron.com/else/streamgraph/
"""
from random import random
from math import exp

def bump(n,w=5):
    a=[0]*n
    for b in range(w):
        x = 1. / (.1 + random())
        y = 2. * random() - .5
        z = 10. / (.1 + random())
        for i in range(n):
            w = (float(i) / n - y) * z;
            a[i] += x * exp(-w * w);
    return [x if x>0 else 0 for x in a]

if __name__ == '__main__': #tests
    import markup
    from math import sin,cos
    from random import uniform, gauss
    from itertools2 import arange
    
    page=markup.page()
    page.init(
            doctype="Content-Type: text/html; charset=utf-8\r\n\r\n<!DOCTYPE html>",
            script=["http://nvd3.org/lib/d3.v2.js","http://nvd3.org/nv.d3.js"], #must be a list to preserve order
            css=['http://nvd3.org/src/nv.d3.css']
            )
    #generate some data
    X=list(arange(0.,2*pi,pi/50))
    def Uniform():return uniform(0,1)
    def Gauss():return gauss(.5,.2)
    Waves=[bump(len(X)) for i in range(4)]

    # a simple linechart
    type="lineChart"
    page.h3(type)
    fig=LineChart(y=Waves,x=X)
    page.add(str(fig))
    
    type="lineWithFocusChart"
    page.h3(type)
    fig=LineWithFocusChart(name=type,x=X,y=Waves)
    page.add(str(fig))
    
    from datetime2 import days
    Date=days(datetime.date.today(),len(X))
    
    type="multiBarHorizontal"
    page.h3(type)
    fig=MultiBarHorizontalChart(width=600,height=600)
    for w in Waves:
        fig.add(w)
    page.add(str(fig))
    
    type="multiBarChart"
    page.h3(type)
    fig=MultiBarChart(name=type,dateformat='%d %b %y')
    for w in Waves:
        fig.add(w,x=Date)
    page.add(str(fig))
    
    type="cumulativeLineChart"
    page.h3(type)
    fig=CumulativeLineChart(name=type,x=X,y=Waves)
    page.add(str(fig))
    
    type="stackedAreaChart"
    page.h3(type)
    fig=StackedAreaChart(name=type,x=X,y=Waves)
    page.add(str(fig))
    
    type="linePlusBarChart"
    page.h3(type)
    fig=LinePlusBarChart(name=type,x=X) # TODO : Date not supported ?
    fig.add(Waves[1],"Wave 1",bar=True)
    fig.add(Waves[2],"Wave 2")
    page.add(str(fig))
    
    type="Pareto"
    page.h3(type)
    fig=Pareto(Gauss)
    page.add(str(fig))
    
    type="multiChart"
    page.h3(type)
    fig=MultiChart(name=type,x=X)  # TODO : Date not supported ?
    fig.add(Waves[1],"Wave 1",type="bar")
    fig.add(Waves[2],"Wave 2",type="line")
    fig.add(Waves[3],"Wave 3",type="area",axis=2)
    fig.add(Waves[0],"Wave 0",type="bar",axis=2)
    page.add(str(fig))
    
    type="scatterChart"
    page.h3(type)
    fig=ScatterChart(name=type,width=400,height=400)
    fig.add(x=map(cos,X), y=map(sin,X), name="Circle",type="line")
    fig.add(x=Gauss, y=Uniform, name="Noise")
    page.add(str(fig))
    print page
    
