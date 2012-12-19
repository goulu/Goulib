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

class Chart:
    def __init__(self,name="chart",model="lineChart",**kwargs):
        self.name=name
        self.model=model
        self.args=kwargs
        self.data=[]
        self.axes={}
        self.axis('xAxis')
        self.axis('yAxis')
        
    def axis(self,name,label=None,format=".2f"):
        axe={}
        axe["tickFormat"]="d3.format(',%s')"%format
        if label:
            axe["axisLabel"]=label
        self.axes[name]=axe
            
    def add(self, values, x=None, name=None, color=None, bar=None, type=None, axis=1):
        if not name:
            name="Stream %d"%(len(self.data)+1)
        if not color:
            color=['red','green','blue','black'][len(self.data)%4]
        if not x:
            x=range(len(values)+1)
        else:
            if isinstance(x[0],(datetime.date,datetime.time,datetime.datetime)):
                self.axes['xAxis']["tickFormat"]="function(d) { return d3.time.format('%x')(new Date(d)) }\n" 
                x=[d.isoformat() for d in x]   
            
        values=[{"x":x[i],"y":y} for i,y in enumerate(values)]
        data={"values":values,"key":name,"color":color}
        if type: #multiChart
            data["type"]=type
            data["yAxis"]=axis
        elif bar: #linePlusBarChart
            data["bar"]='true'
        self.data.append(data)
        
    def __str__(self):
        #generate HTML div
        out='<div id="%s"><svg'%self.name
        try:
            out+=' width="%s"'%self.args["width"]
        except: pass
        try:
            out+=' height="%s"'%self.args["height"]
        except: pass
        out+='></svg></div>\n'
        
        #generate Javascript
        out+="""<script type="text/javascript">
            nv.addGraph(function() {
                var chart = nv.models.%s();\n"""%self.model
                               
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
                
if __name__ == '__main__': #tests
    import markup
    from math import *
    from math2 import arange
    import random
    
    libpath="../../static"
    page=markup.page()
    page.init(
            doctype="Content-Type: text/html; charset=utf-8\r\n\r\n<!DOCTYPE html>",
            script=["%s/d3.v2.js"%libpath,"%s/nv.d3.js"%libpath], #must be a list to preserve order
            css=( '%s/nv.d3.css'%libpath)
            )
    #generate some data
    X=list(arange(0.,2*pi,pi/50))
    SinX=[sin(x) for x in X]
    CosX=[cos(x) for x in X]
    #cumulative functions do not like negative values
    SinX1=[1+sin(x) for x in X] 
    CosX1=[1+cos(x) for x in X]
    Uniform=[random.uniform(0,1) for x in X]
    Gauss=[random.gauss(.5,.2) for x in X]
    
    # a simple linechart
    type="lineChart"
    page.h3(type)
    fig=Chart()
    fig.add(SinX,X,"Sin")
    fig.add(CosX,X,"Cos")
    fig.add(Uniform,X,"Uniform")
    fig.add(Gauss,X,"Gauss")
    page.add(str(fig))
    
    type="lineWithFocusChart"
    page.h3(type)
    fig=Chart(name=type,model=type,height=400)
    fig.axis('y2Axis')
    fig.add(SinX,X,"Sin")
    fig.add(CosX,X,"Cos")
    fig.add(Uniform,X,"Uniform")
    fig.add(Gauss,X,"Gauss")
    page.add(str(fig))
    
    from datetime2 import days
    Date=days(datetime.date.today(),len(X))
    
    type="multiBarChart"
    page.h3(type)
    fig=Chart(name=type,model=type)
    fig.add(SinX1,Date,"Sin+1")
    fig.add(CosX1,Date,"Cos+1")
    fig.add(Uniform,Date,"Uniform")
    fig.add(Gauss,Date,"Gauss")
    page.add(str(fig))
    
    type="cumulativeLineChart"
    page.h3(type)
    fig=Chart(name=type,model=type)
    fig.add(SinX1,X,"Sin+1")
    fig.add(CosX1,X,"Cos+1")
    fig.add(Uniform,X,"Uniform")
    fig.add(Gauss,X,"Gauss")
    page.add(str(fig))
    
    type="stackedAreaChart"
    page.h3(type)
    fig=Chart(name=type,model=type)
    fig.add(SinX1,X,"Sin+1")
    fig.add(CosX1,X,"Cos+1")
    fig.add(Uniform,X,"Uniform")
    fig.add(Gauss,X,"Gauss")
    page.add(str(fig))
    
    type="linePlusBarChart"
    page.h3(type)
    fig=Chart(name=type,model=type)
    fig.axes={} #clear
    fig.axis('xAxis')
    fig.axis('y1Axis')
    fig.axis('y2Axis')
    fig.add(Uniform,X,"Uniform",bar=True)
    fig.add(Gauss,X,"Gauss")
    page.add(str(fig))
    
    type="multiChart"
    page.h3(type)
    fig=Chart(name=type,model=type)
    fig.axes={} #clear
    fig.axis('xAxis')
    fig.axis('yAxis1')
    fig.axis('yAxis2')
    fig.add(SinX,X,"Sin",type="bar")
    fig.add(CosX,X,"Cos",type="line")
    fig.add(Uniform,X,"Uniform",type="area")
    fig.add(Gauss,X,"Gauss",type="bar")
    page.add(str(fig))
    
    type="scatterChart"
    page.h3(type)
    fig=Chart(name=type,model=type,height=400)
    fig.add(CosX,SinX,"Circle")
    fig.add(Uniform,Gauss,"Noise")
    page.add(str(fig))
    print page
    
