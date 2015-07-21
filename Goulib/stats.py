#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
very basic statistics functions
"""

from __future__ import division #"true division" everywhere

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2012, Philippe Guglielmetti"
__credits__ = []
__license__ = "LGPL"

import six, math, logging, matplotlib

from . import plot #sets matplotlib backend
import matplotlib.pyplot as plt # after import .plot

from . import itertools2
from .math2 import vecmul, vecadd, vecsub

def mean(data):
    """:return: mean of data"""
    return float(sum(data))/len(data)

avg=mean #alias

def variance(data,avg=None):
    """:return: variance of data"""
    if avg==None:
        avg=mean(data)
    s = sum(((value - avg)**2) for value in data)
    var = float(s)/len(data)
    return var

var=variance #alias

def stddev(data,avg=None):
    """:return: standard deviation of data"""
    return math.sqrt(variance(data,avg))

def confidence_interval(data,conf=0.95, avg=None):
    """:return: (low,high) bounds of 95% confidence interval of data"""
    if avg is None:
        avg=mean(data)
    e = 1.96 * stddev(data,avg) / math.sqrt(len(data))
    return avg-e,avg+e

def median(data, is_sorted=False):
    """:return: median of data"""
    x=data if is_sorted else sorted(data)
    n=len(data)
    i=n//2
    if n % 2:
        return x[i]
    else:
        return avg(x[i-1:i+1])

def mode(data, is_sorted=False):
    """:return: mode (most frequent value) of data"""
    #we could use a collection.Counter, but we're only looking for the largest value
    x=data if is_sorted else sorted(data)
    res,count=None,0
    prev,c=None,0
    x.append(None)# to force the last loop
    for v in x:
        if v==prev:
            c+=1
        else:
            if c>count: #best so far
                res,count=prev,c
            c=1
        prev=v
    x.pop() #no side effect please
    return res


def stats(l):
    """:return: min,max,sum,sum2,avg,var of a list"""
    lo=float("inf")
    hi=float("-inf")
    n=0
    sum1=0. #must be float
    sum2=0. #must be float
    for i in l:
        if i is not None:
            n+=1
            sum1+=i
            sum2+=i*i
            if i<lo:lo=i
            if i>hi:hi=i
    if n>0:
        avg=sum1/n
        var=sum2/n-avg*avg #mean of square minus square of mean
    else:
        avg=None
        var=None
    return lo,hi,sum1,sum2,avg,var

class Normal(list,plot.Plot):
    """represents a normal distributed variable 
    the base class (list) optionally contains data
    """
    def __init__(self,data=[],mean=0,var=1):
        self.lo=float("inf")
        self.hi=float("-inf")
        if data:
            self.n=0
            self.sum1=0
            self.sum2=0
            self.extend(data)
        else:
            self.n=1
            self.sum1=self.n*mean
            self.sum2=self.n*(var+mean**2)
        
    @property
    def mean(self):
        return self.sum1/self.n
    
    avg=mean #alias
    average=mean #alias
    mu=mean #alias
    
    @property
    def variance(self):
        return self.sum2/self.n-self.mean**2
    
    var=variance #alias
    @property
    def stddev(self):
        return math.sqrt(self.variance)
    
    sigma=stddev
    
    def pdf(self, x):
        """Return the probability density function at x"""
        return 1./(math.sqrt(2*math.pi)*self.sigma)*math.exp(-0.5 * (1./self.sigma*(x - self.mu))**2)
    
    def __call__(self,x):
        try: #is x iterable ?
            return [self(x) for x in x]
        except: pass
        return self.pdf(x)
    
    def __repr__(self):
        return "%s(μ=%s, σ=%s)"%(self.__class__.__name__,self.mean,self.stddev)
    
    def _repr_latex_(self):
        return "\mathcal{N}(\mu=%s, \sigma=%s)"%(self.mean,self.stddev)
    
    def plot(self, fmt='svg', x=None):
        from IPython.core.pylabtools import print_figure

        # plt.rc('text', usetex=True)
        fig, ax = plt.subplots()
        if x is None:
            x=itertools2.ilinear(self.mu-3*self.sigma,self.mu+3*self.sigma, 101)
        x=list(x)
        y=self(x)
        ax.plot(x,y)
        ax.set_title(self._repr_latex_())
        data = print_figure(fig, fmt)
        plt.close(fig)
        return data
    
    def _repr_png_(self):
        return self.plot(fmt='png')

    def _repr_svg_(self):
        return self.plot(fmt='svg')
    
    def append(self,x):
        super(Normal,self).append(x)
        self.n+=1
        self.sum1+=x
        self.sum2+=x*x
        if x<self.lo: self.lo=x
        if x>self.hi: self.hi=x
        
    def extend(self,data):
        for x in data:
            self.append(x)
            
    def linear(self,a,b=0):
        """
        :return: a*self+b 
        """
        return Normal(
            data=[a*x+b for x in self],
            mean=self.mean*a+b, 
            var=abs(self.var*a)
        )
    
    def __mul__(self,a):
        return self.linear(a,0)
    
    def __div__(self,a):
        return self.linear(1./a,0)

    def __add__(self, other):
        if isinstance(other,(int,float)):
            return self.linear(1,other)
        # else: assume other is a Normal variable
        mean=self.mean+other.mean
        var=self.var+other.var+2*self.cov(other)
        data=vecadd(self,other) if len(self)==len(other) else []
        return Normal(data=data, mean=mean, var=var)
    
    def __radd__(self, other):
        return self+other
    
    def __neg__(self):
        return self*(-1)
        
    def __sub__(self, other):
        return self+(-other)
    
    def __rsub__(self, other):
        return -(self-other)
    
    def covariance(self,other):
        try:
            return mean(
                vecmul(
                    vecsub(self,[],self.mean),
                    vecsub(other,[],other.mean)
                )
            )
        except:
            return 0 # consider decorrelated
    
    cov=covariance #alias
    
    def pearson(self,other):
        return self.cov(other)/(self.stddev*other.stddev)
    
    correlation=pearson #alias
    corr=pearson #alias
        


def linear_regression(x, y, conf=None):
    """
    :param x,y: iterable data
    :param conf: float confidence level [0..1]. If None, confidence intervals are not returned
    :return: b0,b1,b2, (b0 
    
    Return the linear regression parameters and their <prob> confidence intervals.
 
    ex:
    >>> linear_regression([.1,.2,.3],[10,11,11.5],0.95)
    """
    # https://gist.github.com/riccardoscalco/5356167
    try:
        import scipy.stats, numpy #TODO remove these requirements
    except:
        logging.error('scipy needed')
        return None
    
    x = numpy.array(x)
    y = numpy.array(y)
    n = len(x)
    xy = x * y
    xx = x * x
 
    # estimates
 
    b1 = (xy.mean() - x.mean() * y.mean()) / (xx.mean() - x.mean()**2)
    b0 = y.mean() - b1 * x.mean()
    s2 = 1./n * sum([(y[i] - b0 - b1 * x[i])**2 for i in range(n)])
    
    if not conf:
        return b1,b0,s2
    
    #confidence intervals
    
    alpha = 1 - conf
    c1 = scipy.stats.chi2.ppf(alpha/2.,n-2)
    c2 = scipy.stats.chi2.ppf(1-alpha/2.,n-2)
    
    c = -1 * scipy.stats.t.ppf(alpha/2.,n-2)
    bb1 = c * (s2 / ((n-2) * (xx.mean() - (x.mean())**2)))**.5
    
    bb0 = c * ((s2 / (n-2)) * (1 + (x.mean())**2 / (xx.mean() - (x.mean())**2)))**.5
    
    return b1,b0,s2,(b1-bb1,b1+bb1),(b0-bb0,b0+bb0),(n*s2/c2,n*s2/c1)
