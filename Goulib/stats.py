#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
statistics
"""

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2012, Philippe Guglielmetti"
__credits__ = ["https://github.com/tokland/pyeuler/blob/master/pyeuler/toolset.py",]
__license__ = "LGPL"

def mean(data):
    """:return: mean of data"""
    return float(sum(data))/len(data)

avg=mean #alias

def variance(data,avg=None):
    """:return: variance of data"""
    if avg==None:
        avg=mean(data)
    s = sum(((value - avg)**2) for value in data)
    var = float(s)/(len(data) - 1)
    return var

var=variance #alias

def stats(l):
    """:return: min,max,sum,sum2,avg,var of a list"""
    lo=float("inf")
    hi=float("-inf")
    n=0
    sum=0. #must be float
    sum2=0. #must be float
    for i in l:
        if i is not None:
            n+=1
            sum+=i
            sum2+=i*i
            if i<lo:lo=i
            if i>hi:hi=i
    if n>0:
        avg=sum/n
        var=sum2/n-avg*avg #mean of square minus square of mean
    else:
        avg=None
        var=None
    return lo,hi,sum,sum2,avg,var

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
    import scipy.stats, numpy #TODO remove these requirements
    
    x = numpy.array(x)
    y = numpy.array(y)
    n = len(x)
    xy = x * y
    xx = x * x
 
    # estimates
 
    b1 = (xy.mean() - x.mean() * y.mean()) / (xx.mean() - x.mean()**2)
    b0 = y.mean() - b1 * x.mean()
    s2 = 1./n * sum([(y[i] - b0 - b1 * x[i])**2 for i in xrange(n)])
    
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