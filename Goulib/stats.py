#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
very basic statistics functions
"""

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2012, Philippe Guglielmetti"
__credits__ = ["https://github.com/tokland/pyeuler/blob/master/pyeuler/toolset.py",]
__license__ = "LGPL"

import logging, six, math

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
