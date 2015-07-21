#!/usr/bin/python
# -*- coding: utf-8 -*-

from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *
from Goulib.stats import *

from random import random

h=[64630,11735,14216,99233,14470,4978,73429,38120,51135,67060] # data from https://www.hackerrank.com/challenges/stat-warmup
r=[random() for _ in range(5000)]
n=10000
f=[float(_)/n for _ in range(n+1)]

class TestMean:
    def test_mean(self):
        assert_equal(mean(h),43900.6)
        assert_equal(mean(r),0.5,places=1)

class TestVariance:
    def test_variance(self):
        assert_equal(variance(r),0.082,places=2)

class TestStddev:
    def test_stddev(self):
        assert_equal(stddev(h),30466.9,1)

class TestConfidenceInterval:
    def test_confidence_interval(self):
        assert_equal(confidence_interval(h),(25017.0,62784.2),1)

class TestMedian:
    def test_median(self):
        assert_equal(median(h),44627.5)
        assert_equal(median(r),0.5,places=1)

class TestMode:
    def test_mode(self):
        assert_equal(mode(h),4978)
        assert_equal(mode([1,1,1,2,2,3]),1) #test when mode is first
        assert_equal(mode([1,2,2,3,3,3]),3) #test when mode is last
        assert_equal(mode([1,2,2,3,3,4]),2) #test equality

class TestStats:
    def test_stats(self):
        # https://www.hackerrank.com/challenges/stat-warmup
        r=[64630,11735,14216,99233,14470,4978,73429,38120,51135,67060]
        min,max,sum,sum2,avg,var=stats(r)
        assert_equal(min,4978)
        assert_equal(max,99233)
        assert_equal(sum,439006)
        assert_equal(sum2,28554975720)
        assert_equal(avg,43900.6)
        assert_equal(var,928234891.64,3)

        min,max,sum,sum2,avg,var=stats(f)
        assert_equal(min,0.,1)
        assert_equal(max,1.,1)
        assert_equal(sum,(n+1)/2.,2)
        assert_equal(sum2,(n+2)/3.,0)
        assert_equal(avg,0.5,4)
        assert_equal(var,1./12,4)

class TestLinearRegression:
    def test_linear_regression(self):
        try:
            import scipy
        except:
            logging.warning('scipy required')
            return
        #first test a perfect fit
        a,b,c=linear_regression([1,2,3],[-1,-3,-5])
        assert_equal((a,b,c),(-2,1,0))
        a,b,c,ai,bi,ci=linear_regression([1,2,3],[-1,-3,-5],.95)
        assert_equal(ai,(-2,-2))
        assert_equal(bi,(1,1))
        assert_equal(ci,(0,0))

class TestNormal:
    @classmethod
    def setup_class(self):
        self.gauss=Normal(mean=1)
        self.two=Normal(mean=2,var=0)
        self.data=Normal([64630,11735,14216,99233,14470,4978,73429,38120,51135,67060])
        
    def test___init__(self):
        pass # tested above
    
    def test_append(self):
        pass # tested above

    def test_extend(self):
        pass # tested above
    
    def test___repr__(self):
        assert_equal(str(self.gauss),'Normal(μ=1.0, σ=1.0)')
    
    def test_linear(self):
        pass # tested below
        
    def test___add__(self):
        twogauss=self.gauss+self.gauss
        assert_equal(twogauss.avg,2)
        assert_equal(twogauss.var,2)
        
        n=self.gauss+1
        assert_equal(n.avg,2)
        assert_equal(n.var,1)
        
    def test___radd__(self):
        n=1+self.gauss
        assert_equal(n.avg,2)
        assert_equal(n.var,1)

    def test___mul__(self):
        twogauss=self.gauss*2
        assert_equal(twogauss.avg,2)
        assert_equal(twogauss.var,2)

    def test___sub__(self):
        zero=self.gauss-self.gauss
        assert_equal(zero.avg,0)
        assert_equal(zero.var,2)

    def test_mean(self):
        assert_equal(self.data.avg,43900.6)

    def test_variance(self):
        assert_equal(self.data.var,928234891.64,6)
        
    def test_stddev(self):
        assert_equal(self.data.stddev,math.sqrt(928234891.64))

if __name__ == "__main__":
    runmodule()