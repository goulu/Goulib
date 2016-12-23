#!/usr/bin/env python
# coding: utf8

from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *
from Goulib.stats import *

from random import random

import os
path=os.path.dirname(os.path.abspath(__file__))

h=[64630,11735,14216,99233,14470,4978,73429,38120,51135,67060] # data from https://www.hackerrank.com/challenges/stat-warmup
hmean=43900.6 
hvar=1031372102 #variance of h computed in Matlab

r=[random() for _ in range(5000)]
n=10000

f=[float(_)/n for _ in range(n+1)]
fvar=1./12 #variance of f, theoretical

class TestMean:
    def test_mean(self):
        assert_equal(avg(f),0.5)
        assert_equal(mean(h),hmean)
        assert_equal(mean(r),0.5,places=1)

class TestVariance:
    def test_variance(self):
        assert_equal(var(f),1./12,4)
        assert_equal(variance(h),hvar,0)
        assert_equal(variance(r),0.082,places=2)

class TestStats:
    @classmethod
    def setup_class(self):
        self.f=Stats(f)
        self.h=Stats(h)
        
    def test___init__(self):
        pass #tested above

    def test_append(self):
        pass #tested above

    def test_extend(self):
        pass #tested above
    
    def test_remove(self):
        # Stats = Stats(data, mean, var)
        # assert_equal(expected, Stats.remove(data))
        raise SkipTest 

    def test_mean(self):
        assert_equal(self.f.mean,0.5)
        assert_equal(self.h.avg,hmean)
        
    def test_variance(self):
        assert_equal(self.f.variance,fvar)
        assert_equal(self.h.var,hvar)


    def test_stddev(self):
        # Stats = Stats(data, mean, var)
        # assert_equal(expected, Stats.stddev())
        raise SkipTest 

class TestStddev:
    def test_stddev(self):
        assert_equal(stddev(h),math.sqrt(hvar),1)

class TestConfidenceInterval:
    def test_confidence_interval(self):
        assert_equal(confidence_interval(h),(23996,63806),0)

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

    def test___init__(self):
        # mode = Mode(name, nchannels, type, min, max)
        raise SkipTest # TODO: implement your test here

    def test___repr__(self):
        # mode = Mode(name, nchannels, type, min, max)
        # assert_equal(expected, mode.__repr__())
        raise SkipTest # TODO: implement your test here

class TestStats:
    def test_stats(self):
        # https://www.hackerrank.com/challenges/stat-warmup
        min,max,sum,sum2,avg,var=stats(h)
        assert_equal(min,4978)
        assert_equal(max,99233)
        assert_equal(sum,439006)
        assert_equal(sum2,math2.dot(h,h))
        assert_equal(avg,hmean)
        assert_equal(var,hvar,0)

        min,max,sum,sum2,avg,var=stats(f)
        assert_equal(min,0.,1)
        assert_equal(max,1.,1)
        assert_equal(sum,(n+1)/2.,2)
        assert_equal(sum2,(n+2)/3.,0)
        assert_equal(avg,0.5,4)
        assert_equal(var,1./12,4)

    def test_stats(self):
        # assert_equal(expected, stats(l))
        raise SkipTest 

    def test___repr__(self):
        # stats = Stats(data)
        # assert_equal(expected, stats.__repr__())
        raise SkipTest 

    def test_sum(self):
        # stats = Stats(data)
        # assert_equal(expected, stats.sum())
        raise SkipTest 

    def test_sum2(self):
        # stats = Stats(data)
        # assert_equal(expected, stats.sum2())
        raise SkipTest 

    def test_stats(self):
        # assert_equal(expected, stats(l))
        raise SkipTest # TODO: implement your test here

    def test___add__(self):
        # stats = Stats(data, mean, var)
        # assert_equal(expected, stats.__add__(other))
        raise SkipTest # TODO: implement your test here

    def test___mul__(self):
        # stats = Stats(data, mean, var)
        # assert_equal(expected, stats.__mul__(other))
        raise SkipTest # TODO: implement your test here

    def test___neg__(self):
        # stats = Stats(data, mean, var)
        # assert_equal(expected, stats.__neg__())
        raise SkipTest # TODO: implement your test here

    def test___pow__(self):
        # stats = Stats(data, mean, var)
        # assert_equal(expected, stats.__pow__(n))
        raise SkipTest # TODO: implement your test here

    def test___repr__(self):
        # stats = Stats(data, mean, var)
        # assert_equal(expected, stats.__repr__())
        raise SkipTest # TODO: implement your test here

    def test___sub__(self):
        # stats = Stats(data, mean, var)
        # assert_equal(expected, stats.__sub__(other))
        raise SkipTest # TODO: implement your test here

    def test_covariance(self):
        # stats = Stats(data, mean, var)
        # assert_equal(expected, stats.covariance(other))
        raise SkipTest # TODO: implement your test here

    def test_sum(self):
        # stats = Stats(data, mean, var)
        # assert_equal(expected, stats.sum())
        raise SkipTest # TODO: implement your test here

    def test_sum2(self):
        # stats = Stats(data, mean, var)
        # assert_equal(expected, stats.sum2())
        raise SkipTest # TODO: implement your test here

    def test_stats(self):
        # assert_equal(expected, stats(l))
        raise SkipTest # TODO: implement your test here

    def test___add__(self):
        # stats = Stats(data, mean, var)
        # assert_equal(expected, stats.__add__(other))
        raise SkipTest # TODO: implement your test here

    def test___mul__(self):
        # stats = Stats(data, mean, var)
        # assert_equal(expected, stats.__mul__(other))
        raise SkipTest # TODO: implement your test here

    def test___neg__(self):
        # stats = Stats(data, mean, var)
        # assert_equal(expected, stats.__neg__())
        raise SkipTest # TODO: implement your test here

    def test___pow__(self):
        # stats = Stats(data, mean, var)
        # assert_equal(expected, stats.__pow__(n))
        raise SkipTest # TODO: implement your test here

    def test___repr__(self):
        # stats = Stats(data, mean, var)
        # assert_equal(expected, stats.__repr__())
        raise SkipTest # TODO: implement your test here

    def test___sub__(self):
        # stats = Stats(data, mean, var)
        # assert_equal(expected, stats.__sub__(other))
        raise SkipTest # TODO: implement your test here

    def test_covariance(self):
        # stats = Stats(data, mean, var)
        # assert_equal(expected, stats.covariance(other))
        raise SkipTest # TODO: implement your test here

    def test_sum(self):
        # stats = Stats(data, mean, var)
        # assert_equal(expected, stats.sum())
        raise SkipTest # TODO: implement your test here

    def test_sum2(self):
        # stats = Stats(data, mean, var)
        # assert_equal(expected, stats.sum2())
        raise SkipTest # TODO: implement your test here

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
        self.h=Normal(h)
        
    def test___init__(self):
        pass # tested above
    
    def test_append(self):
        pass # tested above

    def test_extend(self):
        pass # tested above
    
    def test___str__(self):
        assert_equal(str(self.gauss),'Normal(mean=1, var=1)')
        
    def test_mean(self):
        assert_equal(self.h.avg,hmean)

    def test_variance(self):
        assert_equal(self.gauss.var,1)
        assert_equal(self.two.var,0)
        assert_equal(self.h.var,hvar,0)
        
    def test_stddev(self):
        assert_equal(self.h.stddev,math.sqrt(hvar),5)
    
    def test_linear(self):
        pass # tested below
        
    def test___add__(self):
        twogauss=self.gauss+self.gauss
        assert_equal(twogauss.var,2)
        assert_equal(twogauss.avg,2)
        
        n=self.gauss+1
        assert_equal(n.var,1)
        assert_equal(n.avg,2)
        
    def test___radd__(self):
        n=1+self.gauss
        assert_equal(n.avg,2)
        assert_equal(n.var,1)

    def test___sub__(self):
        zero=self.gauss-self.gauss
        assert_equal(zero.avg,0)
        assert_equal(zero.var,2)
        
    def test___mul__(self):
        twogauss=self.gauss*2
        assert_equal(twogauss.avg,2)
        assert_equal(twogauss.var,2)
        
    def test___div__(self):
        halfgauss=self.gauss/2
        assert_equal(halfgauss.avg,.5)
        assert_equal(halfgauss.var,.5)

    def test___call__(self):
        # normal = Normal()
        # assert_equal(expected, normal.__call__(x))
        raise SkipTest 

    def test___neg__(self):
        # normal = Normal()
        # assert_equal(expected, normal.__neg__())
        raise SkipTest 

    def test___rsub__(self):
        # normal = Normal()
        # assert_equal(expected, normal.__rsub__(other))
        raise SkipTest 

    def test_covariance(self):
        # normal = Normal()
        # assert_equal(expected, normal.covariance(other))
        raise SkipTest 

    def test_pdf(self):
        # normal = Normal()
        # assert_equal(expected, normal.pdf(x))
        raise SkipTest 

    def test_pearson(self):
        # normal = Normal()
        # assert_equal(expected, normal.pearson(other))
        raise SkipTest 

    def test_plot(self):
        # normal = Normal()
        # assert_equal(expected, normal.plot(fmt, x))
        raise SkipTest 

    def test_pop(self):
        # normal = Normal()
        # assert_equal(expected, normal.pop(i, n))
        raise SkipTest 

    def test_remove(self):
        # normal = Normal()
        # assert_equal(expected, normal.remove(x))
        raise SkipTest 
    
    def test_save(self):
        n1=Normal()
        n2=Normal([],2,2)
        n3=Normal([2]) # TODO find why it does not show
        plot.save([n1,n2,n1+n2],path+'/results/stats.gauss.png')

    def test_latex(self):
        # normal = Normal(data, mean, var)
        # assert_equal(expected, normal.latex())
        raise SkipTest # TODO: implement your test here

class TestMeanVar:
    def test_mean_var(self):
        # assert_equal(expected, mean_var(data))
        raise SkipTest 

class TestKurtosis:
    def test_kurtosis(self):
        # assert_equal(expected, kurtosis(data))
        raise SkipTest 

class TestCovariance:
    def test_covariance(self):
        # assert_equal(expected, covariance(data1, data2))
        raise SkipTest 



class TestNormalPdf:
    def test_normal_pdf(self):
        # assert_equal(expected, normal_pdf(x, mu, sigma))
        raise SkipTest 

class TestDiscrete:
    def test___call__(self):
        # discrete = Discrete(data)
        # assert_equal(expected, discrete.__call__(x))
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # discrete = Discrete(data)
        raise SkipTest # TODO: implement your test here

class TestPDF:
    def test___call__(self):
        # p_d_f = PDF(pdf, data)
        # assert_equal(expected, p_d_f.__call__(x, **kwargs))
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # p_d_f = PDF(pdf, data)
        raise SkipTest # TODO: implement your test here

if __name__ == "__main__":
    runmodule()