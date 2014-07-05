from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *
from Goulib.stats import *

from random import random

class TestMean:
    def test_mean(self):
        r=[random() for _ in range(5000)]
        assert_equal(mean(r),0.5,1)

class TestVariance:
    def test_variance(self):
        r=[random() for _ in range(5000)]
        assert_equal(variance(r),0.082,2)

class TestStats:
    def test_stats(self):
        n=10000
        r=[float(_)/n for _ in range(n+1)]
        min,max,sum,sum2,avg,var=stats(r)
        assert_equal(min,0.,1)
        assert_equal(max,1.,1)
        assert_equal(sum,(n+1)/2.,2)
        assert_equal(sum2,(n+2)/3.,0)
        assert_equal(avg,0.5,4)
        assert_equal(var,1./12,4)

class TestLinearRegression:
    def test_linear_regression(self):
        #first test a perfect fit
        a,b,c=linear_regression([1,2,3],[-1,-3,-5])
        assert_equal((a,b,c),(-2,1,0))
        a,b,c,ai,bi,ci=linear_regression([1,2,3],[-1,-3,-5],.95)
        assert_equal(ai,(-2,-2))
        assert_equal(bi,(1,1))
        assert_equal(ci,(0,0))

class TestQuantileFit:
    def test_quantile_fit(self):
        from scipy.stats import norm
        d=quantile_fit([0.25,0.5,0.75],[-0.68,0,0.68], dist=norm,x0=(.5,.5))
        assert_equal(d.mean(),0)
        assert_almost_equal(d.var(),1,delta=0.05)
        
        
if __name__ == "__main__":
    runmodule()