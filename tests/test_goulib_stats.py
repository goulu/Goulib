from goulib.tests import *
from goulib.stats import *

from random import random

import os
path = os.path.dirname(os.path.abspath(__file__))

# data from https://www.hackerrank.com/challenges/stat-warmup
h = [64630, 11735, 14216, 99233, 14470, 4978, 73429, 38120, 51135, 67060]
hmean = 43900.6
hvar = 1031372102  # variance of h computed in Matlab

n = 10000

f = [float(_)/n for _ in range(n+1)]
fvar = 1./12  # variance of f, theoretical


class TestMean:
    def test_mean(self):
        assert avg(f) == pytest.approx(1/2)
        assert mean(h) == pytest.approx(hmean)


class TestVariance:
    def test_variance(self):
        assert var(f) == pytest.approx(1./12,rel=0.01)
        assert variance(h) == pytest.approx(hvar)


class TestStats:
    @classmethod
    def setup_class(self):
        self.f = Stats(f)
        self.h = Stats(h)

    def test___init__(self):
        pass  # tested above

    def test_append(self):
        pass  # tested above

    def test_extend(self):
        pass  # tested above

    def test_remove(self):
        # Stats = Stats(data, mean, var)
        # assert_equal(expected, Stats.remove(data))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_mean(self):
        assert self.f.mean == pytest.approx(0.5)
        assert self.h.avg == pytest.approx(hmean)

    def test_variance(self):
        assert self.f.variance == pytest.approx(fvar,0.01)
        assert math2.rint(self.h.var) == pytest.approx(hvar)

    def test_stddev(self):
        # Stats = Stats(data, mean, var)
        # assert_equal(expected, Stats.stddev())
        pytest.skip("not yet implemented")  # TODO: implement

    def test_stats(self):
        # assert_equal(expected, stats(l))
        # TODO: implement  # implement your test here
        pytest.skip("not yet implemented")

    def test___add__(self):
        # stats = Stats(data, mean, var)
        # assert_equal(expected, stats.__add__(other))
        # TODO: implement  # implement your test here
        pytest.skip("not yet implemented")

    def test___mul__(self):
        # stats = Stats(data, mean, var)
        # assert_equal(expected, stats.__mul__(other))
        # TODO: implement  # implement your test here
        pytest.skip("not yet implemented")

    def test___neg__(self):
        # stats = Stats(data, mean, var)
        # assert_equal(expected, stats.__neg__())
        # TODO: implement  # implement your test here
        pytest.skip("not yet implemented")

    def test___pow__(self):
        # stats = Stats(data, mean, var)
        # assert_equal(expected, stats.__pow__(n))
        # TODO: implement  # implement your test here
        pytest.skip("not yet implemented")

    def test___repr__(self):
        # stats = Stats(data, mean, var)
        # assert_equal(expected, stats.__repr__())
        # TODO: implement  # implement your test here
        pytest.skip("not yet implemented")

    def test___sub__(self):
        # stats = Stats(data, mean, var)
        # assert_equal(expected, stats.__sub__(other))
        # TODO: implement  # implement your test here
        pytest.skip("not yet implemented")

    def test_covariance(self):
        # stats = Stats(data, mean, var)
        # assert_equal(expected, stats.covariance(other))
        # TODO: implement  # implement your test here
        pytest.skip("not yet implemented")

    def test_sum(self):
        # stats = Stats(data, mean, var)
        # assert_equal(expected, stats.sum())
        # TODO: implement  # implement your test here
        pytest.skip("not yet implemented")

    def test_sum2(self):
        # stats = Stats(data, mean, var)
        # assert_equal(expected, stats.sum2())
        # TODO: implement  # implement your test here
        pytest.skip("not yet implemented")


class TestStddev:
    def test_stddev(self):
        assert stddev(h) == pytest.approx(math.sqrt(hvar))


class TestConfidenceInterval:
    def test_confidence_interval(self):
        assert confidence_interval(h) == pytest.approx((23996, 63806),abs=0.5)


class TestMedian:
    def test_median(self):
        assert median(h) == pytest.approx(44627.5)


class TestMode:
    def test_mode(self):
        assert mode(h) == 4978
        assert mode([1, 1, 1, 2, 2, 3]) == 1  # test when mode is first
        assert mode([1, 2, 2, 3, 3, 3]) == 3  # test when mode is last
        assert mode([1, 2, 2, 3, 3, 4]) == 2  # test equality

    def test___init__(self):
        # mode = Mode(name, nchannels, type, min, max)
        # TODO: implement  # implement your test here
        pytest.skip("not yet implemented")

    def test___repr__(self):
        # mode = Mode(name, nchannels, type, min, max)
        # assert_equal(expected, mode.__repr__())
        # TODO: implement  # implement your test here
        pytest.skip("not yet implemented")


class TestLinearRegression:
    def test_linear_regression(self):
        try:
            import scipy
        except:
            logging.warning('scipy required')
            return
        # first test a perfect fit
        a, b, c = linear_regression([1, 2, 3], [-1, -3, -5])
        assert (a, b, c) == pytest.approx((-2, 1, 0))
        a, b, c, ai, bi, ci = linear_regression([1, 2, 3], [-1, -3, -5], .95)
        assert ai == pytest.approx((-2, -2))
        assert bi == pytest.approx((1, 1))
        assert ci == pytest.approx((0, 0))


class TestNormal:
    @classmethod
    def setup_class(self):
        self.gauss = Normal(mean=1)
        self.two = Normal(mean=2, var=0)
        self.h = Normal(h)

    def test___init__(self):
        pass  # tested above

    def test_append(self):
        pass  # tested above

    def test_extend(self):
        pass  # tested above

    def test___str__(self):
        assert str(self.gauss) == 'Normal(mean=1, var=1)'

    def test_mean(self):
        assert self.h.avg == hmean

    def test_variance(self):
        assert self.gauss.var == pytest.approx(1)
        assert self.two.var ==pytest.approx(0)
        assert self.h.var == pytest.approx(hvar)

    def test_stddev(self):
        assert self.h.stddev == pytest.approx(math.sqrt(hvar))

    def test_linear(self):
        pass  # tested below

    def test___add__(self):
        twogauss = self.gauss+self.gauss
        assert twogauss.var == pytest.approx(2)
        assert twogauss.avg == pytest.approx(2)

        n = self.gauss+1
        assert n.var == pytest.approx(1)
        assert n.avg == pytest.approx(2)

    def test___radd__(self):
        n = 1+self.gauss
        assert n.avg == pytest.approx(2)
        assert n.var == pytest.approx(1)

    def test___sub__(self):
        zero = self.gauss-self.gauss
        assert zero.avg == pytest.approx(0)
        assert zero.var == pytest.approx(2)

    def test___mul__(self):
        twogauss = self.gauss*2
        assert twogauss.avg == pytest.approx(2)
        assert twogauss.var == pytest.approx(2)

    def test___div__(self):
        halfgauss = self.gauss/2
        assert halfgauss.avg == pytest.approx(1/2)
        assert halfgauss.var == pytest.approx(1/2)

    def test___call__(self):
        # normal = Normal()
        # assert_equal(expected, normal.__call__(x))
        pytest.skip("not yet implemented")  # TODO: implement

    def test___neg__(self):
        # normal = Normal()
        # assert_equal(expected, normal.__neg__())
        pytest.skip("not yet implemented")  # TODO: implement

    def test___rsub__(self):
        # normal = Normal()
        # assert_equal(expected, normal.__rsub__(other))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_covariance(self):
        # normal = Normal()
        # assert_equal(expected, normal.covariance(other))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_pdf(self):
        # normal = Normal()
        # assert_equal(expected, normal.pdf(x))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_pearson(self):
        # normal = Normal()
        # assert_equal(expected, normal.pearson(other))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_plot(self):
        # normal = Normal()
        # assert_equal(expected, normal.plot(fmt, x))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_pop(self):
        # normal = Normal()
        # assert_equal(expected, normal.pop(i, n))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_remove(self):
        # normal = Normal()
        # assert_equal(expected, normal.remove(x))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_save(self):
        n1 = Normal()
        n2 = Normal([], 2, 2)
        n3 = Normal([2])  # TODO find why it does not show
        plot.save([n1, n2, n1+n2], path+'/results/stats.gauss.png')

    def test_latex(self):
        # normal = Normal(data, mean, var)
        # assert_equal(expected, normal.latex())
        # TODO: implement  # implement your test here
        pytest.skip("not yet implemented")


class TestMeanVar:
    def test_mean_var(self):
        # assert_equal(expected, mean_var(data))
        pytest.skip("not yet implemented")  # TODO: implement


class TestKurtosis:
    def test_kurtosis(self):
        # assert_equal(expected, kurtosis(data))
        pytest.skip("not yet implemented")  # TODO: implement


class TestCovariance:
    def test_covariance(self):
        # assert_equal(expected, covariance(data1, data2))
        pytest.skip("not yet implemented")  # TODO: implement


class TestNormalPdf:
    def test_normal_pdf(self):
        # assert_equal(expected, normal_pdf(x, mu, sigma))
        pytest.skip("not yet implemented")  # TODO: implement


class TestDiscrete:
    def test___call__(self):
        # discrete = Discrete(data)
        # assert_equal(expected, discrete.__call__(x))
        # TODO: implement  # implement your test here
        pytest.skip("not yet implemented")

    def test___init__(self):
        # discrete = Discrete(data)
        # TODO: implement  # implement your test here
        pytest.skip("not yet implemented")


class TestPDF:
    def test___call__(self):
        # p_d_f = PDF(pdf, data)
        # assert_equal(expected, p_d_f.__call__(x, **kwargs))
        # TODO: implement  # implement your test here
        pytest.skip("not yet implemented")

    def test___init__(self):
        # p_d_f = PDF(pdf, data)
        # TODO: implement  # implement your test here
        pytest.skip("not yet implemented")
