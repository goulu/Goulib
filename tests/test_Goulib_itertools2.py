from nose.tools import assert_equal
from nose import SkipTest

from Goulib.itertools2 import *

def iterable(n=10,s=1):
    for i in range(s,n+1):
        yield i

class TestTake:
    def test_take(self):
        assert_equal(list(take(3, iterable())),[1,2,3])

class TestIndex:
    def test_index(self):
        assert_equal(index(4, iterable()),5)
        assert_equal(index(9, iterable()),10) # index=-1 doesn't work

class TestFirst:
    def test_first(self):
        assert_equal(first(iterable()),1)

class TestLast:
    def test_last(self):
        assert_equal(last(iterable()),10)

class TestTakeEvery:
    def test_take_every(self):
        assert_equal(list(take_every(2, iterable())),[1,3,5,7,9])

class TestDrop:
    def test_drop(self):
        assert_equal(list(drop(5, iterable())),[6,7,8,9,10])

class TestIlen:
    def test_ilen(self):
        assert_equal(ilen(iterable(0)),0)
        assert_equal(ilen(iterable(10)),10)

class TestIrange:
    def test_irange(self):
        assert_equal(list(irange(1,5)),[1,2,3,4,5])

class TestArange:
    def test_arange(self):
        assert_equal(list(arange(-1,2.5,.5)),[-1,-0.5,0,0.5,1,1.5,2])

class TestIlinear:
    def test_ilinear(self):
        assert_equal(list(ilinear(-1,2,7)),[-1,-0.5,0,0.5,1,1.5,2])

class TestFlatten:
    def test_flatten(self):
        assert_equal(list(flatten([[1,2],[3]])),[1,2,3])

class TestCompact:
    def test_compact(self):
        # assert_equal(expected, compact(it))
        raise SkipTest # TODO: implement your test here

class TestGroups:
    def test_groups(self):
        # assert_equal(expected, groups(iterable, n, step))
        raise SkipTest # TODO: implement your test here

class TestCompose:
    def test_compose(self):
        # assert_equal(expected, compose(f, g))
        raise SkipTest # TODO: implement your test here

class TestIterate:
    def test_iterate(self):
        # assert_equal(expected, iterate(func, arg))
        raise SkipTest # TODO: implement your test here

class TestTails:
    def test_tails(self):
        # assert_equal(expected, tails(seq))
        raise SkipTest # TODO: implement your test here

class TestIreduce:
    def test_ireduce(self):
        # assert_equal(expected, ireduce(func, iterable, init))
        raise SkipTest # TODO: implement your test here

class TestUnique:
    def test_unique(self):
        # assert_equal(expected, unique(iterable, key))
        raise SkipTest # TODO: implement your test here

class TestIdentity:
    def test_identity(self):
        # assert_equal(expected, identity(x))
        raise SkipTest # TODO: implement your test here

class TestOccurrences:
    def test_occurrences(self):
        # assert_equal(expected, occurrences(it, exchange))
        raise SkipTest # TODO: implement your test here

class TestAny:
    def test_any(self):
        # assert_equal(expected, any(seq, pred))
        raise SkipTest # TODO: implement your test here

class TestAll:
    def test_all(self):
        # assert_equal(expected, all(seq, pred))
        raise SkipTest # TODO: implement your test here

class TestNo:
    def test_no(self):
        # assert_equal(expected, no(seq, pred))
        raise SkipTest # TODO: implement your test here

class TestTakenth:
    def test_takenth(self):
        # assert_equal(expected, takenth(n, iterable))
        raise SkipTest # TODO: implement your test here

class TestTakeevery:
    def test_takeevery(self):
        # assert_equal(expected, takeevery(n, iterable))
        raise SkipTest # TODO: implement your test here

class TestIcross:
    def test_icross(self):
        # assert_equal(expected, icross(*sequences))
        raise SkipTest # TODO: implement your test here

class TestGetGroups:
    def test_get_groups(self):
        # assert_equal(expected, get_groups(iterable, n, step))
        raise SkipTest # TODO: implement your test here

class TestQuantify:
    def test_quantify(self):
        # assert_equal(expected, quantify(iterable, pred))
        raise SkipTest # TODO: implement your test here

class TestPairwise:
    def test_pairwise(self):
        # assert_equal(expected, pairwise(iterable))
        raise SkipTest # TODO: implement your test here
    
class TestInterleave:
    def test_interleave(self):
        assert_equal(interleave([0,2,4],[1,3,5]),[0,1,2,3,4,5])
        assert_equal(interleave([0,2,4],[1,3]),[0,1,2,3,4])
        assert_equal(interleave([0],[]),[0])

class TestRandSeq:
    def test_rand_seq(self):
        # assert_equal(expected, rand_seq(size))
        raise SkipTest # TODO: implement your test here

class TestAllPairs:
    def test_all_pairs(self):
        # assert_equal(expected, all_pairs(size))
        raise SkipTest # TODO: implement your test here

class TestSplit:
    def test_split(self):
        # assert_equal(expected, split(iterable, condition))
        raise SkipTest # TODO: implement your test here

class TestNextPermutation:
    def test_next_permutation(self):
        # assert_equal(expected, next_permutation(seq, pred))
        raise SkipTest # TODO: implement your test here

class TestIter2:
    def test___add__(self):
        i1 = iter2(iterable(5))
        i2 = iter2(iterable(10,6))
        assert_equal(list(i1+i2),range(1,11))

    def test___init__(self):
        # iter2 = iter2(iterable)
        raise SkipTest # TODO: implement your test here

    def test___iter__(self):
        # iter2 = iter2(iterable)
        # assert_equal(expected, iter2.__iter__())
        raise SkipTest # TODO: implement your test here

    def test_append(self):
        # iter2 = iter2(iterable)
        # assert_equal(expected, iter2.append(iterable))
        raise SkipTest # TODO: implement your test here

    def test_insert(self):
        # iter2 = iter2(iterable)
        # assert_equal(expected, iter2.insert(place, iterable))
        raise SkipTest # TODO: implement your test here

    def test_next(self):
        # iter2 = iter2(iterable)
        # assert_equal(expected, iter2.next())
        raise SkipTest # TODO: implement your test here

class TestIflatten:
    def test_iflatten(self):
        # assert_equal(expected, iflatten(iterable))
        raise SkipTest # TODO: implement your test here

class TestProduct:
    def test_product(self):
        # assert_equal(expected, product(*iterables, **kwargs))
        raise SkipTest # TODO: implement your test here

class TestCountUnique:
    def test_count_unique(self):
        # assert_equal(expected, count_unique(iterable, key))
        raise SkipTest # TODO: implement your test here

if __name__ == "__main__":
    import nose
    nose.runmodule()