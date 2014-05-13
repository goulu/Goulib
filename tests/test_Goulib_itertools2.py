from nose.tools import assert_equal, assert_true, assert_false
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
        assert_equal(list(compact([None,1,2,None,3,None])),[1,2,3])

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
        assert_equal(''.join(unique('AAAABBBCCDAABBB')),'ABCD')
        assert_equal(''.join(unique('ABBCcAD', str.lower)),'ABCD')

class TestIdentity:
    def test_identity(self):
        # assert_equal(expected, identity(x))
        raise SkipTest # TODO: implement your test here

class TestAny:
    def test_any(self):
        assert_true(any((1,2,3,4),lambda x:x>3))
        assert_false(any((1,2,3,4),lambda x:x>4))

class TestAll:
    def test_all(self):
        assert_true(all((1,2,3,4),lambda x:x<5))
        assert_false(all((1,2,3,4),lambda x:x<4))
        
class TestNo:
    def test_no(self):
        assert_true(no((1,2,3,4),lambda x:x<1))
        assert_false(no((1,2,3,4),lambda x:x<2))

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
        assert_equal(list(pairwise([1,2,3])),[(1,2),(2,3)])
        # assert_equal(list(pairwise([1,2,3],True)),[(1,2),(2,3),(3,1)])
    
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
    
class TestFilter2:
    def test_filter2(self):
        yes,no=filter2([1,2,3,4,3,2,1],lambda x:x<3)
        assert_equal(yes,[1,2,2,1])
        assert_equal(no,[3,4,3])

class TestIfind:
    def test_ifind(self):
        pass #tested below

class TestFind:
    def test_find(self):
        assert_equal(find([0,1,2,3,4],lambda x:x>2),(3,3))

class TestIsplit:
    def test_isplit(self):
        pass #tested below

class TestSplit:
    def test_split(self):
        assert_equal(split([0,1,2,-1,3,4,5], lambda x:x<0),[[0,1,2],[3,4,5]])
        assert_equal(split([-1,0,1,2,-1,3,4,5,-1], lambda x:x<0),[[],[0,1,2],[3,4,5],[]])
        assert_equal(split([-1,0,1,2,-1,3,4,5,-1], lambda x:x<0,True),[[],[-1,0,1,2],[-1,3,4,5],[-1]])

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

class TestGrouped:
    def test_grouped(self):
        # assert_equal(expected, grouped(iterable, n))
        raise SkipTest # TODO: implement your test here

class TestOccurrences:
    def test_occurrences(self):
        # assert_equal(expected, occurrences(it, exchange))
        raise SkipTest # TODO: implement your test here

class TestBest:
    def test_best(self):
        assert_equal(list(best([3,2,1,2,1])),[1,1])
        assert_equal(list(best([3,2,1,2,1],reverse=True,n=2)),[3,2,2])

if __name__ == "__main__":
    import nose
    nose.runmodule()