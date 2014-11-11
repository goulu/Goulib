from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *

from Goulib.itertools2 import *

class TestTake:
    def test_take(self):
        assert_equal(take(3, irange(1,10)),[1,2,3])

class TestIndex:
    def test_index(self):
        assert_equal(index(4, irange(1,10)),5)
        assert_equal(index(9, irange(1,10)),10) # index=-1 doesn't work

class TestFirst:
    def test_first(self):
        assert_equal(first(irange(1,10)),1)
        assert_equal(first('abc'),'a')

class TestLast:
    def test_last(self):
        assert_equal(last(irange(1,10)),10)

class TestTakeEvery:
    def test_take_every(self):
        assert_equal(take_every(2, irange(1,10)),[1,3,5,7,9])

class TestDrop:
    def test_drop(self):
        assert_equal(drop(5, irange(1,10)),[6,7,8,9,10])

class TestIlen:
    def test_ilen(self):
        assert_equal(ilen(irange(10,0)),0)
        assert_equal(ilen(irange(11,20)),10)

class TestIrange:
    def test_irange(self):
        assert_equal(irange(1,5),[1,2,3,4,5])

class TestArange:
    def test_arange(self):
        assert_equal(arange(-1,2.5,.5),[-1,-0.5,0,0.5,1,1.5,2])

class TestIlinear:
    def test_ilinear(self):
        assert_equal(ilinear(-1,2,7),[-1,-0.5,0,0.5,1,1.5,2])

class TestFlatten:
    def test_flatten(self):
        f=list(flatten([[1,2],[3]]))
        assert_equal(f,[1,2,3])
        assert_equal(flatten([1,[2,[3]]]),[1,2,3])
        assert_equal(flatten(['a',['bc']]),['a','bc']) #do not recurse in strings
        assert_equal(flatten([[[1],(2,[3])]],(tuple)),[1,(2,[3])]) # do not recurse in tuple

class TestCompact:
    def test_compact(self):
        assert_equal(compact([None,1,2,None,3,None]),[1,2,3])

class TestGroups:
    def test_groups(self):
        # assert_equal(expected, groups(iterable, n, step))
        raise SkipTest 

class TestCompose:
    def test_compose(self):
        # assert_equal(expected, compose(f, g))
        raise SkipTest 

class TestIterate:
    def test_iterate(self):
        # assert_equal(expected, iterate(func, arg))
        raise SkipTest 

class TestTails:
    def test_tails(self):
        # assert_equal(expected, tails(seq))
        raise SkipTest 

class TestIreduce:
    def test_ireduce(self):
        import operator
        assert_equal(ireduce(operator.add, irange(10)),[1,3,6,10,15,21,28,36,45,55])

class TestUnique:
    def test_unique(self):
        assert_equal(''.join(unique('AAAABBBCCDAABBB')),'ABCD')
        assert_equal(''.join(unique('ABBCcAD', str.lower)),'ABCD')

class TestIdentity:
    def test_identity(self):
        # assert_equal(expected, identity(x))
        raise SkipTest 

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
        raise SkipTest 

class TestTakeevery:
    def test_takeevery(self):
        # assert_equal(expected, takeevery(n, iterable))
        raise SkipTest 

class TestIcross:
    def test_icross(self):
        # assert_equal(expected, icross(*sequences))
        raise SkipTest 

class TestGetGroups:
    def test_get_groups(self):
        # assert_equal(expected, get_groups(iterable, n, step))
        raise SkipTest 

class TestQuantify:
    def test_quantify(self):
        # assert_equal(expected, quantify(iterable, pred))
        raise SkipTest 

class TestPairwise:
    def test_pairwise(self):
        assert_equal(pairwise([1,2,3]),[(1,2),(2,3)])
        assert_equal(pairwise([1,2,3],True),[(1,2),(2,3),(3,1)])
        assert_equal(pairwise([]),[])
        assert_equal(pairwise([1]),[])
        assert_equal(pairwise([1],True),[(1,1)])
    
class TestInterleave:
    def test_interleave(self):
        assert_equal(interleave([0,2,4],[1,3,5]),[0,1,2,3,4,5])
        assert_equal(interleave([0,2,4],[1,3]),[0,1,2,3,4])
        assert_equal(interleave([0],[]),[0])

class TestRandSeq:
    def test_rand_seq(self):
        # assert_equal(expected, rand_seq(size))
        raise SkipTest 

class TestAllPairs:
    def test_all_pairs(self):
        # assert_equal(expected, all_pairs(size))
        raise SkipTest 
    
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
        raise SkipTest 

class TestIter2:
    def test___add__(self):
        i1 = iter2(irange(1,5))
        i2 = iter2(irange(6,10))
        assert_equal(i1+i2,range(1,11))

    def test___init__(self):
        # iter2 = iter2(iterable)
        raise SkipTest 

    def test___iter__(self):
        # iter2 = iter2(iterable)
        # assert_equal(expected, iter2.__iter__())
        raise SkipTest 

    def test_append(self):
        # iter2 = iter2(iterable)
        # assert_equal(expected, iter2.append(iterable))
        raise SkipTest 

    def test_insert(self):
        # iter2 = iter2(iterable)
        # assert_equal(expected, iter2.insert(place, iterable))
        raise SkipTest 

    def test_next(self):
        # iter2 = iter2(iterable)
        # assert_equal(expected, iter2.next())
        raise SkipTest 

    def test___next__(self):
        # iter2 = iter2(iterable)
        # assert_equal(expected, iter2.__next__())
        raise SkipTest 

class TestIflatten:
    def test_iflatten(self):
        # assert_equal(expected, iflatten(iterable))
        raise SkipTest 

class TestCartesianProduct:
    def test_cartesian_product(self):
        #test case for compatibility with itertools.product
        arrays = [(-1,+1), (-2,+2), (-3,+3)]
        res=cartesian_product(*arrays)
        assert_equal(res,[(-1, -2, -3), (-1, -2, 3), (-1, 2, -3), (-1, 2, 3), (1, -2, -3), (1, -2, 3), (1, 2, -3), (1, 2, 3)])

        #test case from http://stackoverflow.com/questions/12093364/cartesian-product-of-large-iterators-itertools
        import itertools
        g = cartesian_product(lambda: itertools.permutations(range(100)),
            lambda: itertools.permutations(range(100)))
        
        assert_equal(next(g),(range(100),range(100)))
        
class TestCountUnique:
    def test_count_unique(self):
        # assert_equal(expected, count_unique(iterable, key))
        raise SkipTest 

class TestGrouped:
    def test_grouped(self):
        # assert_equal(expected, grouped(iterable, n))
        raise SkipTest 

class TestOccurrences:
    def test_occurrences(self):
        # assert_equal(expected, occurrences(it, exchange))
        raise SkipTest 

class TestBest:
    def test_best(self):
        assert_equal(best([3,2,1,2,1]),[1,1])
        assert_equal(best([3,2,1,2,1],reverse=True,n=2),[3,2,2])

if __name__ == "__main__":
    runmodule()