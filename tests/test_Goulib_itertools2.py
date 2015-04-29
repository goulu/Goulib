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
        assert_equal(every(2, irange(1,10)),[1,3,5,7,9])
        assert_equal(takeevery(3,irange(1,10)), [1,4,7,10])

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
        assert_equal(arange(2,-1.5,.5),reversed([-1,-0.5,0,0.5,1,1.5,2]))
        l=list(arange(0,1,.01))
        assert_equal(len(l),100)

class TestIlinear:
    def test_ilinear(self):
        assert_equal(ilinear(-1,2,7),[-1,-0.5,0,0.5,1,1.5,2])
        assert_equal(ilinear(1,1,7),[1,1,1,1,1,1,1])
        assert_equal(ilinear((1,0),(0,1),3),[(1,0),(.5,.5),(0,1)])

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
        assert_equal(groups([1,2,3,4,5,6],3,2),[[1,2,3],[3,4,5]])
        assert_equal(groups([1,2,3,4,5,6],3),[[1,2,3],[4,5,6]]) 
        assert_equal(groups([1,2,3,4,5,6],4),[[1,2,3,4]]) 

class TestCompose:
    def test_compose(self):
        from math import sin
        f=compose(sin, lambda x:x*x)
        assert_equal(f(2),sin(4))

class TestIterate:
    def test_iterate(self):
        assert_equal(take(4,iterate(lambda x:x*x, 2)), [2,4,16,16*16])

class TestTails:
    def test_tails(self):
        assert_equal(tails([1,2,3]),[[1,2,3], [2,3], [3], []])

class TestIreduce:
    def test_ireduce(self):
        import operator
        assert_equal(ireduce(operator.add, irange(10)),[1,3,6,10,15,21,28,36,45,55])
        assert_equal(ireduce(operator.add, irange(10),2),[2,2,3,5,8,12,17,23,30,38,47,57])

class TestUnique:
    def test_unique(self):
        assert_equal(''.join(unique('AAAABBBCCDAABBB')),'ABCD')
        assert_equal(''.join(unique('ABBCcAD', str.lower)),'ABCD')

class TestIdentity:
    def test_identity(self):
        x=object()
        assert_equal(identity(x),x)

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
        #http://stackoverflow.com/questions/12007820/better-ways-to-get-nth-element-from-an-unsubscriptable-iterable
        from itertools import permutations
        assert_equal(nth(1000,permutations(range(10), 10)),
            (0, 1, 2, 4, 6, 5, 8, 9, 3, 7)
        )

class TestIcross:
    def test_icross(self):
        assert_equal(icross([1,2,5],[2,3]),
            [(1,2),(1,3),(2,2),(2,3),(5,2),(5,3)]
        )

class TestQuantify:
    def test_quantify(self):
        from Goulib.math2 import is_pentagonal
        assert_equal(quantify(irange(1,100), is_pentagonal),8)

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
        assert_equal(count_unique('AAAABBBCCDAABBB'),4)
        assert_equal(count_unique('ABBCcAD', str.lower),4)

class TestOccurrences:
    def test_occurrences(self):
        assert_equal(occurrences("hello world"),
            {'e': 1, 'o': 2, 'w': 1, 'r': 1, 'l': 3, 'd': 1, 'h': 1, ' ': 1}
        )

class TestBest:
    def test_best(self):
        assert_equal(best([3,2,1,2,1]),[1,1])
        assert_equal(best([3,2,1,2,1],reverse=True,n=2),[3,2,2])

class TestRemovef:
    def test_removef(self):
        l=[0,1,'a',None,3.14,[]]
        r=removef(l,lambda x:True if not x else False)
        assert_equal(r,[0,None,[]])
        assert_equal(l,[1,'a',3.14])

class TestShuffle:
    def test_shuffle(self):
        s1=list("hello world")
        s2=shuffle(list("hello world")) #copy, as shuffle works in place
        assert_not_equal(s1,s2) #would really be bad luck ...
        assert_equal(occurrences(s1),occurrences(s2))

class TestIndexMin:
    def test_index_min(self):
        assert_equal(index_min("hallo~welt"),(1,'a'))

class TestIndexMax:
    def test_index_max(self):
        assert_equal(index_max("hello world"),(6,'w'))

class TestTakeevery:
    def test_takeevery(self):
        # assert_equal(expected, takeevery(n, iterable))
        raise SkipTest

class TestSortIndexes:
    def test_sort_indexes(self):
        # assert_equal(expected, sort_indexes(iterable, key, reverse))
        raise SkipTest

class TestSubdict:
    def test_subdict(self):
        # assert_equal(expected, subdict(d, keys))
        raise SkipTest

if __name__ == "__main__":
    runmodule()