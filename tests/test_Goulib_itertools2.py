#!/usr/bin/env python
# coding: utf8
from nose.tools import assert_equal, assert_not_equals
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
        l=list(arange(1,step=.01))
        assert_equal(len(l),100)

class TestLinspace:
    def test_linspace(self):
        assert_equal(linspace(-1,2,7),[-1,-0.5,0,0.5,1,1.5,2])
        assert_equal(linspace(1,1,7),[1,1,1,1,1,1,1])
        assert_equal(linspace((1,0),(0,1),3),[(1,0),(.5,.5),(0,1)])

class TestFlatten:
    def test_flatten(self):
        f=list(flatten([[1,2],[3]]))
        assert_equal(f,[1,2,3])
        assert_equal(flatten([1,[2,[3]]]),[1,2,3])
        assert_equal(flatten(['a',['bc']]),['a','bc']) #do not recurse in strings
        assert_equal(flatten([[[1],(2,[3])]],(tuple)),[1,(2,[3])]) # do not recurse in tuple
        d=dict(enumerate(range(10)))
        assert_equal(flatten(d),range(10))


class TestCompact:
    def test_compact(self):
        assert_equal(compact([None,1,2,None,3,None]),[1,2,3])

class TestGroups:
    def test_groups(self):
        assert_equal(groups(irange(1,6),3,2),[[1,2,3],[3,4,5]])
        assert_equal(groups([1,2,3,4,5,6],3,2),[[1,2,3],[3,4,5]])
        assert_equal(groups([1,2,3,4,5,6],3),[[1,2,3],[4,5,6]])
        assert_equal(groups([1,2,3,4,5,6],4),[[1,2,3,4]])

class TestReshape:
    def test_reshape(self):
        data=[1,[2,[3,4],[5,6,7]]] #data can have any shape...
        assert_equal(reshape(data,(2,3)),[[1,2,3],[4,5,6]])
        assert_equal(reshape(data,(3,2)),[[1,2],[3,4],[5,6]])
        assert_equal(reshape(data,(3,3)),[[1,2,3],[4,5,6],[7]])

class TestCompose:
    def test_compose(self):
        from math import sin
        f=compose(sin, lambda x:x*x)
        assert_equal(f(2),sin(4))

class TestIterate:
    def test_iterate(self):
        assert_equal(take(4,iterate(lambda x:x*x, 2)), [2,4,16,16*16])

class TestIsIterable:
    def test_isiterable(self):
        assert_false(isiterable(123))
        assert_false(isiterable('a string'))
        assert_true(isiterable([]))
        assert_true(isiterable(tuple()))
        assert_true(isiterable({}))
        assert_true(isiterable(set()))
        assert_true(isiterable((x for x in range(10))))
        assert_true(isiterable(map(lambda x:x*x,[1,2,3])))


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
        assert_equal(''.join(unique('AAAABBBCCDAABBB',None,1)),'ABCDAB')

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
        assert_equal(pairwise([1,2,3],operator.add),[3,5])
        assert_equal(pairwise([1,2,3],loop=True),[(1,2),(2,3),(3,1)])
        assert_equal(pairwise([1,2,3],operator.add,loop=True),[3,5,4])
        assert_equal(pairwise([]),[])
        assert_equal(pairwise([1]),[])
        assert_equal(pairwise([1],loop=True),[(1,1)])

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
        res=take(10,next_permutation(list('hello')))
        res=[''.join(x) for x in res]
        res=','.join(res)
        assert_equal(res,'hello,helol,heoll,hlelo,hleol,hlleo,hlloe,hloel,hlole,hoell')

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

class TestCombinationsWithReplacement:
    def test_combinations_with_replacement(self):
        assert_equal(combinations_with_replacement('ABC', 2),
            ['AA','AB','AC','BB','BC','CC'])
        assert_equal(combinations_with_replacement('AB', 4),
            ['AAAA','AAAB','AABB','ABBB','BBBB'])

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

class TestCompress:
    def test_compress(self):
        # https://www.linkedin.com/groups/25827/25827-6166706414627627011
        res=compress('aaaaabbbbccccccaaaaaaa')
        res=''.join('%d%s'%(n,c) for (c,n) in res)
        assert_equal(res,'5a4b6c7a')

class TestAccumulate:
    def test_accumulate(self):
        # assert_equal(expected, accumulate(iterable, func, skip_first))
        raise SkipTest

class TestDiff:
    def test_diff(self):
        # assert_equal(expected, diff(iterable1, iterable2))
        raise SkipTest

class TestSortedIterable:
    def test_sorted_iterable(self):
        data=[1,2,3,7,6,5,4]
        res=sorted(data)
        #with a small buffer, it fails
        def test(iterable,buffer,key=None):
            return [x for x in ensure_sorted(
                sorted_iterable(iterable,key=key, buffer=buffer)
                ,key=key)]
        assert_raises(SortingError,test,data,3)
        #with a larger one, it's ok
        assert_equal(test(data,buffer=4),res)


class TestIsiterable:
    def test_isiterable(self):
        assert_true(isiterable(list()))
        assert_true(isiterable(tuple()))
        assert_true(isiterable(range(1000)))
        assert_false(isiterable(''))

class TestItemgetter:
    def test_itemgetter(self):
        # assert_equal(expected, itemgetter(iterable, i))
        raise SkipTest

class TestTee:
    def test_tee(self):
        it=count()
        it,it1,it2=tee(it,n=3)
        assert_equal(next(it1),next(it2))
        assert_equal(next(it1),next(it2))
        assert_equal(next(it),0)

class TestIremove:
    def test_iremove(self):
        # assert_equal(expected, iremove(iterable, f))
        raise SkipTest

class TestDictsplit:
    def test_dictsplit(self):
        # assert_equal(expected, dictsplit(dic, keys))
        raise SkipTest

class TestShape:
    def test_shape(self):
        data=[[[5,6,7],2,[3,4]],1] #data can have any shape...
        assert_equal(shape(data),(2,3,3)) #... but shape is evaluated from [0]

class TestNdim:
    def test_ndim(self):
        data=[[[5,6,7],2,[3,4]],1] #data can have any shape...
        assert_equal(ndim(data),3) #... but shape is evaluated from [0]

class TestEnumerates:
    def test_enumerates(self):
        r=range(10)
        d=dict(enumerate(r))
        assert_equal(enumerates(d),enumerates(r))

class TestEnsureSorted:
    def test_ensure_sorted(self):
        # assert_equal(expected, ensure_sorted(iterable, key))
        raise SkipTest # TODO: implement your test here

class TestIscallable:
    def test_iscallable(self):
        # assert_equal(expected, iscallable(f))
        raise SkipTest # TODO: implement your test here

class TestIntersect:
    def test_intersect(self):
        # http://stackoverflow.com/questions/969709/joining-a-set-of-ordered-integer-yielding-python-iterators
        postings = [[1,   100, 142, 322, 12312],
            [2,   100, 101, 322, 1221],
            [100, 142, 322, 956, 1222]]

        assert_equal(intersect(*postings),[100, 322])

class TestKeep:
    @classmethod
    def setup_class(self):
        l=[1,2,3,4,5,6,7,8,9]
        k=keep(l)
        kl=list(k)
        assert_equal(kl,l)
        assert_equal(k.val,l[-1])
        
    def test___init__(self):
        pass #tested in test_detect_cycle

    def test___iter__(self):
        pass #tested in test_detect_cycle

    def test_next(self):
        pass #tested in test_detect_cycle

    def test___next__(self):
        # keep = keep(iterable)
        # assert_equal(expected, keep.__next__())
        raise SkipTest # TODO: implement your test here

class TestFirstMatch:
    def test_first_match(self):
        pass #tested in test_detect_cycle

class TestDetectCycle:
    def test_detect_cycle(self):
        
        assert_equal(detect_cycle(list('123412341')),(0,4))
        
        assert_equal(detect_cycle(list('012345'+'678'*4)),(6,3))
        # but the repetition should be long enough (2*i ?):
        assert_equal(detect_cycle(list('012345'+'678'*3)),(None,None))
        
        #test from https://rosettacode.org/wiki/Cycle_detection
        assert_equal(detect_cycle([3,10,101,2,5,26,167,95,101,2,5,26,167,95]),(2,6))
        
        """ does not work yet because of repeating digits
        
        p3=[1, 1, 2, 0, 2, 2, 1, 0, 1, 1, 2, 0, 2, 2, 1, 0, 1, 1, 2, 0, 2, 2, 
            1, 0, 1, 1, 2, 0, 2, 2, 1, 0, 1, 1, 2, 0, 2, 2, 1, 0, 1, 1, 2, 0, 2, 
            2, 1, 0, 1, 1, 2, 0, 2, 2, 1, 0, 1, 1, 2, 0, 2, 2, 1, 0, 1, 1, 2, 0, 
            2, 2, 1, 0, 1, 1, 2, 0, 2, 2, 1, 0, 1, 1, 2, 0, 2, 2, 1, 0, 1, 1, 2, 
            0, 2, 2, 1, 0, 1, 1, 2, 0, 2, 2]
        assert_equal(detect_cycle(p3)[1],8)
        
        from math import pi
        assert_equal(detect_cycle(list(str(pi))),(None,None)) #TODO: find why it's wrong
        """
        
class TestFloyd:
    def test_floyd(self):
        # assert_equal(expected, floyd(iterable, limit))
        raise SkipTest # TODO: implement your test here

if __name__ == "__main__":
    runmodule()
