from goulib.tests import *

from goulib.itertools2 import *


class TestTake:
    def test_take(self):
        assert list(take(3, irange(1, 10))) == [1, 2, 3]


class TestIndex:
    def test_index(self):
        assert index(4, irange(1, 10)) == 3
        assert index(9, irange(1, 10)) == 8


class TestFirst:
    def test_first(self):
        assert first(irange(1, 10)) == 1
        assert first('abc') == 'a'


class TestLast:
    def test_last(self):
        assert last(irange(1, 10)) == 10


class TestTakeEvery:
    def test_take_every(self):
        assert list(every(2, irange(1, 10))) == [1, 3, 5, 7, 9]
        assert list(takeevery(3, irange(1, 10))) == [1, 4, 7, 10]


class TestDrop:
    def test_drop(self):
        assert list(drop(5, irange(1, 10))) == [6, 7, 8, 9, 10]


class TestIlen:
    def test_ilen(self):
        assert ilen(irange(10, 0)) == 0
        assert ilen(irange(11, 20)) == 10


class TestIrange:
    def test_irange(self):
        assert list(irange(1, 5)) == [1, 2, 3, 4, 5]


class TestArange:
    def test_arange(self):
        assert list(arange(-1, 2.5, .5)) == [-1, -0.5, 0, 0.5, 1, 1.5, 2]
        assert list(arange(2, -1.5, .5)
                    ) == list(reversed([-1, -0.5, 0, 0.5, 1, 1.5, 2]))
        l = list(arange(1, step=.01))
        assert len(l) == 100


class TestLinspace:
    def test_linspace(self):
        assert list(linspace(-1, 2, 7)) == [-1, -0.5, 0, 0.5, 1, 1.5, 2]
        assert list(linspace(1, 1, 7)) == [1, 1, 1, 1, 1, 1, 1]
        assert list(linspace((1, 0), (0, 1), 3)) == [(1, 0), (.5, .5), (0, 1)]


class TestFlatten:
    def test_flatten(self):
        f = list(flatten([[1, 2], [3]]))
        assert f == [1, 2, 3]
        assert list(flatten([1, [2, [3]]])) == [1, 2, 3]
        # do not recurse in strings
        assert list(flatten(['a', ['bc']])) == ['a', 'bc']
        assert list(flatten([[[1], (2, [3])]], (tuple))) == [
            1, (2, [3])]  # do not recurse in tuple
        d = dict(enumerate(range(10)))
        assert list(flatten(d)) == list(range(10))


class TestGroups:
    def test_groups(self):
        # assert groups(irange(1, 6), 3, 2) == [[1, 2, 3], [3, 4, 5]]
        assert list(groups([1, 2, 3, 4, 5, 6], 3, 2)) == [(1, 2, 3), (3, 4, 5)]
        assert list(groups([1, 2, 3, 4, 5, 6], 3)) == [(1, 2, 3), (4, 5, 6)]
        assert list(groups([1, 2, 3, 4, 5, 6], 4)) == [(1, 2, 3, 4)]


class TestReshape:
    def test_reshape(self):
        data = [1, [2, [3, 4], [5, 6, 7]]]  # data can have any shape...
        assert reshape(data, (2, 3)) == [[1, 2, 3], [4, 5, 6]]
        assert reshape(data, (3, 2)) == [[1, 2], [3, 4], [5, 6]]
        assert reshape(data, (3, 3)) == [[1, 2, 3], [4, 5, 6], [7]]


class TestCompose:
    def test_compose(self):
        from math import sin
        f = compose(sin, lambda x: x*x)
        assert f(2) == sin(4)


class TestIterate:
    def test_iterate(self):
        assert list(take(4, iterate(lambda x: x*x, 2))) == [2, 4, 16, 16*16]


class TestIsIterable:
    def test_isiterable(self):
        assert not isiterable(123)
        assert not isiterable('a string')
        assert isiterable([])
        assert isiterable(tuple())
        assert isiterable({})
        assert isiterable(set())
        assert isiterable((x for x in range(10)))
        assert isiterable(map(lambda x: x*x, [1, 2, 3]))


class TestTails:
    def test_tails(self):
        assert list(tails([1, 2, 3])) == [[1, 2, 3], [2, 3], [3], []]


class TestIreduce:
    def test_ireduce(self):
        import operator
        assert list(ireduce(operator.add, irange(10))) == [
            1, 3, 6, 10, 15, 21, 28, 36, 45, 55]
        assert list(ireduce(operator.add, irange(10), 2),) == [
            2, 2, 3, 5, 8, 12, 17, 23, 30, 38, 47, 57]


class TestCompress:
    def test_compress(self):
        assert list(compress('AAAABBBCCDAABBB')) == [
            ('A', 4), ('B', 3), ('C', 2), ('D', 1), ('A', 2), ('B', 3)]
        # https://www.linkedin.com/groups/25827/25827-6166706414627627011
        res = compress('aaaaabbbbccccccaaaaaaa')
        res = ''.join('%d%s' % (n, c) for (c, n) in res)
        assert res == '5a4b6c7a'


class TestDecompress:
    def test_decompress(self):
        data = 'aaaaabbbbccccccaaaaaaa'
        res = compress(data)
        data2 = decompress(res)
        assert ''.join(map(str, data2)) == data


class TestUnique:
    def test_unique(self):
        assert ''.join(map(str, unique('AAAABBBCCDAABBB'))) == 'ABCD'
        assert ''.join(map(str, unique('ABBCcAD', str.upper))) == 'ABCD'
        assert ''.join(
            map(str, unique('ZZZZBBBCCDAABBB', buffer=1))) == 'ZBCDAB'
        # harmless regression ...
        # s=list(unique('AAAABBBCCDAABBB',buffer=4))
        # assert_equal(s,'ABCD')


class TestIdentity:
    def test_identity(self):
        x = object()
        assert identity(x) == x


class TestAny:
    def test_any(self):
        assert any((1, 2, 3, 4), lambda x: x > 3)
        assert not any((1, 2, 3, 4), lambda x: x > 4)


class TestAll:
    def test_all(self):
        assert all((1, 2, 3, 4), lambda x: x < 5)
        assert not all((1, 2, 3, 4), lambda x: x < 4)


class TestNo:
    def test_no(self):
        assert no((1, 2, 3, 4), lambda x: x < 1)
        assert not no((1, 2, 3, 4), lambda x: x < 2)


class TestTakenth:
    def test_takenth(self):
        # http://stackoverflow.com/questions/12007820/better-ways-to-get-nth-element-from-an-unsubscriptable-iterable
        from itertools import permutations
        assert (nth(1000, permutations(range(10), 10)) ==
                (0, 1, 2, 4, 6, 5, 8, 9, 3, 7))


class TestIcross:
    def test_icross(self):
        assert list(icross([1, 2, 5], [2, 3])) == [
            (1, 2), (1, 3), (2, 2), (2, 3), (5, 2), (5, 3)]


class TestQuantify:
    def test_quantify(self):
        from goulib.math2 import is_pentagonal
        assert quantify(irange(1, 100), is_pentagonal) == 8


class TestPairwise:
    def test_pairwise(self):
        assert list(pairwise([1, 2, 3])) == [(1, 2), (2, 3)]
        assert list(pairwise([1, 2, 3], operator.add)) == [3, 5]
        assert list(pairwise([1, 2, 3], loop=True)) == [(1, 2), (2, 3), (3, 1)]
        assert list(pairwise([1, 2, 3], operator.add, loop=True)) == [3, 5, 4]
        assert list(pairwise([])) == []
        assert list(pairwise([1])) == []
        assert list(pairwise([1], loop=True)) == [(1, 1)]


class TestInterleave:
    def test_interleave(self):
        assert interleave([0, 2, 4], [1, 3, 5]) == [0, 1, 2, 3, 4, 5]
        assert interleave([0, 2, 4], [1, 3]) == [0, 1, 2, 3, 4]
        assert interleave([0], []) == [0]


class TestRandSeq:
    def test_rand_seq(self):
        # assert_equal(expected, rand_seq(size))
        pytest.skip("not yet implemented")  # TODO: implement


class TestAllPairs:
    def test_all_pairs(self):
        # assert_equal(expected, all_pairs(size))
        pytest.skip("not yet implemented")  # TODO: implement


class TestFilter2:
    def test_filter2(self):
        yes, no = filter2([1, 2, 3, 4, 3, 2, 1], lambda x: x < 3)
        assert yes == [1, 2, 2, 1]
        assert no == [3, 4, 3]


class TestIfind:
    def test_ifind(self):
        pass  # tested below


class TestFind:
    def test_find(self):
        assert find([0, 1, 2, 3, 4], lambda x: x > 2) == (3, 3)


class TestIsplit:
    def test_isplit(self):
        pass  # tested below


class TestSplit:
    def test_split(self):
        assert split([0, 1, 2, -1, 3, 4, 5], lambda x: x <
                     0) == [[0, 1, 2], [3, 4, 5]]
        assert split([-1, 0, 1, 2, -1, 3, 4, 5, -1], lambda x: x <
                     0) == [[], [0, 1, 2], [3, 4, 5], []]
        assert split([-1, 0, 1, 2, -1, 3, 4, 5, -1], lambda x: x <
                     0, True) == [[], [-1, 0, 1, 2], [-1, 3, 4, 5], [-1]]


class TestNextPermutation:
    def test_next_permutation(self):
        res = take(10, next_permutation(list('hello')))
        res = [''.join(x) for x in res]
        res = ','.join(res)
        assert res == 'hello,helol,heoll,hlelo,hleol,hlleo,hlloe,hloel,hlole,hoell'


class TestIter2(TestCase):
    def test___add__(self):
        i1 = iter2(irange(1, 5))
        i2 = iter2(irange(6, 10))
        self.assertEqual(i1+i2, range(1, 11))

    def test___init__(self):
        # iter2 = iter2(iterable)
        pytest.skip("not yet implemented")  # TODO: implement

    def test___iter__(self):
        # iter2 = iter2(iterable)
        # assert_equal(expected, iter2.__iter__())
        pytest.skip("not yet implemented")  # TODO: implement

    def test_append(self):
        # iter2 = iter2(iterable)
        # assert_equal(expected, iter2.append(iterable))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_insert(self):
        # iter2 = iter2(iterable)
        # assert_equal(expected, iter2.insert(place, iterable))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_next(self):
        # iter2 = iter2(iterable)
        # assert_equal(expected, iter2.next())
        pytest.skip("not yet implemented")  # TODO: implement

    def test___next__(self):
        # iter2 = iter2(iterable)
        # assert_equal(expected, iter2.__next__())
        pytest.skip("not yet implemented")  # TODO: implement


class TestProduct:
    def test_product(self):
        # test compatibility with itertools.product
        assert list(itertools2.product()) == list(itertools.product())
        assert list(itertools2.product([])) == list(itertools.product([]))
        assert list(itertools2.product('ABCD', 'xy')) == list(
            itertools.product('ABCD', 'xy'))
        assert list(itertools2.product(range(2), repeat=3)) == list(
            itertools.product(range(2), repeat=3))

        # test case from http://stackoverflow.com/questions/12093364/cartesian-product-of-large-iterators-itertools

        g = product(itertools.permutations(range(100)), repeat=2)
        l100 = tuple(range(100))

        assert next(g) == (l100, l100)


class TestCombinationsWithReplacement:
    def test_combinations_with_replacement(self):
        assert [''.join(map(str, x)) for x in combinations_with_replacement('ABC', 2)] == [
            'AA', 'AB', 'BB', 'AC', 'BC', 'CC']
        assert [''.join(map(str, x)) for x in combinations_with_replacement('AB', 4)] == [
            'AAAA', 'AAAB', 'AABB', 'ABBB', 'BBBB']


class TestCountUnique:
    def test_count_unique(self):
        assert count_unique('AAAABBBCCDAABBB') == 4
        assert count_unique('ABBCcAD', str.lower) == 4


class TestBest:
    def test_best(self):
        assert list(best([3, 2, 1, 2, 1])) == [1, 1]
        assert list(best([3, 2, 1, 2, 1], reverse=True, n=2)) == [3, 2, 2]


class TestRemovef:
    def test_removef(self):
        l = [0, 1, 'a', None, 3.14, []]
        r = removef(l, lambda x: True if not x else False)
        assert r == [0, None, []]
        assert l == [1, 'a', 3.14]


class TestShuffle:
    def test_shuffle(self):
        s1 = list("hello world")
        s2 = shuffle(list("hello world"))  # copy, as shuffle works in place
        assert s1 != s2  # would really be bad luck ...
        assert occurences(s1) == occurences(s2)


class TestIndexMin:
    def test_index_min(self):
        assert index_min("hallo~welt") == (1, 'a')


class TestIndexMax:
    def test_index_max(self):
        assert index_max("hello world") == (6, 'w')


class TestTakeevery:
    def test_takeevery(self):
        # assert_equal(expected, takeevery(n, iterable))
        pytest.skip("not yet implemented")  # TODO: implement


class TestSortIndexes:
    def test_sort_indexes(self):
        # assert_equal(expected, sort_indexes(iterable, key, reverse))
        pytest.skip("not yet implemented")  # TODO: implement


class TestSubdict:
    def test_subdict(self):
        # assert_equal(expected, subdict(d, keys))
        pytest.skip("not yet implemented")  # TODO: implement


class TestAccumulate:
    def test_accumulate(self):
        # assert_equal(expected, accumulate(iterable, func, skip_first))
        pytest.skip("not yet implemented")  # TODO: implement


class TestDiff:
    def test_diff(self):
        # assert_equal(expected, diff(iterable1, iterable2))
        pytest.skip("not yet implemented")  # TODO: implement


class TestSortedIterable:
    def test_sorted_iterable(self):
        data = [1, 2, 3, 7, 6, 5, 4]
        res = sorted(data)
        # with a small buffer, it fails

        def test(iterable, buffer, key=None):
            return [x for x in ensure_sorted(
                sorted_iterable(iterable, key=key, buffer=buffer), key=key)]
        with pytest.raises(SortingError):
            test(data, buffer=3)
        # with a larger one, it's ok
        assert test(data, buffer=4) == res


class TestIsiterable:
    def test_isiterable(self):
        assert isiterable(list())
        assert isiterable(tuple())
        assert isiterable(range(1000))
        assert not isiterable('')


class TestItemgetter:
    def test_itemgetter(self):
        # assert_equal(expected, itemgetter(iterable, i))
        pytest.skip("not yet implemented")  # TODO: implement


class TestTee:
    def test_tee(self):
        it = itertools.count()
        it, it1, it2 = tee(it, n=3)
        assert next(it1) == next(it2)
        assert next(it1) == next(it2)
        assert next(it) == 0


class TestIremove:
    def test_iremove(self):
        # assert_equal(expected, iremove(iterable, f))
        pytest.skip("not yet implemented")  # TODO: implement


class TestDictsplit:
    def test_dictsplit(self):
        # assert_equal(expected, dictsplit(dic, keys))
        pytest.skip("not yet implemented")  # TODO: implement


class TestShape:
    def test_shape(self):
        data = [[[5, 6, 7], 2, [3, 4]], 1]  # data can have any shape...
        assert shape(data) == [2, 3, 3]  # ... but shape is evaluated from [0]


class TestNdim:
    def test_ndim(self):
        data = [[[5, 6, 7], 2, [3, 4]], 1]  # data can have any shape...
        assert ndim(data) == 3  # ... but shape is evaluated from [0]


class TestEnumerates:
    def test_enumerates(self):
        r = range(10)
        d = dict(enumerate(r))
        assert list(enumerates(d)) == list(enumerates(r))


class TestEnsureSorted:
    def test_ensure_sorted(self):
        # assert_equal(expected, ensure_sorted(iterable, key))
        # TODO: implement  # implement your test here
        pytest.skip("not yet implemented")


class TestIscallable:
    def test_iscallable(self):
        # assert_equal(expected, iscallable(f))
        # TODO: implement  # implement your test here
        pytest.skip("not yet implemented")


class TestIntersect:
    def test_intersect(self):
        # http://stackoverflow.com/questions/969709/joining-a-set-of-ordered-integer-yielding-python-iterators
        postings = [[1,   100, 142, 322, 12312],
                    [2,   100, 101, 322, 1221],
                    [100, 142, 322, 956, 1222]]

        assert list(intersect(*postings)) == [100, 322]


class TestKeep:
    @classmethod
    def setup_class(self):
        l = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        k = keep(l)
        kl = list(k)
        assert kl == l
        assert k.val == l[-1]

    def test___init__(self):
        pass  # tested in test_detect_cycle

    def test___iter__(self):
        pass  # tested in test_detect_cycle

    def test_next(self):
        pass  # tested in test_detect_cycle

    def test___next__(self):
        # keep = keep(iterable)
        # assert_equal(expected, keep.__next__())
        # TODO: implement  # implement your test here
        pytest.skip("not yet implemented")


class TestFirstMatch:
    def test_first_match(self):
        pass  # tested in test_detect_cycle


class TestDetectCycle:
    def test_detect_cycle(self):

        assert detect_cycle(list('123412341')) == (0, 4)

        assert detect_cycle(list('012345'+'678'*4)) == (6, 3)
        assert detect_cycle(list('012345'+'678'*3)) == (6, 3)

        # Floyd fails when repetition isn't long enough (2*i ?):
        assert floyd(list('012345'+'678'*3)) == (None, None)

        # test from https://rosettacode.org/wiki/Cycle_detection
        assert detect_cycle([3, 10, 101, 2, 5, 26, 167, 95,
                            101, 2, 5, 26, 167, 95]) == (2, 6)

        """does not work yet because of repeating digits

        p3=[1, 1, 2, 0, 2, 2, 1, 0, 1, 1, 2, 0, 2, 2, 1, 0, 1, 1, 2, 0, 2, 2,
            1, 0, 1, 1, 2, 0, 2, 2, 1, 0, 1, 1, 2, 0, 2, 2, 1, 0, 1, 1, 2, 0, 2,
            2, 1, 0, 1, 1, 2, 0, 2, 2, 1, 0, 1, 1, 2, 0, 2, 2, 1, 0, 1, 1, 2, 0,
            2, 2, 1, 0, 1, 1, 2, 0, 2, 2, 1, 0, 1, 1, 2, 0, 2, 2, 1, 0, 1, 1, 2,
            0, 2, 2, 1, 0, 1, 1, 2, 0, 2, 2]
        assert_equal(detect_cycle(p3)[1],8)
        """
        from goulib.math2 import pi_digits_gen
        assert detect_cycle(pi_digits_gen()) == (1, 2)  # same problem ..


class TestFloyd:
    def test_floyd(self):
        # assert_equal(expected, floyd(iterable, limit))
        # TODO: implement  # implement your test here
        pytest.skip("not yet implemented")
