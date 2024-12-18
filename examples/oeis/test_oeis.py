"""
tests for OEIS.py and database.py

(OEIS is Neil Sloane's On-Line Encyclopedia of Integer Sequences at https://oeis.org/)
"""

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright (c) 2015 Philippe Guglielmetti"
__license__ = "LGPL"
__credits__ = []

import os
import sys
import logging
import re

from goulib import itertools2, decorators
from goulib.tests import *

from oeis import *
from database import database

path = os.path.dirname(os.path.abspath(__file__))

slow = []  # list of slow sequences


def assert_generator(f, ref, name, timeout=10):
    ref = list(ref)
    n = len(ref)
    timeout, f.timeout = f.timeout, timeout  # save Sequence's timeout
    l = []
    try:
        for i, x in enumerate(f):
            l.append(x)
            if i > n:
                break
    except decorators.TimeoutError:
        if i < min(10, n / 2):
            slow.append((i, name, f.desc))
            logging.warning('%s timeout after only %d loops' % (name, i))
        elif _DEBUG:
            logging.info('%s timeout after %d loops' % (name, i))
    finally:
        f.timeout = timeout  # restore Sequence's timeout


def data(s):
    s2 = s[1:]

    # http://stackoverflow.com/questions/807863/how-to-output-list-of-floats-to-a-binary-file-in-python
    try:
        return database[s]
    except IndexError:
        pass

    from urllib.request import urlopen
    try:  # is there a local, patched file ?
        file = open('b%s.txt' % s2, 'r')
        logging.warning('reading b%s.txt' % s2)
    # FileNotFoundError (not defined in Py2.7) download the B-file from OEIS
    except OSError:
        file = urlopen('http://oeis.org/A%s/b%s.txt' % (s2, s2))
        logging.info('downloading b%s.txt' % s2)
    res = []
    for line in file:  # files are iterable
        c = chr(line[0])
        if c == '#':
            continue  # skip comment
        m = re.search(b'(\d+)\s+(-?\d+)', line)
        if m:
            m = m.groups()
            if len(m) == 2:
                n = int(m[1])
                res.append(n)

    database[s] = res

    return res


def check(f, s=None):
    try:
        name = f.__name__
    except:
        name, f = f, globals()[f]
    # to break as soon as an error happens
    assert_generator(f, s or name, name=name, timeout=None)


class TestOEIS:

    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        global slow
        print('slower Sequences:')
        slow.sort()
        for (i, name, desc) in slow:
            print(i, name, desc)

    # http://stackoverflow.com/questions/32899/how-to-generate-dynamic-parametrized-unit-tests-in-python
    # http://nose.readthedocs.org/en/latest/writing_tests.html
    # this is ABSOLUTELY FANTASTIC !
    def test_generator(self):
        for name in sorted(oeis.keys()):  # test in sorted order to spot slows faster
            logging.info(name)
            time_limit = 0.1  # second
            from functools import partial
            f = partial(assert_generator, oeis[name], data(
                name), name, time_limit)
            f.description = str(oeis[name]) + '\n'
            yield (f,)


_DEBUG = True
if __name__ == "__main__":
    if _DEBUG:
        runmodule(logging.DEBUG, argv=['-x', '-v'])
    else:
        runmodule()
