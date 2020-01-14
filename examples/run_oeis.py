__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright (c) 2015 Philippe Guglielmetti"
__license__ = "LGPL"
__credits__ = ["https://oeis.org/"]

__docformat__ = 'restructuredtext'
__version__ = '$Id$'
__revision__ = '$Revision$'

import sqlite3
import os
import sys
import logging
import re

from Goulib import itertools2, decorators
from Goulib.tests import *

from examples.oeis import *

path = os.path.dirname(os.path.abspath(__file__))

slow = []  # list of slow sequences


class Database:

    def __init__(self, dbpath):
        dbexists = os.path.exists(dbpath)
        try:
            self.db = sqlite3.connect(dbpath)
            if not dbexists:
                self.execute(open('create.sql', 'r').read())
                logging.info("tables created")
        except sqlite3.Error as error:
            logging.error("Error while connecting to sqlite", error)
            
    def execute(self, sql):
        cursor = self.db.cursor()
        cursor.execute(sql)
        res = cursor.fetchall() 
        cursor.close();
        return res
    
    @property        
    def version(self):
        return str(self.execute("select sqlite_version();")[0][0])
    
    def __setitem__(self, key, value):
        key = int(key[1:])
        cursor = self.db.cursor()
        req = "INSERT OR REPLACE INTO seq (id, i, n) VALUES"
        for i, n in enumerate(value):
            s = req + str((key, i, n))
            cursor.execute(s)
        cursor.close();
        self.db.commit();
    
    def __getitem__(self, key):
        key = int(key[1:])
        s = "SELECT n FROM seq WHERE id=%d ORDER BY i" % key
        res = self.execute(s)
        if not res : 
            raise IndexError
        return res
    
    def find(self, n, id=None, i=None):
        s = "SELECT id,i FROM seq WHERE n=%d" % n
        if id is not None:
            s = s + " AND id=%d" % id
        if i is not None:
            s = s + " AND i=%d" % i
        return self.execute(s)   
    
    def search(self, l):
        res = None
        p = [(None, None)]  # array of possible (id,i) : start with all
        for n in l:
            r = []
            for (id, i) in p:
                if i is not None: i=i+1
                r.extend(self.find(n, id, i))
            p = r
        return p

        
database = Database(path + '/oeis.db')
logging.info("SQLite DB Version : " + database.version)

database.search([3, 5, 8])
        
        
def assert_generator(f, ref, name, timeout=10):
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
