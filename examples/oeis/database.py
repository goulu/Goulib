"""
database for OEIS sequences and tests
"""

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright (c) 2015 Philippe Guglielmetti"
__license__ = "LGPL"
__credits__ = ["https://oeis.org/"]

import sqlite3
import os
import sys
import logging

from goulib import itertools2
from goulib.tests import *


class Database:

    def __init__(self, path):
        dbpath = os.path.join(path, 'oeis.db')
        dbexists = os.path.exists(dbpath)
        try:
            self.db = sqlite3.connect(dbpath)
            if not dbexists:
                self.execute(
                    open(os.path.join(path, 'create.sql'), 'r').read())
                logging.info("tables created")
        except sqlite3.Error as error:
            logging.error("Error while connecting to sqlite", error)

    def execute(self, sql):
        cursor = self.db.cursor()
        cursor.execute(sql)
        res = cursor.fetchall()
        cursor.close()
        return res

    @property
    def version(self):
        return str(self.execute("select sqlite_version();")[0][0])

    def lenghtOf(self, key):
        ''' return the length of the sequence if it is stored, 0 if it is not'''
        s = "SELECT MAX(i),repeat FROM seq WHERE id=%d" % int(key[1:])
        imax = self.execute(s)[0]
        if imax == (None, None):
            return 0
        else:
            return sum(imax) - 1

    def __contains__(self, key):
        return self.lenghtOf(key) > 0

    def store(self, key, value, timeout=1):
        """
        :param key: string sequence id
        :param value: iterable. if finite
        :param timeout: float , max time in seconds to store digits in db
        """
        n = self.lenghtOf(key)
        if n > 0:
            try:
                if n >= len(value):  # already stored, even more ...
                    return n
            except TypeError:
                pass  # value is a generator, we'll store as much as possible

        from goulib.decorators import itimeout
        from multiprocessing import TimeoutError

        cursor = self.db.cursor()
        req = "INSERT OR REPLACE INTO seq (id, i, n, repeat) VALUES"
        i = 1
        try:
            for n, r in itimeout(itertools2.compress(value), timeout):
                s = req + str((key[1:], i, n, r))
                cursor.execute(s)
                i = i + r
        except TimeoutError:
            pass
        finally:
            cursor.close()
            self.db.commit()
            return i

    def __setitem__(self, key, value):
        return self.store(key, value)

    def __getitem__(self, key):
        key = int(key[1:])
        s = "SELECT n,repeat FROM seq WHERE id=%d ORDER BY i" % key
        res = self.execute(s)
        if not res:
            raise IndexError
        return itertools2.decompress(res)

    def search(self, value):
        # efficient search first requires to dynamically build a tricky request
        for j, (n, r) in enumerate(itertools2.compress(value)):
            if j == 0:
                s1 = "WITH q0 AS (SELECT * FROM seq WHERE n = %d AND repeat >= %d)," % (n, r)
                s2 = "SELECT q0.id, q0.i FROM q0"
            else:
                s1 = s1 + \
                    "\nq%d AS (SELECT * FROM seq WHERE n = %d AND repeat = %d)," % (j, n, r)
                s2 = s2 + \
                    '\nJOIN q%d ON q%d.id=q0.id AND q%d.i=q%d.i+q%d.repeat' % tuple([j] * 3 + [
                                                                                    j - 1] * 2)
        s = s1[:-1] + '\n\n' + s2
        res = self.execute(s)
        # keep first result for each sequence only
        for (id, i) in itertools2.unique(res, key=lambda x: x[0], buffer=None):
            key = '000000' + str(id)
            yield 'A' + key[-6:], i

    def populate(self, file='http://oeis.org/stripped.gz'):
        import gzip
        if file[:5] == 'http:':
            from urllib.request import urlopen
            file = urlopen(file)

        file = gzip.open(file)

        for line in file:
            l = line.decode("utf-8")
            if l[0] != 'A':
                logging.info(line)
                continue
            l = l.split(',')
            id = l[0]
            value = list(map(int, l[1:-1]))  # skip the last \n
            logging.debug('%s %d' % (id, len(value)))
            self[id] = value


logging.basicConfig(level=logging.DEBUG)
path = os.path.dirname(os.path.abspath(__file__))
database = Database(path)
logging.info("SQLite DB Version : " + database.version)


# from goulib.tests import *


class TestDatabase:

    def setUp(self):
        if 'A000045' not in database:
            import oeis
            database['A000045'] = oeis.A000045

    def test_search(self):
        res = database.search([3, 5, 8, 13])
        assert_true(('A000045', 5) in res)


_DEBUG = True
if __name__ == "__main__":
    database.populate()
    if _DEBUG:
        runmodule(logging.DEBUG, argv=['-x', '-v'])
    else:
        runmodule()
