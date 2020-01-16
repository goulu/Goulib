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

from Goulib import itertools2


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
        req = "INSERT OR REPLACE INTO seq (id, i, n, repeat) VALUES"
        i = 1
        for n, r in itertools2.compress(value):
            s = req + str((key, i, n, r))
            cursor.execute(s)
            i = i + r
        cursor.close();
        self.db.commit();
    
    def __getitem__(self, key):
        key = int(key[1:])
        s = "SELECT n,repeat FROM seq WHERE id=%d ORDER BY i" % key
        res = self.execute(s)
        if not res : 
            raise IndexError
        return itertools2.decompress(res)
    
    def search(self, value):
        #efficient search first requires to dynamically build a tricky request 
        for j, (n, r) in enumerate(itertools2.compress(value)):
            if j == 0:
                s1 = "WITH q0 AS (SELECT * FROM seq WHERE n = %d AND repeat >= %d)," % (n, r)
                s2 = "SELECT q0.id, q0.i FROM q0"
            else: 
                s1 = s1 + "\nq%d AS (SELECT * FROM seq WHERE n = %d AND repeat = %d)," % (j, n, r)
                s2 = s2 + '\nJOIN q%d ON q%d.id=q0.id AND q%d.i=q%d.i+q%d.repeat' % tuple([j] * 3 + [j - 1] * 2)
        s = s1[:-1] + '\n\n' + s2
        res=self.execute(s)
        # keep first result for each sequence only
        for (id,i) in itertools2.unique(res, key=lambda x:x[0], buffer=None):
            key='000000'+str(id)
            yield 'A'+key[-6:],i

        
path = os.path.dirname(os.path.abspath(__file__))
database = Database(path + '/oeis.db')
logging.info("SQLite DB Version : " + database.version)

from Goulib.tests import *


class TestDatabase:

    def test_search(self):
        res = list(database.search([3, 5, 8, 13]))
        assert_true(('A000045', 5) in res)
        res = list(database.search([0, 0, 0]))
        res = list(database.search([1, 0, 0, 0, 1]))
        assert_true(('A000796', 14202) in res)
        assert_true(('A008683', 1322) in res)


_DEBUG = True
if __name__ == "__main__":
    if _DEBUG:
        runmodule(logging.DEBUG, argv=['-x', '-v'])
    else:
        runmodule()
