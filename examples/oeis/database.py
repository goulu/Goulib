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
                if i is not None: i = i + 1
                r.extend(self.find(n, id, i))
            p = r
        return p

        
path = os.path.dirname(os.path.abspath(__file__))
database = Database(path + '/oeis.db')
logging.info("SQLite DB Version : " + database.version)

from Goulib.tests import *

class TestDatabase:

    def test_search(self):
        database.search([3,5,8,13])
        database.search([0,0,0])
        database.search([1,0,0,0,1])


_DEBUG = True
if __name__ == "__main__":
    if _DEBUG:
        runmodule(logging.DEBUG, argv=['-x', '-v'])
    else:
        runmodule()
