#!/usr/bin/env python
# coding: utf8

from __future__ import division, print_function #"true division" everywhere

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright (c) 2015 Philippe Guglielmetti"
__license__ = "LGPL"
__credits__ = ["https://oeis.org/"]

__docformat__ = 'restructuredtext'
__version__ = '$Id$'
__revision__ = '$Revision$'

import os, sys, six, logging, re
from six.moves import zip

from Goulib import itertools2, decorators
from Goulib.tests import *

from examples.oeis import *

path=os.path.dirname(os.path.abspath(__file__))

slow=[] #list of slow sequences

def assert_generator(f,ref,name,timeout=10):
    n=len(ref)
    timeout,f.timeout=f.timeout,timeout #save Sequence's timeout
    l=[]
    try:
        for i,x in enumerate(f) : 
            l.append(x)
            if i>n : break
    except decorators.TimeoutError:
        if i<min(10,n/2):
            slow.append((i,name,f.desc))
            logging.warning('%s timeout after only %d loops'%(name,i))
        elif _DEBUG:
            logging.info('%s timeout after %d loops'%(name,i))
    finally:
        f.timeout=timeout #restore Sequence's timeout

import pickle
cachef=path+'/oeis.pck'
database = dict()

def data(s):
    s2=s[1:]

    #http://stackoverflow.com/questions/807863/how-to-output-list-of-floats-to-a-binary-file-in-python
    try:
        return database[s]
    except:
        pass

    from six.moves.urllib.request import urlopen
    try: # is there a local, patched file ?
        file = open('b%s.txt'%s2,'rb')
        logging.warning('reading b%s.txt'%s2)
    except OSError: # FileNotFoundError (not defined in Py2.7) download the B-file from OEIS
        file = urlopen('http://oeis.org/A%s/b%s.txt'%(s2,s2))
        logging.info('downloading b%s.txt'%s2)
    res=[]
    for line in file: # files are iterable
        c=six.unichr(line[0])
        if c=='#' :
            continue # skip comment
        m=re.search(b'(\d+)\s+(-?\d+)',line)
        if m:
            m=m.groups()
            if len(m)==2:
                n=int(m[1])
                res.append(n)

    database[s]=res
    pickle.dump(tuple((s,res)), open(cachef,"ab"), protocol=2) #append to pickle file

    return res

#data('A004042') #to force creating the database from scratch
with open(cachef, "rb") as f:
    while True:
        try:
            k,v=pickle.load(f)
            database[k]=v
        except EOFError:
            break

def check(f,s=None):
    try:
        name=f.__name__
    except:
        name,f=f,globals()[f]
    assert_generator(f,s or name,name=name,time_limit=None) #to break as soon as an error happens

class TestOEIS:

    @classmethod
    def setup_class(cls):
        pass
    
    @classmethod
    def teardown_class(cls):
        global slow
        print('slower Sequences:')
        slow.sort()
        for (i,name,desc) in slow:
            print(i,name,desc)

    #http://stackoverflow.com/questions/32899/how-to-generate-dynamic-parametrized-unit-tests-in-python
    #http://nose.readthedocs.org/en/latest/writing_tests.html
    #this is ABSOLUTELY FANTASTIC !
    def test_generator(self):
        for name in sorted(oeis.keys()): # test in sorted order to spot slows faster
            logging.info(name)
            time_limit=0.1 #second

            yield assert_generator,oeis[name],data(name),name, time_limit

_DEBUG=True
if __name__ == "__main__":
    if _DEBUG:
        runmodule(logging.DEBUG,argv=['-x'])
    else:
        runmodule()

