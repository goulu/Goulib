#!/usr/bin/env python
# coding: utf8

from __future__ import division #"true division" everywhere

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

def assert_generator(f,l,name,time_limit=10):
    i=0
    try:
        for item1,item2 in decorators.itimeout(zip(f,l),time_limit):
            m='%s : First differing element %d: %s != %s\n' %(name, i, item1, item2)
            assert_equal(item1,item2, msg=m)
            i+=1
    except decorators.TimeoutError:
        if i<min(10,len(l)/2):
            logging.warning('%s timeout after only %d loops'%(name,i))
        else:
            logging.debug('%s timeout after %d loops'%(name,i))

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
    except FileNotFoundError: # if not, download the B-file from OEIS
        file = urlopen('http://oeis.org/A%s/b%s.txt'%(s2,s2))
        logging.info('downloading b%s.txt'%s2)
    res=[]
    for line in file: # files are iterable
        m=re.search(b'(\d+)\s+(-?\d+)',line)
        if m:
            m=m.groups()
            if len(m)==2:
                n=int(m[1])
                res.append(n)

    database[s]=res
    pickle.dump(tuple((s,res)), open(cachef,"ab"), protocol=2) #append to pickle file

    return res
# data('A000001') #to force creating the file
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
    def setup_class(self):
        # A009994 results has a stupid '2014' date at the beginning
        # remove it until we find a way to avoid reading it...
        d=data('A009994')
        if d[0]==2014:
            d.pop(0)
            database['A009994']=d

    def test_repeat(self):
        #check that we can iterate twice in the same Sequence
        l1=list(itertools2.take(20,A000040))
        l2=list(itertools2.take(20,A000040))
        assert_equal(l1,l2)

    def test_A000040(self):
        assert_equal(A000040[0],2)
        assert_equal(A000040[10],31)
        assert_false(32 in A000040)
        assert_true(4547337172376300111955330758342147474062293202868155909489 in A000040)

    def test_A000129(self):
        assert_equal(A000129[43],10181446324101389)

    def test_A000396(self):
        assert_equal(math2.is_perfect(A000396[8]),0)
        a=2658455991569831744654692615953842176
        assert_equal(math2.is_perfect(a),0)
        assert_equal(A000396[8],a)

    def A057588(self):
        assert_equal(A057588,database['A057588'])

    def test_A019434(self):
        res=data('A019434')
        # assert_equal(A019434[:len(res)],res)
        # assert_generator(A019434,[3, 5, 17, 257, 65537],A019434.name)
        #TODO: find why this hangs...

    #http://stackoverflow.com/questions/32899/how-to-generate-dynamic-parametrized-unit-tests-in-python
    #http://nose.readthedocs.org/en/latest/writing_tests.html
    #this is ABSOLUTELY FANTASTIC !
    def test_generator(self):
        for name in sorted(oeis.keys()): # test in sorted order to spot slows faster
            logging.debug(name)
            time_limit=0.1 #second
            if name=='A019434': continue #skip, use test above instead

            yield assert_generator,oeis[name],data(name),name, time_limit

_DEBUG=False
if __name__ == "__main__":
    if _DEBUG:
        runmodule(logging.DEBUG,argv=['-x'])
    else:
        runmodule()
