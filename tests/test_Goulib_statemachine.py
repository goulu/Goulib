#!/usr/bin/env python
# coding: utf8
from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *
from Goulib.statemachine import *

class SM_Test(StateMachine):
    def state000(self):
        """titre
           actions
           --> 001: time out
        """
        logging.debug('000')
        self.time += V(3,'s')
    
    def state001(self):
        """titre
           actions
           --> 000: condition :to move
        """ 
        logging.debug('001')
        self.time += V(7,'s')


class TestSM_test:
    
    def test___init__(self):
        sm = SM_Test()
        logging.debug(sm.states)
    
    def test__html_repr_(self):
        sm = SM_Test()
        assert_equal(sm._repr_html_(), '<table border="1"><caption>SM_Test</caption><tr><td>0</td><td>titre</td><td>actions</td></tr><tr><td>1</td><td>titre</td><td>actions</td></tr></table>')
        

    def test_run(self):
        sm = SM_Test()
        sm.run(start=0,maxState=4)
        assert_equal(sm.log, [(0, 0), (1, 3), (0, 10), (1, 13)])
 
if __name__ == "__main__":
    runmodule()           