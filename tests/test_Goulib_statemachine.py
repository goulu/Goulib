#!/usr/bin/env python
# coding: utf8
from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *
from Goulib.StateMachine import *

class SM_Test(StateMachine):
    def state000(self):
        """titre
           actions
        """
        print('000')
    
    def state001(self):
        """titre
           actions
           --> 002: condition :to move
        """ 


class TestSegment:
    
    def test___init__(self):
        sm = SM_Test()
        print(sm.states)
    
    def test__html_repr_(self):
        sm = SM_Test()
        assert_equal(sm._repr_html_(), '<table border="1"><caption>SM_Test</caption><tr><td>0</td><td>titre</td><td>actions</td></tr><tr><td>1</td><td>titre</td><td>actions</td></tr></table>')
        

        
 
if __name__ == "__main__":
    runmodule()           