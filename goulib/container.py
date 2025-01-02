"""
advanced containers : Record (struct) 
INFINITE Sequence has been moved to OEIS project
"""
__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2015, Philippe Guglielmetti"
__credits__ = []
__license__ = "LGPL"

from bisect import bisect_left, bisect_right
from collections import OrderedDict

import operator

from itertools import count, tee, islice, chain
from goulib import itertools2, decorators, tests


class Record(OrderedDict):
    """mimics a Pascal record or a C struct"""

    # https://stackoverflow.com/a/5491708/1395973
    def __init__(self, *args, **kwargs):
        super(Record, self).__init__(*args, **kwargs)
        self._initialized = True

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if not name:
            return
        if '_initialized' in self.__dict__:
            super(Record, self).__setitem__(name, value)
        else:
            super(Record, self).__setattr__(name, value)

    def __str__(self):
        res = ['%s:%s' % (k, self[k]) for k in self]
        return '{{%s}}' % (','.join(res))