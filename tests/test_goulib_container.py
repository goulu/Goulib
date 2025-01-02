from goulib.tests import *      # pylint: disable=wildcard-import, unused-wildcard-import
from goulib.container import *  # pylint: disable=wildcard-import, unused-wildcard-import

class TestRecord:
    @classmethod
    def setup_class(cls):
        cls.d = {'first': 'Albert', 'last': 'Einstein'}
        cls.r = Record(cls.d)
        # assert_raises(AttributeError, lambda: cls.r.birth)

    def test___init__(self):
        pass

    def test___getattr__(self):
        assert self.r.first == self.d['first']
        assert self.r['first'] == self.d['first']

    def test___setattr__(self):
        self.r.born = 1879
        assert self.r['born'] == 1879

        r2 = self.r.copy()
        r2.first = 'Franck'  # Franck Enstein ;-)
        # check the fields are copied
        assert self.r.first == self.d['first']

    def test___str__(self):
        # record = Record(*args, **kwargs)
        # assert_equal(expected, record.__str__())
        pytest.skip("not yet implemented")  # TODO: implement



