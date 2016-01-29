#!/usr/bin/env python
# coding: utf8

from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *

from Goulib.markup import *

class TestCgiprint:
    def test_cgiprint(self):
        # assert_equal(expected, cgiprint(inline, unbuff, line_end))
        raise SkipTest 

class TestTag:
    def test_tag(self):
        t=tag('tag', u'bétweêñ', class_='class')
        assert_equal(t,'<tag class="class">b&#233;twe&#234;&#241;</tag>')
        
        t=tag('tag', None, style={'align':'left', 'color':'red'}, single=True)
        assert_equal(t,'<tag style="color:red; align:left;" />')
        

class TestElement:
    def test___call__(self):
        # element = element(tag, case, parent)
        # assert_equal(expected, element.__call__(*args, **kwargs))
        raise SkipTest 

    def test___init__(self):
        # element = element(tag, case, parent)
        raise SkipTest 

    def test_close(self):
        # element = element(tag, case, parent)
        # assert_equal(expected, element.close())
        raise SkipTest 

    def test_open(self):
        # element = element(tag, case, parent)
        # assert_equal(expected, element.open(**kwargs))
        raise SkipTest 

    def test_render(self):
        # element = element(tag, case, parent)
        # assert_equal(expected, element.render(t, single, between, kwargs))
        raise SkipTest 

class TestPage:
    @classmethod
    def setup_class(self):
        self.page=page()
        pass
        
    def test___init__(self):
        pass
        
    def test___call__(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.__call__(escape))
        raise SkipTest 

    def test___getattr__(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.__getattr__(attr))
        raise SkipTest 


    def test___str__(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.__str__())
        raise SkipTest 

    def test_add(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.add(text))
        raise SkipTest 

    def test_addcontent(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.addcontent(text))
        raise SkipTest 

    def test_addfooter(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.addfooter(text))
        raise SkipTest 

    def test_addheader(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.addheader(text))
        raise SkipTest 

    def test_css(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.css(filelist))
        raise SkipTest 

    def test_init(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.init(lang, css, metainfo, title, header, footer, charset, encoding, doctype, bodyattrs, script, base))
        raise SkipTest 

    def test_metainfo(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.metainfo(mydict))
        raise SkipTest 

    def test_scripts(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.scripts(mydict))
        raise SkipTest 

class test__oneliner:
    def test___getattr__(self):
        # _oneliner = _oneliner(case)
        # assert_equal(expected, _oneliner.__getattr__(attr))
        raise SkipTest 

    def test___init__(self):
        # _oneliner = _oneliner(case)
        raise SkipTest 

class TestEscape:
    def test_escape(self):
        # assert_equal(expected, escape(text, newline))
        raise SkipTest 

class TestUnescape:
    def test_unescape(self):
        # assert_equal(expected, unescape(text))
        raise SkipTest 

class TestRussell:
    def test___contains__(self):
        # russell = russell()
        # assert_equal(expected, russell.__contains__(item))
        raise SkipTest 

class TestClosingError:
    def test___init__(self):
        # closing_error = ClosingError(tag)
        raise SkipTest 

class TestOpeningError:
    def test___init__(self):
        # opening_error = OpeningError(tag)
        raise SkipTest 

class TestArgumentError:
    def test___init__(self):
        # argument_error = ArgumentError(tag)
        raise SkipTest 

class TestInvalidElementError:
    def test___init__(self):
        # invalid_element_error = InvalidElementError(tag, mode)
        raise SkipTest 

class TestDeprecationError:
    def test___init__(self):
        # deprecation_error = DeprecationError(tag)
        raise SkipTest 

class TestModeError:
    def test___init__(self):
        # mode_error = ModeError(mode)
        raise SkipTest 

class TestCustomizationError:
    def test___init__(self):
        # customization_error = CustomizationError()
        raise SkipTest 

class TestStyleDict2str:
    def test_style_dict2str(self):
        # assert_equal(expected, style_dict2str(style))
        raise SkipTest 

class TestStyleStr2dict:
    def test_style_str2dict(self):
        # assert_equal(expected, style_str2dict(style))
        raise SkipTest 

if __name__=="__main__":
    runmodule()