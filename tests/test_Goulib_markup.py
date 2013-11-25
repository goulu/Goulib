from nose.tools import assert_equal
from nose import SkipTest

class TestCgiprint:
    def test_cgiprint(self):
        # assert_equal(expected, cgiprint(inline, unbuff, line_end))
        raise SkipTest # TODO: implement your test here

class TestElement:
    def test___call__(self):
        # element = element(tag, case, parent)
        # assert_equal(expected, element.__call__(*args, **kwargs))
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # element = element(tag, case, parent)
        raise SkipTest # TODO: implement your test here

    def test_close(self):
        # element = element(tag, case, parent)
        # assert_equal(expected, element.close())
        raise SkipTest # TODO: implement your test here

    def test_open(self):
        # element = element(tag, case, parent)
        # assert_equal(expected, element.open(**kwargs))
        raise SkipTest # TODO: implement your test here

    def test_render(self):
        # element = element(tag, case, parent)
        # assert_equal(expected, element.render(tag, single, between, kwargs))
        raise SkipTest # TODO: implement your test here

class TestPage:
    def test___call__(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.__call__(escape))
        raise SkipTest # TODO: implement your test here

    def test___getattr__(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.__getattr__(attr))
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        raise SkipTest # TODO: implement your test here

    def test___str__(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.__str__())
        raise SkipTest # TODO: implement your test here

    def test_add(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.add(text))
        raise SkipTest # TODO: implement your test here

    def test_addcontent(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.addcontent(text))
        raise SkipTest # TODO: implement your test here

    def test_addfooter(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.addfooter(text))
        raise SkipTest # TODO: implement your test here

    def test_addheader(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.addheader(text))
        raise SkipTest # TODO: implement your test here

    def test_css(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.css(filelist))
        raise SkipTest # TODO: implement your test here

    def test_init(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.init(lang, css, metainfo, title, header, footer, charset, encoding, doctype, bodyattrs, script, base))
        raise SkipTest # TODO: implement your test here

    def test_metainfo(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.metainfo(mydict))
        raise SkipTest # TODO: implement your test here

    def test_scripts(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.scripts(mydict))
        raise SkipTest # TODO: implement your test here

class test__oneliner:
    def test___getattr__(self):
        # _oneliner = _oneliner(case)
        # assert_equal(expected, _oneliner.__getattr__(attr))
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # _oneliner = _oneliner(case)
        raise SkipTest # TODO: implement your test here

class TestEscape:
    def test_escape(self):
        # assert_equal(expected, escape(text, newline))
        raise SkipTest # TODO: implement your test here

class TestUnescape:
    def test_unescape(self):
        # assert_equal(expected, unescape(text))
        raise SkipTest # TODO: implement your test here

class TestRussell:
    def test___contains__(self):
        # russell = russell()
        # assert_equal(expected, russell.__contains__(item))
        raise SkipTest # TODO: implement your test here

class TestClosingError:
    def test___init__(self):
        # closing_error = ClosingError(tag)
        raise SkipTest # TODO: implement your test here

class TestOpeningError:
    def test___init__(self):
        # opening_error = OpeningError(tag)
        raise SkipTest # TODO: implement your test here

class TestArgumentError:
    def test___init__(self):
        # argument_error = ArgumentError(tag)
        raise SkipTest # TODO: implement your test here

class TestInvalidElementError:
    def test___init__(self):
        # invalid_element_error = InvalidElementError(tag, mode)
        raise SkipTest # TODO: implement your test here

class TestDeprecationError:
    def test___init__(self):
        # deprecation_error = DeprecationError(tag)
        raise SkipTest # TODO: implement your test here

class TestModeError:
    def test___init__(self):
        # mode_error = ModeError(mode)
        raise SkipTest # TODO: implement your test here

class TestCustomizationError:
    def test___init__(self):
        # customization_error = CustomizationError()
        raise SkipTest # TODO: implement your test here


class test__oneliner:
    def test___getattr__(self):
        # _oneliner = _oneliner(case)
        # assert_equal(expected, _oneliner.__getattr__(attr))
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # _oneliner = _oneliner(case)
        raise SkipTest # TODO: implement your test here

