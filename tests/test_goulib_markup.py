from goulib.tests import *
from goulib.markup import *


class TestCgiprint:
    def test_cgiprint(self):
        # assert_equal(expected, cgiprint(inline, unbuff, line_end))
        pass  # TODO: implement


class TestTag:
    def test_tag(self):
        t = tag('tag', 'between', class_='class')
        assert t == '<tag class="class">between</tag>'  # Py 3

        t = tag('tag', u'b\xc3\xa9twe\xc3\xaa\xc3\xb1', class_='class')
        assert t == '<tag class="class">b&#195;&#169;twe&#195;&#170;&#195;&#177;</tag>'

        t = tag('tag', None, style={
                'align': 'left', 'color': 'red'}, single=True)
        assert t in (
            '<tag style="color:red; align:left;" />',
            '<tag style="align:left; color:red;" />',
        )

        t = tag('test', r'$\left(x\right)$')
        assert t == r'<test>$\left(x\right)$</test>'


class TestElement:
    def test___call__(self):
        # element = element(tag, case, parent)
        # assert_equal(expected, element.__call__(*args, **kwargs))
        pass  # TODO: implement

    def test___init__(self):
        # element = element(tag, case, parent)
        pass  # TODO: implement

    def test_close(self):
        # element = element(tag, case, parent)
        # assert_equal(expected, element.close())
        pass  # TODO: implement

    def test_open(self):
        # element = element(tag, case, parent)
        # assert_equal(expected, element.open(**kwargs))
        pass  # TODO: implement

    def test_render(self):
        # element = element(tag, case, parent)
        # assert_equal(expected, element.render(t, single, between, kwargs))
        pass  # TODO: implement


class TestPage:
    @classmethod
    def setup_class(self):
        self.page = page()
        pass

    def test___init__(self):
        pass

    def test___call__(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.__call__(escape))
        pass  # TODO: implement

    def test___getattr__(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.__getattr__(attr))
        pass  # TODO: implement

    def test___str__(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.__str__())
        pass  # TODO: implement

    def test_add(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.add(text))
        pass  # TODO: implement

    def test_addcontent(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.addcontent(text))
        pass  # TODO: implement

    def test_addfooter(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.addfooter(text))
        pass  # TODO: implement

    def test_addheader(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.addheader(text))
        pass  # TODO: implement

    def test_css(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.css(filelist))
        pass  # TODO: implement

    def test_init(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.init(lang, css, metainfo, title, header, footer, charset, encoding, doctype, bodyattrs, script, base))
        pass  # TODO: implement

    def test_metainfo(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.metainfo(mydict))
        pass  # TODO: implement

    def test_scripts(self):
        # page = page(mode, case, onetags, twotags, separator, class_)
        # assert_equal(expected, page.scripts(mydict))
        pass  # TODO: implement


class TestEscape:
    def test_escape(self):
        # assert_equal(expected, escape(text, newline))
        pass  # TODO: implement


class TestUnescape:
    def test_unescape(self):
        # assert_equal(expected, unescape(text))
        pass  # TODO: implement


class TestRussell:
    def test___contains__(self):
        # russell = russell()
        # assert_equal(expected, russell.__contains__(item))
        pass  # TODO: implement


class TestClosingError:
    def test___init__(self):
        # closing_error = ClosingError(tag)
        pass  # TODO: implement


class TestOpeningError:
    def test___init__(self):
        # opening_error = OpeningError(tag)
        pass  # TODO: implement


class TestArgumentError:
    def test___init__(self):
        # argument_error = ArgumentError(tag)
        pass  # TODO: implement


class TestInvalidElementError:
    def test___init__(self):
        # invalid_element_error = InvalidElementError(tag, mode)
        pass  # TODO: implement


class TestDeprecationError:
    def test___init__(self):
        # deprecation_error = DeprecationError(tag)
        pass  # TODO: implement


class TestModeError:
    def test___init__(self):
        # mode_error = ModeError(mode)
        pass  # TODO: implement


class TestCustomizationError:
    def test___init__(self):
        # customization_error = CustomizationError()
        pass  # TODO: implement


class TestStyleDict2str:
    def test_style_dict2str(self):
        # assert_equal(expected, style_dict2str(style))
        pass  # TODO: implement


class TestStyleStr2dict:
    def test_style_str2dict(self):
        # assert_equal(expected, style_str2dict(style))
        pass  # TODO: implement


class test__oneliner:
    def test___getattr__(self):
        # _oneliner = _oneliner(case)
        # assert_equal(expected, _oneliner.__getattr__(attr))
        pass  # TODO: implement  # implement your test here

    def test___init__(self):
        # _oneliner = _oneliner(case)
        pass  # TODO: implement  # implement your test here
