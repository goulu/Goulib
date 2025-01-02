'''
simple symbolic math expressions
'''

__author__ = "Philippe Guglielmetti, J.F. Sebastian, Geoff Reedy"
__copyright__ = "Copyright 2013, Philippe Guglielmetti"
__credits__ = [
    'http://stackoverflow.com/questions/2371436/evaluating-a-mathematical-expression-in-a-string',
    'http://stackoverflow.com/questions/3867028/converting-a-python-numeric-expression-to-latex',
]
__license__ = "LGPL"

import logging
import copy
import typing  # collections
import inspect
import re
import ast
import math
import operator as op
from sortedcollections import SortedDict

from goulib import plot  # sets matplotlib backend
from goulib import itertools2, math2

# http://stackoverflow.com/questions/2371436/evaluating-a-mathematical-expression-in-a-string
# https://github.com/erwanp/pytexit

# indexes in _operators, _ functions and _constants to use for corresponding symbols
_dialect_str = 2
_dialect_python = 3
_dialect_latex = 4


class Context:
    constants = {  # constants in this dict are recognized in output
        bool: {True: (None, None, 'True', 'True', 'True'), False: (None, None, 'False', 'False', 'False')},
        float: {},
        complex: {complex(0, 1): (None, None, 'i', 'i', 'i')}
    }
    variables = {}  # dict of variables to subsitute in expr

    functions = SortedDict()  # only functions listed in this dict can be used in Expr

    # supported _operators with precedence and text + LaTeX repr
    # precedence as in https://docs.python.org/reference/expressions.html#operator-precedence
    #

    # table of allowed operators
    # note we very slightly prefer + over - and * over / for simpler expression generation
    operators = {
        ast.Or: (op.or_, 300, ' or ', ' or ', ' \\vee '),
        ast.And: (op.and_, 400, ' and ', ' and ', ' \\wedge '),
        ast.Not: (op.not_, 500, 'not ', 'not ', '\\neg'),
        ast.Eq: (op.eq, 600, '=', ' == ', ' = '),
        ast.Gt: (op.gt, 600, ' > ', ' > ', ' \\gtr '),
        ast.GtE: (op.ge, 600, ' >= ', ' >= ', ' \\gec '),
        ast.Lt: (op.lt, 600, ' < ', ' < ', ' \\ltr '),
        ast.LtE: (op.le, 600, ' <= ', ' <= ', ' \\leq '),
        ast.BitXor: (op.xor, 800, ' xor ', ' xor ', ' xor '),
        ast.LShift: (op.lshift, 1000, ' << ', ' << ', ' \\ll '),
        ast.RShift: (op.rshift, 1000, ' >> ', ' >> ', ' \\gg '),
        ast.Add: (op.add, 1100, '+', '+', '+'),
        ast.Sub: (op.sub, 1101, '-', '-', '-'),
        ast.Mult: (op.mul, 1200, '*', '*', ' \\cdot '),
        ast.Div: (op.truediv, 1201, '/', '/', '\\frac{%s}{%s}'),
        ast.FloorDiv: (op.floordiv, 1201, '//', '//', '\\left\\lfloor\\frac{%s}{%s}\\right\\rfloor'),
        ast.Mod: (op.mod, 1200, ' mod ', '%', ' \\bmod '),
        ast.Invert: (op.not_, 1300, '~', '~', '\\sim '),
        ast.UAdd: (op.pos, 1150, '+', '+', '+'),
        ast.USub: (op.neg, 1150, '-', '-', '-'),
        # returns an integer when result is integer ...
        ast.Pow: (math2.pow, 1400, '^', '**', '^'),

        # precedence of other types below
        ast.Call: (None, 9000),
        ast.Name: (None, 9000),
        ast.Constant: (None, 9000),
        ast.Constant: (None, 9000),
    }

    def add_function(self, f, s=None, r=None, l=None):
        ''' add a function to those allowed in Expr.

        :param f: function
        :param s: string representation, should be formula-like
        :param r: repr representation, should be cut&pastable in a calculator, or in python ...
        :param l: LaTeX representation
        '''
        self.functions[f.__name__] = (f, 9999, s, r or s, l)
        return self.functions[f.__name__]

    def add_constant(self, c, name, s=None, r=None, l=None):
        ''' add a constant to those recognized in Expr.

        :param c: constant
        :param s: string representation, should be formula-like
        :param r: repr representation, should be cut&pastable in a calculator, or in python ...
        :param l: LaTeX representation
        '''
        self.constants[type(c)][c] = (
            None, None, s or name, r or name, l or '\\' + name)

    def add_module(self, module):
        for fname, f in module.__dict__.items():
            if fname[0] == '_':
                continue
            if isinstance(f, typing.Callable):
                self.add_function(f)
            elif math2.is_number(f):
                self.add_constant(f, fname)

    def eval(self, node):
        '''safe eval of ast node : only functions and _operators listed above can be used

        :param node: ast.AST to evaluate
        :param ctx: dict of varname : value to substitute in node
        :return: number or expression string
        '''
        if isinstance(node, ast.Constant):  # <number>
            return node.n
        elif isinstance(node, ast.Name):
            return self.variables.get(node.id, node.id)  # return value or var
        elif isinstance(node, ast.Attribute):
            return getattr(self.variables, [node.value.id], node.attr)
        elif isinstance(node, ast.Tuple):
            return tuple(self.eval(e) for e in node.elts)
        elif isinstance(node, ast.Call):
            params = [self.eval(arg) for arg in node.args]
            if node.func.id not in self.functions:
                raise NameError('%s function not allowed' % node.func.id)
            f = self.functions[node.func.id][0]
            res = f(*params)
            # try to correct small error
            return math2.int_or_float(res, 0, 1e-12)
        elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
            op = self.operators[type(node.op)]
            left = self.eval(node.left)
            right = self.eval(node.right)
            if math2.is_number(left) and math2.is_number(right):
                res = op[0](left, right)
                # no correction here !
                return res
            else:
                return "%s%s%s" % (left, op[_dialect_python], right)
        elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
            right = self.eval(node.operand)
            return self.operators[type(node.op)][0](right)
        elif isinstance(node, ast.Compare):
            left = self.eval(node.left)
            for op, right in zip(node.ops, node.comparators):
                # TODO: find what to do when multiple items in list
                return self.operators[type(op)][0](left, self.eval(right))
        elif isinstance(node, ast.NameConstant):
            return node.value
        else:
            logging.warning(ast.dump(node, False, False))
            return self.eval(node.body)  # last chance

    def __init__(self):
        self.add_module(math)
        self.add_function(abs, l='\\lvert{%s}\\rvert')
        self.add_function(math.fabs, l='\\lvert{%s}\\rvert')
        self.add_function(math.factorial, '%s!', 'fact', '%s!')
        self.add_function(math2.factorial2, '%s!', 'fact', '%s!!')
        self.add_function(math2.sqrt, l='\\sqrt{%s}')
        self.add_function(math.trunc, l='\\left\\lfloor{%s}\\right\\rfloor')
        self.add_function(math.floor, l='\\left\\lfloor{%s}\\right\\rfloor')
        self.add_function(math.ceil, l='\\left\\lceil{%s}\\right\\rceil')
        self.add_function(math.asin, l='\\arcsin')
        self.add_function(math.acos, l='\\arccos')
        self.add_function(math.atan, l='\\arctan')
        self.add_function(math.asinh, l='\\sinh^{-1}')
        self.add_function(math.acosh, l='\\cosh^{-1}')
        self.add_function(math.atanh, l='\\tanh^{-1}')
        self.add_function(math.log, l='\\ln')
        self.add_function(math.log1p, l='\\ln\\left(1-{%s}\\rvert)')
        self.add_function(math.log10, l='\\log_{10}')
        self.add_function(math.log2, l='\\log_2')
        self.add_function(math.gamma, l='\\Gamma')
        self.add_function(math.exp, l='e^{%s}')
        self.add_function(math.expm1, l='e^{%s}-1')
        self.add_function(math.lgamma, 'log(abs(gamma(%s)))',
                          l='\\ln\\lvert\\Gamma\\left({%s}\\rvert)\\right)')
        self.add_function(math.degrees, l='%s\\cdot\\frac{360}{2\\pi}')
        self.add_function(math.radians, l='%s\\cdot\\frac{2\\pi}{360}')


default_context = Context()


def get_function_source(f):
    '''returns cleaned code of a function or lambda
    currently only supports:
    - lambda x:formula_of_(x)
    - def anything(x): return formula_of_(x)
    '''
    f = inspect.getsource(f).rstrip('\n')  # TODO: merge lines more subtly
    g = re.search(r'lambda(.*):(.*)(\)|#)', f)
    if g:
        res = g.group(2).strip()  # remove leading+trailing spaces
        bra, ket = res.count('('), res.count(')')
        if bra == ket:
            return res
        else:  # closing parenthesis ?
            return res[:-(ket - bra)]
    else:
        g = re.search(r'def \w*\((.*)\):\s*return (.*)', f)
        if g is None:
            raise ValueError('not a valid function code %s' % f)
        res = g.group(2)
    return res


def plouffe(f, epsilon=1e-6):
    if f < 0:
        r = plouffe(-f)
        if isinstance(r, str):
            return '-' + r
        return f
    if f != 0 and math2.is_integer(1 / f, epsilon):
        f = '1/%d' % math2.rint(1 / f)
    elif math2.is_integer(f * f, epsilon):
        f = 'sqrt(%d)' % math2.rint(f * f)
    return f


class Expr(plot.Plot):
    '''
    Math expressions that can be evaluated like standard functions
    combined using standard operators
    and plotted in IPython/Jupyter notebooks
    '''

    def __init__(self, f, context=default_context):
        '''
        :param f: function or operator, Expr to copy construct, or formula string
        '''
        self.context = context

        if isinstance(f, Expr):  # copy constructor
            self.body = f.body
            return
        elif isinstance(f, ast.AST):
            self.body = f
            return
        elif inspect.isfunction(f):
            try:
                f = get_function_source(f)
            except ValueError:
                f = '%s(x)' % f.__name__
        elif isinstance(f, typing.Callable):  # builtin function
            f = '%s(x)' % f.__name__
        elif f in ('True', 'False'):
            f = bool(f == 'True')

        if type(f) is bool:
            self.body = ast.Constant(f)
            return

        if type(f) is float:  # try to beautify it
            if math2.is_integer(f):
                f = math2.rint(f)
            else:
                f = plouffe(f)

        if math2.is_number(f):  # store it with full precision
            self.body = ast.Constant(f)
            return

        # accept ^ as power operator rather than xor ...
        f = str(f).replace('^', '**')

        self.body = compile(f, 'Expr', 'eval', ast.PyCF_ONLY_AST).body

    @property
    def isNum(self):
        return isinstance(self.body, ast.Constant)

    @property
    def isconstant(self):
        ''':return: True if Expr evaluates to a constant number or bool'''
        res = self()
        if math2.is_number(res):
            return True
        if isinstance(res, bool):
            return True
        return False

    def __call__(self, x=None, **kwargs):
        '''evaluate the Expr at x OR compose self(x())'''
        if isinstance(x, Expr):  # composition
            return self.applx(x)
        if itertools2.isiterable(x):
            return [self(x) for x in x]  # return a displayable list
        if x is not None:
            kwargs['x'] = x
        kwargs['self'] = self  # allows to call methods such as in Stats
        self.context.variables = kwargs
        try:
            e = self.context.eval(self.body)
        except TypeError:  # some params remain symbolic
            return self
        except Exception as error:  # ZeroDivisionError, OverflowError
            return None
        if math2.is_number(e):
            return e
        return Expr(e, self.context)

    def __float__(self):
        return self()

    def __repr__(self):
        return TextVisitor(_dialect_python).visit(self.body)

    def __str__(self):
        return TextVisitor(_dialect_str).visit(self.body)

    def _repr_html_(self):
        '''default rich format is LaTeX'''
        return self._repr_latex_()

    def latex(self):
        ''':return: string LaTex formula'''
        return TextVisitor(_dialect_latex).visit(self.body)

    def _repr_latex_(self):
        return r'${%s}$' % self.latex()

    def points(self, xmin=-1, xmax=1, step=0.1):
        ''':return: x,y lists of float : points for a line plot'''
        if self.isconstant:
            return [xmin, xmax], [self(xmin), self(xmax)]
        x = list(itertools2.arange(xmin, xmax, step))
        y = self(x)
        return x, y

    def _plot(self, ax, x=None, y=None, **kwargs):
        if x is None:
            x, y = self.points()

        if y is None:
            y = self(x)

        # slightly shift the points to make superimposed curves more visible
        offset = kwargs.pop('offset', 0)

        points = list(zip(x, y))  # might contain (x,None) for undefined points

        # curves between defined points
        for xy in itertools2.isplit(points, lambda _: not math2.is_real(_[1])):
            x, y = [], []  # matplotlib doesn't support generators...
            for v in xy:
                x.append(v[0] + offset)
                y.append(v[1] + offset)
            ax.plot(x, y, **kwargs)
        return ax

    def apply(self, f, right=None):
        '''function composition self o f = f(self(x))'''

        if right is None:
            if isinstance(f, ast.unaryop):
                node = ast.UnaryOp(f, self.body)
            else:
                # if not isinstance(f,Expr): f=Expr(f) #not useful as applx does the reverse
                return f.applx(self)
        else:
            if not isinstance(right, Expr):
                right = Expr(right, self.context)
            node = ast.BinOp(self.body, f, right.body)
        return Expr(node, self.context)

    def applx(self, f, var='x'):
        '''function composition f o self = self(f(x))'''
        if isinstance(f, Expr):
            f = f.body

        class Subst(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id == var:
                    return f
                else:
                    return node

        node = copy.deepcopy(self.body)
        return Expr(Subst().visit(node), self.context)

    def __eq__(self, other):
        if math2.is_number(other):
            try:
                return self() == other
            except:
                return False
        if not isinstance(other, Expr):
            other = Expr(other, self.context)
        return str(self()) == str(other())

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if math2.is_number(other):
            try:
                return self() < other
            except:
                return False
        if not isinstance(other, Expr):
            other = Expr(other, self.context)
        return float(self()) < float(other())

    def __le__(self, other):
        return self < other or self == other

    def __ge__(self, other):
        return not self < other

    def __gt__(self, other):
        return self >= other and not self == other

    def __add__(self, right):
        return self.apply(ast.Add(), right)

    def __sub__(self, right):
        return self.apply(ast.Sub(), right)

    def __neg__(self):
        return self.apply(ast.USub())

    def __mul__(self, right):
        return self.apply(ast.Mult(), right)

    def __rmul__(self, right):
        return Expr(right, self.context) * self

    def __truediv__(self, right):
        return self.apply(ast.Div(), right)

    def __pow__(self, right):
        return self.apply(ast.Pow(), right)

    __div__ = __truediv__

    def __invert__(self):
        return self.apply(ast.Invert())

    def __and__(self, right):
        return self.apply(ast.And(), right)

    def __or__(self, right):
        return self.apply(ast.Or(), right)

    def __xor__(self, right):
        return self.apply(ast.BitXor(), right)

    def __lshift__(self, dx):
        return self.applx(ast.BinOp(ast.Name('x', None), ast.Add(), ast.Constant(dx)))

    def __rshift__(self, dx):
        return self.applx(ast.BinOp(ast.Name('x', None), ast.Sub(), ast.Constant(dx)))

    def complexity(self):
        ''' measures the complexity of Expr
        :return: int, sum of the precedence of used ops
        '''

        def _node_complexity(node):
            try:
                res = self._operators[type(node.op)][1]
            except AttributeError:
                try:
                    res = self._operators[type(node)][1]
                except AttributeError:
                    res = 1
            try:
                res += _node_complexity(node.operand)
            except AttributeError:
                pass
            try:
                res += _node_complexity(node.left)
            except AttributeError:
                pass
            try:
                res += _node_complexity(node.right)
            except AttributeError:
                pass
            return res

        return _node_complexity(self.body)


# http://stackoverflow.com/questions/3867028/converting-a-python-numeric-expression-to-latex

class TextVisitor(ast.NodeVisitor):

    def __init__(self, dialect, context=default_context):
        ''':param dialect: int index in _operators of symbols to use
        '''
        self.dialect = dialect
        self.context = context

    def prec(self, op):
        ''' calculate the precedence of op '''
        if isinstance(op, (ast.BinOp, ast.UnaryOp)):
            op = op.op
        if isinstance(op, ast.Constant) and math2.is_real(op.n) and op.n < 0:
            return self.context.operators[ast.USub][1]
        try:
            return self.context.operators[type(op)][1]
        except KeyError:
            return self.context.operators[type(op)][1]

    def _par(self, content):
        if self.dialect == _dialect_latex:
            return '\\left(%s\\right)' % content
        else:
            return '(%s)' % content

    def visit_Call(self, n):
        args = r', '.join(map(self.visit, n.args))

        func = self.visit(n.func)
        fname = self.context.functions[func][self.dialect]
        if fname is None:
            if self.dialect == _dialect_latex:
                fname = '\\' + func
            else:
                fname = func

        if '%s' in fname:
            if len(n.args) > 1:  # TODO: or ... what ?
                args = self._par(args)
            return fname % args

        return fname + self._par(args)

    def visit_Name(self, n):
        return n.id

    def visit_NameConstant(self, node):
        return str(node.value)

    def visit_UnaryOp(self, n):
        op = self.visit(n.operand)
        if self.prec(n.op) > self.prec(n.operand):
            op = self._par(op)

        symbol = self.context.operators[type(n.op)][self.dialect]

        if '%s' in symbol:
            return symbol % op

        return symbol + op

    def _Bin(self, left, op, right):

        # commute x*3 in 3*x
        if isinstance(op, ast.Mult):
            if isinstance(right, ast.Constant):
                if not Expr(left, self.context).isconstant:
                    return self._Bin(right, op, left)

        l, r = self.visit(left), self.visit(right)

        symbol = self.context.operators[type(op)][self.dialect]

        if '%s' in symbol:  # no parenthesis required in this case
            return symbol % (l, r)

        # handle precedence (parenthesis) if needed

        if self.prec(op) > self.prec(left):
            l = self._par(l)

        if self.prec(op) > self.prec(right):
            if self.dialect == _dialect_latex and isinstance(op, ast.Pow):
                r = '{' + r + '}'
            else:
                r = self._par(r)

        # remove * if possible
        if self.dialect != _dialect_python and isinstance(op, ast.Mult):
            if not l[-1].isdigit() or not r[0].isdigit():
                symbol = ''

        res = l + symbol + r

        # TODO: find a better way to do this ...
        plusminus = self.context.operators[ast.Add][self.dialect] + \
            self.context.operators[ast.USub][self.dialect]
        minusminus = self.context.operators[ast.Sub][self.dialect] + \
            self.context.operators[ast.USub][self.dialect]
        res = res.replace(
            plusminus, self.context.operators[ast.Sub][self.dialect])
        res = res.replace(
            minusminus, self.context.operators[ast.Add][self.dialect])
        return res

    def visit_BinOp(self, n):
        return self._Bin(n.left, n.op, n.right)

    def visit_Compare(self, n):
        # TODO: what to do with multiple ops/comparators ?
        return self._Bin(n.left, n.ops[0], n.comparators[0])

    def visit_Num(self, n):
        try:
            d = self.context.constants[type(n.n)]
            return d[n.n][self.dialect]
        except KeyError:
            pass
        return str(math2.int_or_float(n.n))

    def generic_visit(self, n):
        try:
            l = list(map(self.visit, n))
            return ''.join(l)
        except TypeError:
            pass

        if isinstance(n, ast.AST):
            l = map(self.visit, [getattr(n, f) for f in n._fields])
            return ''.join(l)
        else:
            return str(n)
