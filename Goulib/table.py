#!/usr/bin/env python
# coding: utf8
"""
"mini pandas.DataFrame"
Table class with Excel + CSV I/O, easy access to columns, HTML output, and much more.
"""
from pandas.core.frame import DataFrame
__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2013, Philippe Guglielmetti"
__credits__ = []
__license__ = "LGPL"

import six, logging
from six.moves import html_parser, reduce

import csv, itertools, codecs, json, collections



import pandas
#from pandas.core.common import PandasError
import numpy as np

try: # using http://lxml.de/
    from lxml import etree as ElementTree
    defaultparser=ElementTree.HTMLParser
except: #ElementTree
    logging.info('LXML unavailable : falling back to ElementTree')
    from xml.etree import ElementTree
    from html_parser import HTMLParser
    defaultparser=HTMLParser

Element=ElementTree._Element

import datetime as std_datetime
from datetime import datetime, date, time, timedelta
from .datetime2 import datef, datetimef, timef, timedeltaf, strftimedelta

from .markup import tag, style_str2dict
from .itertools2 import isiterable

def attr(args):
    res=''
    for key,val in args.items():
        k="class" if key=="_class" else key
        res+=' %s="%s"'%(k,val)
    return res

class Table(pandas.DataFrame):
    """Table class with CSV I/O, easy access to columns, HTML output"""

    def __init__(self,data=None, index=None, columns=None, copy=False, **kwargs):
        """inits a table, optionally by reading a Excel, csv or html file
        :param data: list of list of cells, or string as filename
        :param columns: optional list of strings used as column id
        """
        columns=kwargs.pop('titles',columns) #alias for compatibility
        filename=None
        if isinstance(data,six.string_types):
            filename=data
            data=[]
        elif isinstance(data,pandas.DataFrame):
            columns=columns or data.columns
            index=index or data.index

        try:
            super().__init__(data,index=index,columns=columns,copy=copy)
        except ValueError:
            data = np.array(data)
            if columns is None:
                try:
                    columns=range(data.shape[1])
                except IndexError: # single column
                    columns=[0]
            super().__init__(data,index=index,columns=columns,copy=copy)

        if filename:
            self.load(filename,**kwargs)

    def _init_from(self,dataframe):
        self._data=dataframe._data
        self.__finalize__(dataframe) #copy metadata
        return self

    @property
    def titles(self):
        res=self.axes[-1]
        return res.values

    def row(self,key):
        return Row(self.loc[[key]],copy=False)

    def rowaslist(self,key):
        return self.row(key).values.tolist()[0]
    
    def rowasdict(self,i):
        """returns a line as a dict"""
        return collections.OrderedDict(zip(self.titles,self.rowaslist(i)))
    
    def __getitem__(self, key):
        #allow table[row][column] indexing
        if isinstance(key,(int,slice)):
            return self.row(key)
        #allow table[row,column] indexing
        
        try:
            return self.iloc[key]
        except TypeError:
            pass
        except ValueError: # slice must be defined
            pass
        return pandas.DataFrame.__getitem__(self, key)
    
    def ncols(self):
        return len(self.columns)

    def col(self,key):
        return self[key].values.tolist()

    
    def append(self,data):
        """appends data to table
        :param data: can be either:
        * a DataFrame (or Table)
        * a Series with name (column)
        * a dict of column names:values
        """
        if isinstance(data,(list,tuple)): #lines, not columns as in Pandas
            self.loc[len.self]=data
        elif isinstance(data,dict): #lines, not columns as in Pandas
            for k in data:
                self.loc[len(self):k]=data[k]
        else:
            super().append(data)
        return self

    def addcol(self,title,val=None):
        """add column to the right"""
        if not isiterable(val):
            val=[val]*(len(self))
        c=pandas.Series(val, index=self.index)
        self.loc[:,title]=c
        return self

    def __eq__(self,other):
        return self.equals(other)

    def __ne__(self,other):
        return not self.equals(other)

    def load(self,filename,**kwargs):
        if any(self.titles): # already explicitly set
            kwargs['names']=self.titles
            
        kwargs.setdefault('encoding','utf-8') #iso-8859-15 in some cases

        ext=filename.split('.')[-1].lower()
        if ext in ('xls','xlsx'):
            res=pandas.read_excel(filename,**kwargs)
        elif ext in ('htm','html'):
            res=pandas.read_html(filename,**kwargs)[0] # first <table> in file
        elif ext in ('json'):
            kwargs.pop('names',None)
            res=pandas.read_json(filename,**kwargs)
        else: #try ...
            kwargs.setdefault('delimiter',';')
            res=pandas.read_csv(filename,**kwargs)
        self._init_from(res)
        return self #to allow chaining

    
    def save(self,filename,**kwargs):
        ext=filename.split('.')[-1].lower()
        kwargs.setdefault('index',False) # do not create index column
        self.sort_index(inplace=True) # make sure order is preserved
        
        if ext in ('xls','xlsx'):
            self.write_xlsx(filename,**kwargs)
        elif ext in ('htm','html'):
            self.to_html(filename, **kwargs)
        elif ext in ('json'):
            kwargs.pop('index') #unused
            kwargs.setdefault('orient','records') # 'index'
            self.to_json(filename, **kwargs)
        else: #try ...
            kwargs.setdefault('encoding','utf-8') 
            kwargs.setdefault('sep',';') 
            self.to_csv(filename, **kwargs)
        return self #to allow chaining

    
    def write_xlsx(self,filename,
            sheetname='Sheet1',
            df='dd/mm/yyyy',
            tf='hh:mm:ss',
            dtf='dd/mm/yyyy hh:mm:ss',
            **kwargs):
        
        #https://xlsxwriter.readthedocs.io/working_with_pandas.html
        writer = pandas.ExcelWriter(filename, engine='xlsxwriter',)
        '''
            date_format=df,
            time_format=tf,
            datetime_format=dtf,
        )
        '''
        
        self.to_excel(writer, sheet_name=sheetname, **kwargs)
        writer.save()
        return self
    
    

    def applyf(self,col,f,skiperrors=False,skipnull=True):
        """ apply a function to a column
        
        :param col: column name of number
        :param f: function of the form lambda cell:content
        :param skiperrors: bool. if True, errors while running f are ignored
        :return: bool True if ok, False if skiperrors==True and function failed
        """
        res=True
        def f2(x):
            try:
                return f(x)
            except ValueError:
                if skipnull or skiperrors:
                    return x
                raise
            except Exception:
                if skiperrors:
                    return x
                raise
            
        self[col] = self[col].apply(f2)
        return res
    
    def remove_lines_where(self,col,values=(None,0,'')):
        """
        :param col: name of a column
        :param values: values for which the row must be removed
        :return: int number of lines removed
        """
        n=len(self)
        if not isiterable(values):
            values=list(values)
        self.drop(self[col].isin(values),inplace=True)
        return n-len(self)
    
    def sort(self,by,reverse=False):
        """sort by column"""
        self.sort_values(by, ascending=not reverse, inplace=True)
        return self
    
    def groupby_gen(self,by,sort=True,removecol=True):
        """generates subtables
        """
        if sort:
            self.sort(by)
        for k in self[by].unique():
            t=Table(self[by==k])
            if removecol: del t[by]
            yield k,t

    def groupby(self,by,sort=True,removecol=True):
        """ ordered dictionary of subtables
        """
        return collections.OrderedDict(
            (k,t) for (k,t) in self.groupby_gen(by,sort,removecol)
        )

    def _datetimeformat(self,by,fmt,function,skiperrors):
        """convert a column to a date, time or datetime
        :param by: column name of number
        :param fmt: string defining format, or list of formats to try one by one
        :param function: function to call
        :param skiperrors: bool. if True, conversion errors are ignored
        :return: bool True if ok, False if skiperrors==True and conversion failed
        """
        if isinstance(fmt,list):
            for f in fmt:
                res=self._datetimeformat(by, f, function, True if f!=fmt[-1] else skiperrors)
            return res
        return self.applyf(by,lambda x: function(x,fmt=fmt),skiperrors)

    def to_datetime(self,by,fmt='%Y-%m-%d %H:%M:%S',skiperrors=False):
        """convert a column to datetime
        """
        # https://stackoverflow.com/questions/17134716/convert-dataframe-column-type-from-string-to-datetime
        return self._datetimeformat(by, fmt, datetimef, skiperrors)

    def to_date(self,by,fmt='%Y-%m-%d',skiperrors=False):
        """convert a column to date
        """
        return self._datetimeformat(by, fmt, datef, skiperrors)

    def to_time(self,by,fmt='%H:%M:%S',skiperrors=False):
        """convert a column to time
        """
        return self._datetimeformat(by, fmt, timef, skiperrors)

    def to_timedelta(self,by,fmt=None,skiperrors=False):
        """convert a column to time
        """
        return self._datetimeformat(by, fmt, timedeltaf, skiperrors)

    def _repr_html_(self):
        return self.style._repr_html_()

    def html(self):
        return self.style.render(id='')

    '''

    def __repr__(self):
        """:return: repr string of titles+5 first lines"""
        return 'Table(len=%d,titles=%s,data=%s)'%(len(self),self.titles,self[:5])

    def __str__(self):
        """:return: string of full tables with linefeeds"""
        res=''
        if self.titles:
            res+=str(self.titles)+'\n'
        for line in self:
            res+=str(line)+'\n'
        return res

    def _repr_html_(self):
        return self.html()

    def html(self,head=None,foot=None,colstyle={},**kwargs):
        """HTML representation of table

        :param head: optional column headers, .titles by default
        :param foot: optional column footers, .footer by default
        :param style: (list of) dict of style attributes
        :param kwargs: optional parameters passed along to tag('table'...
            except:
            * start=optional start row
            * stop=optional end row
            used to display a subset of lines. in this case rows with '...' cells
            are displayed before and/or after the lines
        :return: string HTML representation of table
        """

        def TR(data,align=None,fmt=None,tag=None,style={}):
            res=''
            row=Row(data=data,align=align,fmt=fmt,tag=tag,style=style)
            res+=row.html()+'\n'
            return res

        def THEAD(data,fmt=None,style={}):
            res="<thead>\n"
            res+=TR(data=data,fmt=fmt,tag='th',style=style)
            res+="</thead>\n"
            return res

        def TFOOT(data,fmt=None,style={}):
            res="<tfoot>\n"
            res+=TR(data=data,fmt=fmt,tag='th',style=style)
            res+="</tfoot>\n"
            return res

        res=''

        if head is None:
            head=self.titles
        if head:
            res+=THEAD(head)
        start=kwargs.pop('start',0)
        stop=kwargs.pop('stop',len(self))
        if start!=0:
            res+=TR(['...']*self.ncols(),style=colstyle)
        for row in self[start:stop]:
            res+=TR(row,style=colstyle)
        if stop!=-1 and stop<len(self):
            res+=TR(['...']*self.ncols(),style=colstyle)
        if foot is None:
            foot=self.footer
        if foot:
            res+=TFOOT(foot)

        return tag('table',res,**kwargs)+'\n'





    def read_json(self,filename, **kwargs):
        """appends a json file made of lines dictionaries"""
        with open(filename, 'r') as file:
            for row in json.load(file):
                self.append(row)
        return self

    
    

    def json(self, **kwargs):
        """
        :return: string JSON representation of table
        """
        def json_serial(obj):
            """JSON serializer for objects not serializable by default json code"""
            if isinstance(obj, (datetime,date,time)):
                return obj.isoformat()
            if isinstance(obj, (timedelta)):
                return str(obj)
            raise TypeError ("Type %s not serializable"%(type(obj)))
        array=[self.rowasdict(i) for i in range(len(self))]
        kwargs.setdefault('default',json_serial)
        return json.dumps(array, **kwargs)

    def __eq__(self,other):
        """compare 2 Tables contents, mainly for tests"""
        if self.titles!=other.titles:
            return False
        if len(self)!=len(other):
            return False
        for i in range(len(self)):
            if self[i]!=other[i]:
                return False
        return True

    def ncols(self):
        """
        :return: number of columns, ignoring title
        """
        return reduce(max,list(map(len,self)))

    def find_col(self,title):
        """finds a column from a part of the title"""
        title=title.lower()
        for i,c in enumerate(self.titles):
            if c.lower().find(title)>=0:
                return i
        return None

    def _i(self,column):
        """column index"""
        if isinstance(column, int):
            return column
        try:
            return self.titles.index(column)
        except ValueError:
            return None

    def icol(self,column):
        """iterates a column"""
        i=self._i(column)
        for row in self:
            try:
                yield row[i]
            except:
                yield None

    def col(self,column,title=False):
        i=self._i(column)
        res=[x for x in self.icol(i)]
        if title:
            res=[self.titles[i]]+res
        return res

    def cols(self,title=False):
        """iterator through columns"""
        for i in range(self.ncols()):
            yield self.col(i,title)

    def transpose(self,titles_column=0):
        """transpose table
        :param: titles_column
        :return: Table where rows are made from self's columns and vice-versa
        """
        res=Table()
        for i,row in enumerate(self.cols(self.titles)):
            if i==titles_column:
                res.titles=row
            else:
                res.append(row)
        return res

    def index(self,value,column=0):
        """
        :return: int row number of first line where column contains value
        """
        for i,v in enumerate(self.icol(column)):
            if v==value:
                return i
        return None

    def get(self,row,col):
        return self[row,col]


    def set(self,row,col,value):
        col=self._i(col)
        if row>=len(self):
            self.extend([list()]*(1+row-len(self)))
        if col>=len(self[row]):
            self[row].extend([None]*(1+col-len(self[row])))
        self[row][col]=value

    def setcol(self,col,value,i=0):
        """set column values
        :param col: int or string column index
        :param value: single value assigned to whole column or iterable assigned to each cell
        :param i: optional int : index of first row to assign
        """
        j=self._i(col)
        if isiterable(value):
            for v in value:
                self.set(i,j,v)
                i+=1
        else:
            for i in range(i,len(self)):
                self.set(i,j,value)


    def asdict(self):
        for i in range(len(self)):
            yield self.rowasdict(i)


    def hierarchy(self,by='Level',
                  factory=lambda row:(row,[]),          #creates an object from a line
                  linkfct=lambda x,y,row:x[1].append(y) #creates a parend/child relation between x and y. raw is also available (for qty)
            ):
        """builds a structure from a table containing a "level" column"""
        res=[]
        i=self._i(by)
        stack=[]
        for row in self:
            obj=factory(row)
            level=row[i]
            if level==1:
                res.append(obj)
            while level<=len(stack):
                stack.pop()
            if stack:
                linkfct(stack[-1],obj,row)
            stack.append(obj)
        return res

    def total(self,funcs):
        """build a footer row by appling funcs to all columns
        """
        funcs=funcs+[None]*(len(self.titles)-len(funcs))
        self.footer=[]
        for i,f in enumerate(funcs):
            try:
                self.footer.append(f(self.col(i)))
            except:
                self.footer.append(f)
        return self.footer


'''

class Row(Table):
    """ a DataFrame of a single Table row"""
    def __getitem__(self, key):
        if isinstance(key,int):
            key=self.titles[key]
        res=super(Table,self).__getitem__(key)
        return res.values[0] if res.size==1 else res
    def __setitem__(self, key,value):
        if isinstance(key,int):
            key=self.titles[key]
        return super(Table,self).__setitem__(key,value)

'''
class Row(object):
    """Table row with HTML attributes"""
    def __init__(self,data,align=None,fmt=None,tag=None,style={}):
        """
        :param data: (list of) cell value(s) of any type
        :param align: (list of) string for HTML align attribute
        :param fmt: (list of) format string applied applied to data
        :param tag: (list of) tags called to build each cell. defaults to 'td'
        :param style: (list of) dict or string for HTML style attribute
        """

        if not isiterable(data) :
            data=[data]
        data=list(data) #to make it mutable

        #ensure params have the same length as data

        if not isinstance(style,list): style=[style]
        style=style+[None]*(len(data)-len(style))

        if not isinstance(align,list): align=[align]
        align=align+[None]*(len(data)-len(align))

        if not isinstance(fmt,list)  : fmt=[fmt]
        fmt=fmt+[None]*(len(data)-len(fmt))

        if not tag:tag='td'
        if not isinstance(tag,list)  :
            tag=[tag]*(len(data)) #make a full row, in case it's a 'th'
        tag=tag+[None]*(len(data)-len(fmt)) #fill the row with None, which will be 'td's

        for i,cell in enumerate(data):
            if not isinstance(cell,Cell):
                cell=Cell(cell,align[i],fmt[i],tag[i],style[i])
            else:
                pass #ignore attribs for now
            data[i]=cell



        self.data=data

    def __repr__(self):
        return str(self.data)

            def _repr_html_(self):
        return self.html()

    def html(self,cell_args={},**kwargs):
        """return in HTML format"""
        res=''
        for k,v in self.iteritems():
            cell=Cell(v)
            res+=cell.html(**cell_args)
        return tag('tr',res,**kwargs)


'''

class Cell(object):
    """Table cell with HTML attributes"""
    def __init__(self,data=None,align=None,fmt=None,tag=None,style={}):
        """
        :param data: cell value(s) of any type
        :param align: string for HTML align attribute
        :param fmt: format string applied applied to data
        :param tag: called to build each cell. defaults to 'td'
        :param style: dict or string for HTML style attribute
        """

        if isinstance(data,Element):
            tag=data.tag
            assert(tag in ('td','th'))

            def _recurse(data):
                "grab all possible text from the cell content"
                if data is None:
                    return ''
                s=''.join(_recurse(x) for x in data.getchildren())
                if data.text:
                    s=data.text+s
                if data.tail:
                    s=s+data.tail
                return s

            data=Cell.read(_recurse(data))

        if isinstance(data,str):
            data=data.lstrip().rstrip() #remove spaces, but also trailing \r\n

        self.data=data
        self.align=align
        self.fmt=fmt
        self.tag=tag if tag else 'td'
        if not isinstance(style,dict):
            style=style_str2dict(style)
        self.style=style

    def __repr__(self):
        return str(self.data)

    def _repr_html_(self):
        return self.html()

    @staticmethod
    def read(x):
        """interprets x as int, float, string or None"""
        try:
            x=float(x) #works for x in unicode
            xi=int(x)  #does not work if x is floating point in unicode (?)
            if xi==x: x=xi
        except:
            if x=='': x=None
        return x

    def html(self,**kwargs):
        """:return: string HTML formatted cell:

        * if data is int, default align="right"
        * if data is float, default align="right" and fmt='%0.2f'
        * if data is :class:`~datetime.timedelta`, align = "right" and formatting is done by :func:`datetime2.strftimedelta`

        """
        args={}
        args.update(kwargs) #copy needed to avoid side effects

        v=self.data
        f=self.fmt

        if hasattr(v,'_repr_html_'):
            try:
                s=v._repr_html_()
            except Exception as e:
                s='ERROR : %s _repr_html_ failed : %s'%(v,e)
            return tag(self.tag,s,**args)

        style=args.get('style',{})
        if not isinstance(style,dict):
            style=style_str2dict(style)

        if not 'text-align' in style: #HTML 4 and before
            a=args.pop('align',self.align)
            if isinstance(v,int):
                if not a: a="right"
            elif isinstance(v,float):
                if not a: a="right"
                if not f: f='%0.2f'
                v=f%v
                f=None #don't reformat below
            elif isinstance(v,date):
                if not a: a="right"
                if not f: f='%Y-%m-%d'
                v=v.strftime(f)
                f=None #don't reformat below
            elif isinstance(v,timedelta):
                if not a: a="right"
                if not f: f='%H:%M:%S'
                v=strftimedelta(v,f)
                f=None #don't reformat below

            if a:
                style['text-align']=a

        # create style dict by merging default Cell style + parameters
        style=dict(self.style,**style) #http://stackoverflow.com/questions/9819602/union-of-dict-objects-in-python
        if style:
            args['style']=style

        if v is None or v=='':
            v="&nbsp;" #for IE8
        else:
            v=f%v if f else six.text_type(v)
        return tag(self.tag,v,**args)


