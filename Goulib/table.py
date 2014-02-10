#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Table class with Excel + CSV I/O, easy access to columns, HTML output, and much more.
"""
__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2013, Philippe Guglielmetti"
__credits__ = []
__license__ = "LGPL"

import csv, itertools, operator, string
from datetime import datetime, date, timedelta
import logging

try: # using http://lxml.de/
    from lxml import etree as ElementTree
    defaultparser=ElementTree.HTMLParser
except: #ElementTree
    logging.info('LXML unavailable : falling back to ElementTree')
    from xml.etree import ElementTree
    from HTMLParser import HTMLParser
    defaultparser=HTMLParser
    
Element=ElementTree._Element

from Goulib.datetime2 import datef, datetimef,strftimedelta

def attr(args):
    res=''
    for key,val in args.iteritems():
        k="class" if key=="_class" else key
        res+=' %s="%s"'%(k,val)
    return res

class Cell():
    """Table cell with HTML attributes"""
    def __init__(self,data=None,align=None,fmt=None,tag=None,style=None):
        """
        :param data: cell value(s) of any type
        :param align: string for HTML align attribute
        :param fmt: format string applied applied to data
        :param tag: called to build each cell. defaults to 'td'
        :param style: string for HTML style attribute
        """
        
        if isinstance(data,Element):
            tag=data.tag
            assert(tag in ('td','th'))
            data=Cell.read(data.text)
            
        if isinstance(data,basestring):
            data=data.lstrip().rstrip() #remove spaces, but also trailing \r\n
                    
        self.data=data
        self.align=align
        self.fmt=fmt
        self.tag=tag if tag else 'td'
        self.style=style 
        
    def __repr__(self):
        return str(self.data)
    
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
    
    def __repr__html(self):
        return self.html()
        
    def html(self):
        """:return: string HTML formatted cell
        * if data is int, default align="right"
        * if data is float, default align="right" and fmt='%0.2f'
        * if data is :class:`~datetime.timedelta`, align = "right" and formatting is done by :func:`datetime2.strftimedelta`
        """
        v=self.data
        a=self.align
        f=self.fmt
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
            
        attr='' #tag attributes
        if a:
            attr+=' align="%s"'%a
            
        if not v or v=='':
            v="&nbsp;" #for IE8
        else:
            v=f%v if f else str(v)
                    
        if self.style: attr+=' style="%s"'%self.style
        return '<%s%s>%s</%s>'%(self.tag,attr,v,self.tag)

class Row():
    """Table row with HTML attributes"""
    def __init__(self,data,align=None,fmt=None,tag=None,style=None):
        """
        :param data: (list of) cell value(s) of any type
        :param align: (list of) string for HTML align attribute
        :param fmt: (list of) format string applied applied to data
        :param tag: (list of) tags called to build each cell. defaults to 'td'
        :param style: (list of) string for HTML style attribute
        """
        if isinstance(data,Element):
            line=[]
            for td in data:
                cell=Cell(td)
                line.append(cell.data)
            data=line
        
        
        if not isinstance(data,list) : data=list(data)
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
            
        self.data=data
        self.align=align
        self.fmt=fmt
        self.tag=tag
        self.style=style 
        
    def __repr__(self):
        return str(self.data)
    
    def __repr__html(self):
        return self.html()

    def html(self):
        """return in HTML format"""
        res='<tr>'
        for i,v in enumerate(self.data):
            cell=Cell(v,self.align[i],self.fmt[i],self.tag[i],self.style[i])
            res+=cell.html()
        res+='</tr>'
        return res
    
class Table(list):
    """Table class with CSV I/O, easy access to columns, HTML output"""
    def __init__(self,filename=None,titles=None,data=[],**kwargs):
        """inits a table, optionally by reading a Excel, csv or html file"""
        list.__init__(self, data)
        self.titles=titles if titles else []
        if filename:
            if titles: #were specified
                kwargs['titles_line']=0 
            ext=filename.split('.')[-1].lower()
            if ext=='xls':
                self.read_xls(filename,**kwargs)
            elif ext[:3]=='htm':
                self.read_html(filename,**kwargs)
            else: #try ...
                self.read_csv(filename,**kwargs)
            
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
    
    def __repr__html(self):
        return self.html()
    
    def html(self,head=None,foot=None,colstyle=None,**kwargs):
        """:return: string HTML representation of table"""
                
        def TR(data,align=None,fmt=None,tag=None,style=None):
            if not isinstance(data[0],list):
                data=[data]
            res=''
            for line in data:
                row=Row(data=line,align=align,fmt=fmt,tag=tag,style=style)
                res+=row.html()+'\n'
            return res
            
        def THEAD(data,fmt=None,style=None):
            res="<thead>\n"
            res+=TR(data=data,fmt=fmt,tag='th',style=style)
            res+="</thead>\n"
            return res
            
        def TFOOT(data,fmt=None,style=None):
            res="<tfoot>\n"
            res+=TR(data=data,fmt=fmt,tag='th',style=style)
            res+="</tfoot>\n"
            return res
        
        res="<table%s>\n"%attr(kwargs)
            
        if head is None:
            head=self.titles
        if head is not None:
            res+=THEAD(head)
        for row in self:
            res+=TR(row,style=colstyle)  
        if foot is not None:
            res+=TFOOT(foot)             
        
        return res+"</table>\n"
    
    def read_element(self,element, **kwargs):
        """read table from a DOM element"""
        head=element.find('thead')
        if head is not None:
            self.titles=Row(head.find('tr')).data
        body=element.find('tbody')
        if body is None:
            body=element
        for row in body.findall('tr'):
            line=Row(row).data
            if not line: continue #skip empty lines
            if not self.titles:
                self.titles=line
            else:
                self.append(line)
        return self
    
    def read_html(self,filename, **kwargs):
        """read first table in HTML file
        """
        parser=kwargs.get('parser',defaultparser)
        element = ElementTree.parse(filename,parser()).getroot()
        try:
            if element.tag!='table': # file contains table as topmost tag
                element = element.find('.//table') #find first table
        except: 
            pass
        
        if element is None:
            raise LookupError('no table found in file')
        
        self.read_element(element, **kwargs)
        return self
    
    def read_xls(self, filename, **kwargs):
        """appends an Excel table"""
        titles_line=kwargs.get('titles_line',1)-1
        data_line=kwargs.get('data_line',2)-1
        
        from xlrd import open_workbook
        wb = open_workbook(filename)
        for s in wb.sheets():
            for i in range(s.nrows):
                line=[Cell.read(s.cell(i,j).value) for j in range(s.ncols)]
                if i==titles_line:
                    self.titles=line
                elif i>=data_line:
                    self.append(line)
        return self
        
    def read_csv(self, filename, **kwargs):
        """appends a .csv or similar file to the table"""
        titles_line=kwargs.get('titles_line',1)-1
        data_line=kwargs.get('data_line',2)-1
        dialect=kwargs.get('dialect','excel')
        delimiter=kwargs.get('delimiter',';')
        encoding=kwargs.get('encoding','iso-8859-15')
        reader = csv.reader(open(filename, 'rb'), dialect=dialect, delimiter=delimiter)
        #reader = open(filename,'rb') 
        for i,row in enumerate(reader):
            # row=row.replace('\x00', '') #avoid NULLS from .XLS saved as .CSV
            # row=row.rstrip('\r\n')
            #row=row.split(delimiter)
            if encoding:
                row=[x.decode(encoding) for x in row]
            if i==titles_line: #titles can have no left/right spaces
                self.titles=[Cell.read(x) for x in row]
            elif i>=data_line:
                line=[Cell.read(x) for x in row]
                if line!=[None]: #strange last line ...
                    self.append(line)
        return self
            
    def write_csv(self,filename, transpose=False, **kwargs):
        """ write the table in Excel csv format, optionally transposed"""
    
        dialect=kwargs.get('dialect','excel')
        delimiter=kwargs.get('delimiter',';')
        encoding=kwargs.get('encoding','iso-8859-15')
        empty=''.encode(encoding)
        def encode(line): return [empty if s is None else str(s).encode(encoding) for s in line]
        f=open(filename, 'wb')
        writer=csv.writer(f, dialect=dialect, delimiter=delimiter)
        if transpose:
            i=0
            while self.col(i)[0]:
                line=[empty]
                if self.titles: line=[self.titles[i]]
                line.extend(self.col(i))
                writer.writerow(encode(line))
                i+=1
        else:
            if self.titles: writer.writerow(encode(self.titles))
            for line in self:
                writer.writerow(encode(line))
        f.close()
    
    def ncols(self):
        """return number of columns, ignoring title"""
        return reduce(max,map(len,self))
                
    def find_col(self,title):
        """finds a column from a part of the title"""
        title=title.lower()
        for i,c in enumerate(self.titles):
            if c.lower().find(title)>=0:
                return i
        return None

    def _i(self,by):
        '''column index'''
        if isinstance(by, basestring):
            try:
                return self.titles.index(by)
            except ValueError:
                return None
        return by
    
    def icol(self,by):
        '''iterates column'''
        for row in self:
            try:
                yield row[self._i(by)]
            except:
                yield None
                
    def col(self,by):
        return [x for x in self.icol(by)]
    
    def get(self,row,col):
        col=self._i(col)
        return self[row][col]
    
    def set(self,row,col,value):
        col=self._i(col)
        if row>=len(self): 
            self.extend([list()]*(1+row-len(self)))
        if col>=len(self[row]):
            self[row].extend([None]*(1+col-len(self[row])))
        self[row][col]=value
    
    def setcol(self,by,val,i=0):
        '''set column'''
        j=self._i(by)
        for v in val:
            self.set(i,j,v)
            i+=1
            
    def append(self,line):
        ''' appends a line to table
        :param line: can be either:
        * a list
        * a dict or column names:values
        '''
        if isinstance(line,dict):
            r=len(self) #row number
            for k,v in line.iteritems():
                i=self._i(k)
                if i is None: #column doesn't exist:
                    i=len(self.titles)
                    self.titles.append(k)
                self.set(r,i,v)
        else:
            list.append(self,line)
            
    def addcol(self,title,val=None,i=0):
        '''add column to the right'''
        col=len(self.titles)
        self.titles.append(title)
        if not isinstance(val,list):
            val=[val]*(len(self)-i)
        for v in val:
            self.set(i,col,v)
            i+=1
            
    def sort(self,by,reverse=False):
        '''sort by column'''
        i=self._i(by)
        if isinstance(i, int):
            list.sort(self,key=lambda x:x[i],reverse=reverse)
        else:
            list.sort(i,reverse=reverse)
        return self
    
    def rowasdict(self,i):
        ''' returns a line as a dict '''
        return dict(zip(self.titles,self[i]))
        
    def groupby(self,by,sort=True,removecol=True):
        '''dictionary of subtables grouped by a column'''
        i=self._i(by)
        t=self.titles
        if removecol: t=t[:i]+t[i+1:]
        res={}
        if sort: 
            self.sort(i) 
        else:
            pass #groupby will group CONSECUTIVE lines with same i, so entries at bottom of table will replace the earlier entries in dict
        for k, g in itertools.groupby(self, key=lambda x:x[i]):
            if removecol:
                r=Table(titles=t,data=[a[:i]+a[i+1:] for a in g])
            else:
                r=Table(titles=t,data=list(g))
            res[k]=r
        return res
    
    def hierarchy(self,by='Level',
                  factory=lambda row:(row,[]),          #creates an object from a line
                  linkfct=lambda x,y,row:x[1].append(y) #creates a parend/child relation between x and y. raw is also available (for qty)
            ):
        '''builds a structure from a table containing a "level" column'''
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
        
    def applyf(self,by,f,skiperrors=False):
        '''apply a function to a column'''
        i=self._i(by)
        for row in self:
            x=row[i]
            try:
                row[i]=f(x)
            except:
                if not skiperrors:
                    logging.error('could not applyf to %s'%x)
                    raise(ValueError)
                pass
            
    def to_datetime(self,by,fmt='%Y-%m-%d',skiperrors=False):
        '''convert a column to datetime'''    
        self.applyf(by,lambda x: datetimef(x,fmt=fmt),skiperrors)
        
    def to_date(self,by,fmt='%Y-%m-%d',skiperrors=False):
        '''convert a column to date'''
        self.applyf(by,lambda x: datef(x,fmt=fmt),skiperrors)
            

    def total(self,funcs):
        """builds a list by appling f functions to corresponding columns"""
        funcs=funcs+[None]*(len(self.titles)-len(funcs))
        res=[]
        for i,f in enumerate(funcs):
            try:
                res.append(f(self.col(i)))
            except:
                res.append(f)
        return res
    
    def remove_lines_where(self,filter):
        """remove lines on a condition, returns the number of lines removed"""
        res=0
        if len(self)>0:
            for line in reversed(self):
                if filter(line):
                    self.remove(line)
                    res+=1
        return res
    
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
                
