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

from Goulib.datetime2 import datef, datetimef
    
class Table(list):
    """Table class with CSV I/O, easy access to columns, HTML output"""
    def __init__(self,filename=None,titles=[],init=[],**kwargs):
        """inits a table either from data or csv file"""
        list.__init__(self, init)
        self.titles=titles
        if filename:
            if titles: #were specified
                kwargs['titles_line']=0 
            if filename[-4:].lower()=='.xls':
                self.read_xls(filename,**kwargs)
            else:
                self.read_csv(filename,**kwargs)
                
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
            
    def __repr__(self):
        """:return: repr string of titles+5 first lines"""
        return 'Table(%s,%s)'%(self.titles,self[:5])
    
    def __str__(self):
        """:return: string of full tables with linefeeds"""
        res=''
        if self.titles:
            res+=str(self.titles)+'\n'
        for line in self:
            res+=str(line)+'\n'
        return res
    
    def html(self,page,head=None,foot=None,colstyle=None,**kwargs):
        """output HTML on a markup.page"""
        page.table(**kwargs) #if style is defined, it's applied to the table here
        if not head:
            head=self.titles
        if head:
            page.THEAD(head)
        for row in self:
            page.TR(row,style=colstyle)  
        if foot:
            page.TFOOT(foot)             
        page.table.close()
    
    def read_xls(self, filename, **kwargs):
        """appends an Excel table"""
        titles_line=kwargs.get('titles_line',1)-1
        data_line=kwargs.get('data_line',2)-1
        
        from xlrd import open_workbook
        wb = open_workbook(filename)
        for s in wb.sheets():
            for i in range(s.nrows):
                line = []
                for j in range(s.ncols):
                    x=s.cell(i,j).value
                    try:
                        xf=float(x)
                        xi=int(x)
                        x=xi if xi==xf else xf
                    except:
                        if x=='': x=None
                    line.append(x)
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
        #reader = csv.reader(open(filename, 'rb'), dialect=dialect, delimiter=delimiter)
        reader = open(filename,'rb') 
        for i,row in enumerate(reader):
            row=row.replace('\x00', '') #avoid NULLS from .XLS saved as .CSV
            row=row.rstrip('\r\n')
            
            row=row.split(delimiter)
            if encoding:
                row=[x.decode(encoding) for x in row]
            if i==titles_line: #titles can have no left/right spaces
                self.titles=[x.lstrip().rstrip() for x in row]
            elif i>=data_line:
                line=[]
                for x in row:
                    try:
                        xf=float(x)
                        xi=int(xf)
                        x=xi if xi==xf else xf
                    except:
                        if x=='': x=None
                    line.append(x)
                if line!=[None]: #strange last line ...
                    self.append(line)
        return self
            
    def write_csv(self,filename, transpose=False, **kwargs):
        """ write the table in Excel csv format, optionally transposed"""
        dialect=kwargs.get('dialect','excel')
        delimiter=kwargs.get('delimiter',';')
        encoding=kwargs.get('encoding','iso-8859-15')
        f=open(filename, 'wb')
        writer=csv.writer(f, dialect=dialect, delimiter=delimiter)
        if transpose:
            i=0
            while self.col(i)[0]:
                line=['']
                if self.titles: line=[self.titles[i]]
                line.extend(self.col(i))
                writer.writerow(line)
                i+=1
        else:
            if self.titles: writer.writerow(self.titles)
            for line in self:
                writer.writerow(line)
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
            val=[val]*len(self)
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
            res[k]=Table(None,titles=t,init=[a[:i]+a[i+1:] if removecol else a for a in g])
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
        
    def applyf(self,by,f,safe=True):
        '''apply a function to a column'''
        i=self._i(by)
        for row in self:
            if safe:
                try:
                    row[i]=f(row[i])
                except:
                    pass
                    # logging.debug('applyf could not process %s'%row[i]) #might cause another error
            else: # will fail if any problem occurs 
                row[i]=f(row[i])
            
    def to_datetime(self,by,fmt='%d.%m.%Y',safe=True):
        '''convert a column to datetime'''    
        self.applyf(by,lambda x: datetimef(x,fmt=fmt),safe)
        
    def to_date(self,by,fmt='%d.%m.%Y',safe=True):
        '''convert a column to date'''
        self.applyf(by,lambda x: datef(x,fmt=fmt),safe)
            

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
                
if __name__ == "__main__":
    import nose
    nose.runmodule()