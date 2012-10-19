#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Table class with CSV I/O, easy access to columns, HTML output
"""
__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2012, Philippe Guglielmetti"
__credits__ = []
__license__ = "LGPL"

import csv, itertools, operator, string
import datetime
import logging
    
class Table(list):
    """Table class with CSV I/O, easy access to columns, HTML output"""
    def __init__(self,filename=None,titles=[],init=[],**kwargs):
        """inits a table either from data or csv file"""
        list.__init__(self, init)
        self.titles=titles
        if filename:
            if titles: #were specified
                kwargs['titles_line']=0 
            self.read_csv(filename,**kwargs)
        
    def read_csv(self, filename, **kwargs):
        """appends a .csv or similar file to the table"""
        titles_line=kwargs.get('titles_line',1)
        data_line=kwargs.get('data_line',2)
        dialect=kwargs.get('dialect','excel')
        delimiter=kwargs.get('delimiter',';')
        encoding=kwargs.get('encoding','iso-8859-1')
        titles_line-=1 #0 base
        data_line-=1 #0 base
        #reader = csv.reader(open(filename, 'rb'), dialect=dialect, delimiter=delimiter)
        reader = open(filename,'rb') 
        for i,row in enumerate(reader):
            row=row.replace('\x00', '') #avoid NULLS from .XLS saved as .CSV
            row=row.rstrip('\r\n')
            
            row=row.split(delimiter)
            if encoding:
                res=[]
                for x in row:
                    try:
                        x.decode('ascii')
                    except UnicodeDecodeError:
                        x=x.decode(encoding)
                    res.append(x)
                row=res
            if i==titles_line: #titles can have no left/right spaces
                self.titles=[x.lstrip().rstrip() for x in row]
            elif i>=data_line:
                line=[]
                for x in row:
                    try:
                        x=float(x) if '.' in x else int(x)
                    except:
                        if x=='': x=None
                    line.append(x)
                if line!=[None]: #strange last line ...
                    self.append(line)
            
    def write_csv(self,filename, dialect='excel', delimiter=';', transpose=False):
        """ write the table in Excel csv format, optionally transposed"""
        writer=csv.writer(open(filename, 'wb'), dialect=dialect, delimiter=delimiter)
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
            return self.titles.index(by)
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
    
    def set(self,row,col,value):
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
    def addcol(self,title,val,i=0):
        '''add column to the right'''
        col=len(self.titles)
        self.titles.append(title)
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
            
    def groupby(self,by):
        '''dictionary of subtables grouped by a column'''
        i=self._i(by)
        self.sort(i)
        t=self.titles[:i]+self.titles[i+1:]
        res={}
        for k, g in itertools.groupby(self, key=lambda x:x[i]):
            res[k]=Table([a[:i]+a[i+1:] for a in list(g)])
            res[k].titles=t
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
                    logging.warning('applyf could not process %s'%row[i])
            else: # will fail if any problem occurs 
                row[i]=f(row[i])
            
    def to_datetime(self,by,fmt='%d.%m.%Y',safe=True):
        '''convert a column to datetime'''
        self.applyf(by,lambda x:datetime.datetime.strptime(x,fmt),safe)
        
    def to_date(self,by,fmt='%d.%m.%Y',safe=True):
        '''convert a column to date'''
        self.to_datetime(by,fmt,safe)
        self.applyf(by,lambda x:x.date(),safe)
            
    def __str__(self):
        res=''
        if self.titles:
            res+=str(self.titles)+'\n'
        for line in self:
            res+=str(line)+'\n'
        return res
    
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
    
    def html(self,page,head=None,foot=None,width=None):
        """output page as HTML, optionally generating a "total" line by mapping functions to columns"""
        page.table()
        if head:
            head=[self.titles,head]
        else:
            head=self.titles
        page.THEAD(head,width=width)
        for row in self:
            page.TR(row)  
        if foot:
            page.TFOOT(foot)             
        page.table.close()
    
if __name__ == '__main__':
    t=Table(titles=['A','B','C'])
    t.append([1,2.0,3.3])
    t.append(['one','two','three'])
    t.append([None,['a','b','c'],0])
    print t
    print t.html()