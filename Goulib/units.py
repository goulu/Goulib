'''
Created on 8 oct. 2014

@author: Marc Nicole
'''
import unittest
from pint import UnitRegistry #https://pypi.python.org/pypi/Pint/
from pprint import pprint
from inspect import isfunction,getsource


#defines V as a way to express any physical value (and Currency)
ureg = UnitRegistry()
V = ureg.Quantity

def magnitudeIn(self,unit):
    return self.to(unit).magnitude
V.__call__ = magnitudeIn

class Row():
    """
    a Raw is internally a dict of {'colname1':value,'colname2',value...}
    
    """
    def __init__(self,row):  #row has the same format as internal
        self.row = row
       
    def __getitem__(self,col): 
        v = self.row[col]
        if isfunction(v): #in that case, one must resolve the calculation
            v = v()
        return v
    
    def __setitem__(self,col,value):
        self.row[col] = value
       
    def isfunc(self,col):
        return isfunction(self.row[col])
    
    def _repr_html_(self,col,convertToUnits=''):
        if self.isfunc(col):
            c = self.row[col]
            style = 'style="background-color: #f1f1c1" title="'+ getsource(c)+'"'
        else:
            style = ''
            
        try:
            disp = self[col] if convertToUnits=='' else self[col].to(convertToUnits)
            return '<td align="right" '+style+'>{0.magnitude}</td><td>{0.units}</td>'.format(disp)
        except:
            return '<td colspan=2>'+str(self[col])+'</td>'            
        

     
class Table():
    """
    to create a table, use the following syntax
    
    t =  Table ('myTable', ['col1',          'col2',            'col3'],
            [ 'distance',   V(20,'m'),       V(30,'m'),         V(40,'m'),
              'speed',      V(2,'m/min'),    V(5,'m/s'),        V(10,'inch/s')
            ])
            
    in the future, the label could be replaced by a dictionary having more properties like 
    {'label':<label>, 'comment':<comment>....}
    
            
    ---------
    internally, the rows are represented as
    {'<the label>':{'col1':<value>, 'col2':<value> ....},
     '<the next label>'         
    """
    def __init__(self,name,cols,cells):
        self.name = name
        self.cols = cols
        self.rows = {}
        self.rowLabels = [] # in order to keep the natural order of the table
        
        i = iter(cells)
        tableOk = False  #just here to check consistency of rows
        try:
            while True:
                tableOk = True
                label = next(i)
                self.rowLabels.append(label)
                row = {}
                tableOk = False
                for col in cols:
                    row[col] = next(i)
                self.rows[label] = Row(row)    
                tableOk = True
        except StopIteration:
            if not tableOk:
                raise Exception('the cells array has inconsistent number of cells regarding the number of columns passed in cols')
            
    def __getitem__(self,key):
        return self.rows[key]
        
    def _repr_html_(self):
        html = '<table border="1"><caption>'+self.name+'</caption><tr><th></th>'
        for col in self.cols:
            html += '<th colspan="2">'+col+'</th>'
        html += '</tr><tr>'
        for row in self.rowLabels:
            html += '<td>'+row+'</td>'
            r = self.rows[row]
            for col in self.cols:
                html += r._repr_html_(col)
            html += '</tr>'
        html += '</table>'
        return html
        
        
    def appendCol(self,colname,values):
        """ appends a column at the right of the table
            :parameter values: is a dict of {<row>:V(...),...}
        """
        self.cols.append(colname)
        for rowname in values:
            self.rows[rowname][colname] = values[rowname]
            
class View():
    def __init__(self,table,rows=[],cols=[],rowUnits={},name=''):
        self.table = table
        self.rows = rows if rows != [] else table.rowLabels
        self.name = name if name !='' else table.name
        
        if cols == []:
            self.cols = table.cols
        else:
            self.cols = cols
        self.rowUnits = rowUnits

    def _repr_html_(self):
        html = '<table border="1"><caption>'+self.name+'</caption><tr><th></th>'
        for col in self.cols:
            html += '<th colspan="2">'+col+'</th>'
        html += '</tr><tr>'
        for row in self.rows:
            if row in self.rowUnits:
                units = self.rowUnits[row] 
                html += '<td>'+row+'['+units+']</td>'
            else:
                units = '' 
                html += '<td>'+row+'</td>'
            r = self.table.rows[row]    
            for col in self.cols:
                html += r._repr_html_(col,convertToUnits=units)
            html += '</tr>'
        html += '</table>'
        return html        
        
        
        
class Test(unittest.TestCase):
    

    def test001_simple_value(self):
        ureg.define('CHF = [Currency]')
        ureg.define('EUR = 1.21*CHF')
        ureg.define('USD = 0.93*CHF')
        dist = V(1000,'m')
        self.assertEqual(str(dist),'1000 meter')
        speed = V(10,'m/s')
        self.assertEqual(str(speed),'10 meter / second')
        time = dist/speed
        self.assertEqual(str(time),'100.0 second')
        hourlyRate = V(50,'USD/hour')
        cost = hourlyRate*time
        self.assertEqual(str(cost.to('CHF')),'1.2916666666666667 CHF')

    def test002_table(self):
        t = Table(   'mytable',      ['car',          'bus',                                     'pedestrian'],
                  [  'speed',        V(120,'km/hour'), V(100,'km/hour'),                          V(5,'km/hour'),
                     'acceleration', V(1,'m/s^2'),     V(0.1,'m/s^2'),                            V(0.2,'m/s^2'),
                     'autonomy',     V(600,'km'),      lambda: t['autonomy']['pedestrian']*10,    lambda: t['speed']['pedestrian']*V(6,'hour') #coucou
                  ])
        self.assertCountEqual(t.cols,['car',        'bus',         'pedestrian'])
        self.assertCountEqual(t.rowLabels,['speed','acceleration','autonomy'])
        pprint(t.rows)
        self.assertEqual(t['speed']['bus'],V(100,'km/hour'))
        print(t._repr_html_())
        
        v = View(t,rows=['autonomy','speed'],cols=['car','pedestrian'],rowUnits={'speed':'mile/hour'},name='my view')
        print(v._repr_html_())
        
        t.appendCol('cheval',{'speed':V(60,'km/hour'),'acceleration':V(0.3,'m/s^2'),'autonomy':V(40,'km')})
        print(t._repr_html_())
        
    def test003_m(self):
        v = V(60,'m/min')
        self.assertEqual(v('m/s'), 1)
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()