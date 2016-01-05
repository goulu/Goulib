#!/usr/bin/env python
# coding: utf8
"""
motion simulation (kinematics)
"""
from Goulib.statemachine import StateMachine,StateChangeLog,TooLateLog,WaitLog,noPrint
import Goulib.table

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2013, Philippe Guglielmetti"
__credits__= ["http://osterone.bobstgroup.com/wiki/index.php?title=UtlCam"]
__license__ = "LGPL"

from math import sin
from . import plot, polynomial, itertools2, math2
from Goulib.units import V,Table, View

from numpy import allclose

class PVA(plot.Plot): #TODO: make it an Expr
    """represents a function of time returning position, velocity, and acceleration
    """
    
    def __init__(self,funcs):
        self.funcs=funcs
        
    def __call__(self,t,t0=0):
        return tuple(f(t-t0) for f in self.funcs)

class Segment(PVA):
    """ a PVA defined between 2 times, null elsewhere
    """
    
    def __init__(self,t0,t1,funcs):
        super(Segment, self).__init__(funcs)
        self.t0=t0
        self.t1=t1
        self.ticks= []
        
    def dt(self):
        return self.t1-self.t0
        
    def start(self):
        return super(Segment, self).__call__(self.t0, self.t0)
    
    def startPos(self):
        return self.start()[0]
    
    def startSpeed(self):
        return self.start()[1]
    
    def startAcc(self):
        return self.start()[2]
    
    def startJerk(self):
        return self.start()[3]
    
    def startTime(self):
        return self.t0
    
    def end(self):
        return super(Segment, self).__call__(self.t1, self.t0)
    
    def endPos(self):
        return self.end()[0]
    
    def endSpeed(self):
        return self.end()[1]
    
    def endAcc(self):
        return self.end()[2]
    
    def endJerk(self):
        return self.end()[3]
    
    def endTime(self):
        return(self.t1)
    
    def timeWhenPosBiggerThan(self,pos,resolution=0.010):
        """ search the first time when the position is bigger than pos
        :params pos: the pos that must at least be reached
        :params resolution: the time resolution in sec"""
        #brute force and stupid!!!
        for t in itertools2.arange(self.t0,self.t1,resolution):
            if self(t)[0] >= pos:
                break
        return t
        
    def __call__(self,t):
        if t>=self.t0 and t<self.t1:
            return super(Segment, self).__call__(t, self.t0)
        else:
            return (0,)*len(self.funcs)
        
    def _plot(self, ax, t0=None,t1=None, ylim=None, **kwargs):
        """
        :params ticks: a list of (time,state) to add ticks
        """
        if t0 is None: t0=self.t0
        if t1 is None: t1=self.t1
        step=(t1-t0)/500.
        x=[ t for t in itertools2.arange(t0,t1,step)  ]
        y = [self(t) for t in x]
        y=map(list, zip(*y)) #transpose because of 
        labels=['pos','vel','acc','jrk']
        for y_arr, label in zip(y, labels):
            ax.plot(x, y_arr, label=label)
        ax.legend(loc='best')
#        for tick in self.ticks:
#            ax.axvline(x=tick[0],color='0.5')
        return ax
    
class Segments(Segment):
    def __init__(self,segments=[],label='Segments'):
        """
        can be initialized with a list of segment (that of course can also be a Segments)
        :param label: a label can be given
        """
        self.label = label
        self.t0 = -float('inf')
        self.t1 = -float('inf')
        self.segments = []
        self.add(segments)
        
    def __str__(self):
        return self.label
    
    def html(self):
        t = self.label+'<br/>'
        for s in self.segments:
            t += 't=%f (%f,%f,%f,%f) --> t=%f (%f,%f,%f,%f)<br/>'%(s.t0,s.start()[0],s.start()[1],s.start()[2],s.start()[3],
                                                                   s.t1,s.endPos(),s.endSpeed(),s.endAcc(),s.end()[3])
        return t
            
    def update(self):
        """ yet only calculates t0 and t1 """
        for s in self.segments:
            if s.t0 < self.t0:
                self.t0 = s.t0
            if s.t1 > self.t1:
                self.t1 = s.t1

    def insert(self,segment,autoJoin=True):
        """ insert a segment into Segments
        :param segment: the segment to add. must be in a range that is not already defined or it will rise a value error exception
        :param autoJoin: if True and the added segment has the same starting position as the last segment's end
                         and both velocity are 0 then a segment of (pos,v=0,a=0) is automatically added.
                         this help discribing movements only where there is curently a movement
        """
        
        t0 = segment.t0
        t1 = segment.t1
        if self.segments == []:
            self.segments = [segment]
            self.t0 = segment.t0
            self.t1 = segment.t1
            return
        if t0 >= self.segments[-1].t1:
            previous = self.segments[-1]
            previousP = previous.endPos()
            previousT = previous.endTime()
            previousV = previous.endSpeed()
            segmentP = segment.start()[0]
            segmentV = segment.start()[1]
            if autoJoin and t0 > previousT and allclose([previousP,previousV,segmentV],[segmentP,0,0],atol=0.001):
                self.segments.append(SegmentPoly(previous.t1,t0,[previous.endPos()]))
            self.segments.append(segment)
            return
        if t1 <= self.segments[0].t0:
            self.segments.insert(0,segment)
            return
        for i in range(0,len(self.segments)-1):
            if self.segments[i].t1 <= t0 and self.segments[i+1].t0 >= t1:
                self.segments.insert(i+1, segment)
                return
        l = ''
        for s in self.segments:
            l += '\n'+str(s.t0)+'-->'+str(s.t1)
        raise ValueError('impossible to add the segment t0='+str(segment.t0)+' t1='+str(segment.t1)+' to already existing segments'+l)
    
    def add(self,segments,autoJoin=True):
        """ add a segment or a list of segment to the segments """
        if type(segments) is not list:
            self.insert(segments,autoJoin)
        else:
            for s in segments:
                self.insert(s,autoJoin)
        self.update()
        self.label = 'Segments starts='+str(self.t0)+' ends='+str(self.t1)
        
    def start(self):
        if self.segments != []:
            return self.segments[0].start()
        else:
            return (0,0,0,0)
        
    def end(self):
        if self.segments != []:
            return self.segments[-1].end()
        else:
            return (0,0,0,0)
                
    def __call__(self,t):
        for s in self.segments:
            if t>=s.t0 and t<s.t1:
                return s(t)
        return (0,0,0,0)  #oversimplified: assuming PVAJ; should check that all segments are of the same nature
    

            
class SegmentPoly(Segment):
    """ a segment defined by a polynomial position law
    """
    def __init__(self,t0,t1,p):
        p=polynomial.Polynomial(p)
        v=p.derivative()
        a=v.derivative()
        j=a.derivative()
        super(SegmentPoly, self).__init__(t0,t1,(p,v,a,j))
        
    def _latex(self):
        """:return: string LaTex formula"""
        return 'pos(t)=%s'%self.funcs[0]._latex(x='t')
    
    def _repr_latex_(self):
        return '$%s$'%self._latex()
        
class Actuator():
    """ simulate an actuator. each movements are recorded in a Segments object
        the goal of this class is to simplify the writing in most common cases
      
    """
    
    def __init__(self,stateMachine,vmax,acc,name='',pos=V(0,'m'),distPerTurn=V(1,'mm'),angle=V(0,'deg'),mass=V(1,'kg'),friction=V(0,'N')):
        """
        :params stateMachine: a stateMachine. the only requirement for the simulation is to have a .time as V(time,'s') and a .displayMove boolean
        :params acc: the default acceleration of the actuator
        :params vmax: the default vmax
        :params name: name of the actuator
        :params pos: the initial position
        :params distPerTurn: the distance of the actuator per motor (or reductor) turn
        :params angle: if 0, the mass is moving horizontally, if 90° vertically: CAREFULL in that case the bigger the position, the higher
        :params mass: the mass to move (for both intertia and lifting force)
        :params friction: the friction force TODO: currently no difference between u0 and udynamique
        intertia of the pulley should be taken into consideration
        
        WARNING: the maxForce simulation is minimalist: as we know nothing about the reversibility of the grears and where is the friction, friction is always added
        even if it might be compensated by the mass in the case we go down
        """
        self.segs = Segments([])
        self.log = [] #list of (startTime,startPos,endTime,endPos) one tuple per move
        self.name = name
        self.stateMachine = stateMachine
        self.vmax = vmax
        self.acc = acc
        self.pos = pos
        self.distPerTurn = distPerTurn
        self.angle = angle
        self.mass = mass
        self.friction = friction.to('N')        
        self._maxAbsAcc = V(0,'m/s^2')
        self._maxAbsSpeed = V(0,'m/s')
        self._maxForce = V(0,'N')
        

        
    def move(self,newpos,relative=False,time = None, wait=True, vmax=None,acc=None):
        """ moves the actuator to newpos
        :params newpos: the new absolute position
        :params time: the starting time of the move. by default (None) the state machine time will be used but
                      one can force the starting poing in the past typically to do parallel moves of 
                      different actuators
        :params vmax: by default the values given at initialisation, but a value for this move can be given
        :params acc:  by default the values given at initialisation, but a value for this move can be given
        """
        
        if time is None:
            time = self.stateMachine.time
            
        if time < self.endTime():
            self.stateMachine.hwarning(self.name,'received at',time,'an order to move while it was already moving. had to wait',self.endTime())
            time = self.endTime()        

        if relative:
            newpos = self.pos + newpos
        
        if newpos == self.pos:
            if self.stateMachine.displayMove:
                from IPython.display import display,HTML
                display(HTML('<h4>{0}</h4> already in place @ {1}[m]'.format(self.name,newpos)))            
            return self.segs
        elif newpos > self.pos:
            acc = self.acc if acc is None else acc
            vmax = self.vmax if vmax is None else vmax
        else:
            acc = - self.acc if acc is None else -acc
            vmax= - self.vmax if vmax is None else -vmax
        self._maxAbsAcc = max(abs(acc),self._maxAbsAcc)
        s = sin(self.angle)
        fg = (self.mass*s*V(1,'gravity')).to('N')
        fa = (self.mass*acc).to('N')
        fm = abs(fg+fa)
        ft = fm+self.friction
        self._maxForce = max(self._maxForce, ft)                        

        m = SegmentsTrapezoidalSpeed(time('s'), self.pos('m'), newpos('m'),  a=acc('m/s^2'), vmax=vmax('m/s'))
        self._maxAbsSpeed = max(self._maxAbsSpeed,V(abs(m.segments[0].endSpeed()),'m/s'))
        self.lastmove = m
        self.log.append((time,self.pos,V(m.endTime(),'s'),newpos))
        
        self.stateMachine.simulation.displayPlot(self.name,m)

        self.pos = newpos
        self.segs.add(m)        
        self.stateMachine.time = max(V(m.endTime(),'s'),self.stateMachine.time)
        return m
    
    def endTime(self):
        return V(self.segs.endTime(),'s')
    
    def P(self,t):
        if isinstance(t, V):
            t = t('s')
        _p = self.segs(t)[0]
        return V(_p,'m')
    
    def maxAbsAcc(self):
        return self._maxAbsAcc
    
    def maxAbsSpeed(self):
        return self._maxAbsSpeed
    
    def maxForce(self):
        return self._maxForce
    
    def maxTork(self):
        return (self._maxForce*self.distPerTurn/V(360,'deg')).to('N m')
    
    def maxRpm(self):
        return self.maxAbsSpeed()/self.distPerTurn
    
    def displayLast(self):
        from IPython.display import display
        display(self.lastmove.svg())
        
    def display(self,fromTime=None,toTime=None):
        from IPython.display import display,HTML
        display(HTML('<h4>{0}</h4>'.format(self.name)))
        if fromTime is not None:
            fromTime = fromTime('s')
        if toTime is not None:
            toTime =toTime('s')
        self.segs.ticks = self.stateMachine.log
        display(self.segs.svg(xlim=(fromTime,toTime)))
        table = Table(self.name,[],self.varNames())
        table.appendCol('values',self.varDict())
        v = View(table,rowUnits=self.varRowUnits())
        display(v)
        timeTable = Table('timing',[self.stateMachine.name+' cycle',('start time','s'),('start pos','mm'),('stop time','s'),('stop pos','mm')],[])
        for i,l in enumerate(self.log):
            timeTable.appendRow('move {:0>4d}'.format(i), [self.stateMachine(l[0]('s')),l[0],l[1],l[2],l[3]])
        display(timeTable)
        
    def varNames(self):
        """ returns a list of internal variables. intended to be used in the Table.__init__ """
        return [self.name+'.mass',
                self.name+'.friction',
                self.name+'.dist/turn', 
                self.name+'.maxSpeed',
                self.name+'.maxAcc',               
                self.name+'.maxForce',
                self.name+'.maxRpm',
                self.name+'.maxTork',
                ]
    
    def varRowUnits(self):
        return {self.name+'.mass'     : 'kg',
                self.name+'.friction' : 'N',
                self.name+'.dist/turn': 'mm',
                self.name+'.maxSpeed' : 'm/s', 
                self.name+'.maxAcc'   : 'm/s^2',               
                self.name+'.maxForce' : 'N',
                self.name+'.maxRpm'   : 'rpm',
                self.name+'.maxTork'  : 'N m',
                }
            
    def varDict(self):
        return {self.name+'.mass'    : self.mass,
                self.name+'.friction': self.friction,
                self.name+'.dist/turn': self.distPerTurn,
                self.name+'.maxSpeed' : self.maxAbsSpeed(),
                self.name+'.maxAcc'   : self.maxAbsAcc(),               
                self.name+'.maxForce' : self.mass*self.maxAbsAcc()+self.friction,
                self.name+'.maxRpm'  : self.maxRpm(),
                self.name+'.maxTork' : self.maxTork(),
                }
    
class TimeDiagram(plot.Plot):        
    def __init__(self,actuators,stateMachines=[],fromTime=None,toTime=None):
        """
        :params stateMachines: [(stateMachine,pos,posShift),...]
        """
        self.actuators = actuators
        self.t0 = min(a.segs.startTime() for a in actuators) if fromTime is None else fromTime('s')
        self.t1 = max(a.segs.endTime() for a in actuators) if toTime is None else toTime('s')
        self.stateMachines = stateMachines
        
    def __repr__(self):
        return 'Time Diagram from {0:f}[s] to {1:f}[s]'.format(self.t0,self.t1)
        
    def _plot(self, ax, **kwargs):
        """
        plots a list of actuators
        """
        linewidth = kwargs.pop('linewidth',0)
        (t0,t1) = kwargs.setdefault('xlim',(self.t0,self.t1))
        (ymin,ymax) = (-1000,1000)
        from matplotlib.font_manager import FontProperties
        fontP = FontProperties()
        fontP.set_size('xx-small')

        step=(t1-t0)/500.
        x=[ t for t in itertools2.arange(self.t0,self.t1,step)  ]
        for a in self.actuators:
            y = [a.segs(t)[0] for t in x]
            ax.plot(x, y, label=a.name, linewidth=linewidth)
        
        for s,pos,posShift in self.stateMachines:
            shift = False
            for (t,event) in s.log:
                if isinstance(event, StateChangeLog):
                    if t >= t0 and t <= t1:
                        y = posShift if shift else pos
                        ax.text(t, y, '<'+str(event.newState), fontsize=(6),
                                horizontalalignment='left',
                                verticalalignment='center')
                        shift = not shift
                    ax.text(t1,pos,s.name)
                elif isinstance(event,TooLateLog):
                    tend = event.pastTime('s')
                    if (t<=t1 and t>=t0) or (tend <= t1 and tend >= t1):
                        ax.broken_barh([(tend,t-tend)],(ymin,ymax-ymin),facecolor='white',edgecolor='red')
                        ax.broken_barh([(tend,t-tend)],(pos,posShift-pos),facecolor='red',edgecolor='')
                        
                elif isinstance(event,WaitLog):
                    tend = event.untilTime('s')
                    if (t<=t1 and t>=t0) or (tend <= t1 and tend >= t1):
                        ax.broken_barh([(tend,t-tend)],(ymin,ymax-ymin),facecolor='white',edgecolor='green')
                        ax.broken_barh([(tend,t-tend)],(pos,posShift-pos),facecolor='green')
                    
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop = fontP)
        return ax
    
    def saveAsCsv(self,filename):
        step=(self.t1-self.t0)/500.
        x=[ t for t in itertools2.arange(self.t0,self.t1,step)]
        data=[['time']+x]
        
        for sm,_p,_sp in self.stateMachines:
            data.append(["'"+sm.name]+[sm(t) for t in x]) 
        for a in self.actuators:
            data.append(["'"+a.name]+[(a.P(t))('m') for t in x]) 
        tab = Goulib.table.Table(data=data)    
        tab.write_csv(filename)

        
def _pva(val):
    try: p=val[0]
    except: p=val
    try: v=val[1]
    except: v=None
    try: a=val[2]
    except: a=None
    return p,v,a
           
def _delta(x0,x1):
    try:
        return float(x1-x0)
    except:
        return None
    
def ramp(dp,v0,v1,a):
    """
    :param dp: float delta position or None if unknown
    :param v0: float initial velocity or None if unknown
    :param v1: float final velocity or None if unknown
    :param a: float acceleration
    :return: float shortest time to accelerate between constraints
    """
    dt=[]
    dv=_delta(v0,v1)
    if dv:
        dt.append(dv/a) #time to accelerate
    try: # solve a.t^2/2+v0.t == dp
        dt.extend(list(math2.quad(a/2.,v0,-dp)))
    except: 
        try: # solve v1.t-a.t^2/2 == dp
            dt.extend(list(math2.quad(-a/2.,v1, -dp)))
        except: pass
    return min(t for t in dt if t > 0) #return smallest positive

def trapeze(dp,vmax,a,v0=0,v2=0):
    """
    :param dp: float delta position
    :param vmax: float maximal velocity
    :param a: float acceleration
    :param v0: float initial velocity, 0 by default 
    :param v2: float final velocity, 0 by default 
    :return: tuple of 6 values:
    
    * time at end of acceleration
    * position at end of acceleration
    * velocity at end of acceleration
    * time at begin of deceleration
    * position at begin of deceleration
    * total time
    """
    t1=ramp(dp/2.,v0,vmax,a) #acceleration time
    v1=v0+a*t1 #speed reached
    p1=t1*(v0+v1)/2. #position at end of acceleration
    t3=ramp(dp/2.,v1,v2,-a) #deceleration time
    p2=t3*(v1+v2)/2. #distance to decelerate
    t2=float(dp-p1-p2)/v1 #time at constant velocity
    return t1,p1,v1,t1+t2,dp-p2,t1+t2+t3
    
def Segment2ndDegree(t0,t1,start,end=(None)):
    """calculates a constant acceleration Segment between start and end
    
    :param t0,t1: float start,end time. one of both may be None for undefined
    :param start: (position, velocity, acceleration) float tuple. some values may be None for undefined
    :param end: (position, velocity, acceleration) float tuple. some values may be None for undefined
    :return: :class:`SegmentPoly`
    
    the function can cope with almost any combination of defined/undefined parameters,
    among others (see tests):
    
    * Segment2ndDegree(t0,t1,(p0,v0),p1) # time interval and start + end positions  + initial speed
    * Segment2ndDegree(t0,t1,(p0,v0,a)) # time interval and start with acceleration
    * Segment2ndDegree(t0,t1,None,(p1,v1,a)) # time interval and end pva
    * Segment2ndDegree(t0,None,(p0,v0),(p1,v1)) # start + end positions + velocities
    * Segment2ndDegree(t0,None,(p0,v0,a),(None,v1)) # start pva + end velocity
    * Segment2ndDegree(None,t1,p0,(p1,v1,a)) # end pva + start position
    
    the function also accepts some combinations of overconstraining parameters:
    
    * Segment2ndDegree(t0,t1,(p0,v0,a),p1) # time interval, start pva, end position => adjust t1
    * Segment2ndDegree(t0,t1,(p0,v0,a),(None,v1)) # time interval, start pva, v1=max vel => adjust t1
    
    :raise ValueError: when not enough parameters are specified to define the Segment univoquely
    
    """
    p0,v0,a0=_pva(start)
    p1,v1,a1=_pva(end)
    if a0 is None: a0=a1
    #to handle the many possible cases, we evaluate missing information in a loop
    for _retries in range(2): #two loops are enough to solve all cases , according to tests   
        dt=_delta(t0,t1)
        dp=_delta(p0,p1)
        dv=_delta(v0,v1)
        
        if not itertools2.any((dt,p0,v0,a0),lambda x:x is None): #we have all required data
            res=SegmentPoly(t0,t1,[p0,v0,a0/2.])
            end=res.end()
            if p1 is not None and not math2.isclose(end[0],p1): #consider p1 as max position
                res2=Segment2ndDegree(t0,None,(p0,v0,a0),p1)
                if res2.dt()<res.dt(): #this case arises earlier
                    res=res2
            if v1 is not None and not math2.isclose(end[1],v1): #consider v1 as max velocity
                res2=Segment2ndDegree(t0,None,(p0,v0,a0),(None,v1))
                if res2.dt()<res.dt(): #this case arises earlier
                    res=res2
            return res

        if dt is None: #try to determine it from available params
            if a0:
                dt=ramp(dp,v0,v1,a0)
            else:
                try:
                    dt=2.*dp/(v0+v1) #time to reach the position
                except: pass

        if t0 is None:
            try: t0=t1-dt
            except: pass
        if t1 is None: 
            try: t1=t0+dt
            except: pass
                
        if a0 is None:
            try: a0=float(dv)/dt
            except:
                try: a0=2.*(dp-v0*dt)/dt*dt
                except: pass
        if v0 is None:
            try: v0=v1-a0*dt
            except: pass
        if p0 is None:
            try: p0=p1-dt*float(v1+v0)/2.
            except: pass
    
    raise ValueError
    
def Segment4thDegree(t0,t1,start,end):
    """smooth trajectory from an initial position and initial speed (p0,v0) to a final position and speed (p1,v1)
    * if t1<=t0, t1 is calculated
    """
    p0,v0,_a0=_pva(start)
    p1,v1,_a1=_pva(end)
    
    if t1<=t0:
        dt=float(p1-p0)/((v1-v0)/2. + v0) #truediv
        t1=t0+dt
    else:
        dt=t1-t0
    return SegmentPoly(t0,t1,[p0,v0,0,float(v1-v0)/(dt*dt),-float(v1-v0)/(2*dt*dt*dt)]) #truediv


def SegmentsTrapezoidalSpeed(t0,p0,p3,a,T=0,vmax=float('inf'),v0=0,v3=0):
    """
    :param t0: float start time
    :param p0: float start position
    :param p3: float end position
    :param a: float specified acceleration. if =0, use specified time
    :param T: float specified time. if =0 (default), use specified acceleration
    :param vmax: float max speed. default is infinity (i.e. triangular speed)
    :param v0: initial speed
    :param v3: final speed    if T <> 0 then v3 = v0
       v1  +-------+    
          /         \
         /           +  v3
    v0  +
        |  |       | |
       t0  t1     t2 t3 
    """
    dp = p3-p0
    if T != 0:
        assert t0 == 0.0, 'must fix this bug'
        assert v3==v0,'if T is the constraint, v0 must equal v3'
        t1 = T/2
        v1 = dp/t1 -v0 # (v0+v1)/2 *t1 = dp/2  ==> v0*t1 + v1*t1 =dp
        if v1 > vmax:
            RuntimeError('vmax not yet implemented: must be infinite')
        else:
            a = (v1-v0)/t1
    (t1,p1,v1,t2,p2,t3) = trapeze(dp,vmax,a,v0,v3)
    label = 'Trapeze start={0}, end={1}'.format(t0,t0+t3) 
    acc = Segment2ndDegree(t0,t1+t0,(p0,v0,a))
    cst = SegmentPoly(t1+t0,t2+t0,[p1+p0,v1])
    dec = Segment2ndDegree(t2+t0,t3+t0,(p2+p0,v1,-a))
    trap = Segments([acc,cst,dec],label=label)
    return trap
            

