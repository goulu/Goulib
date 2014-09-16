#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
optimization algorithms : knapsack, travelling salesman, simulated annealing
"""
__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2012, Philippe Guglielmetti"
__credits__ = ["http://www.psychicorigami.com/category/tsp/",]
__license__ = "LGPL"

import random
import math
import logging

import Goulib.itertools2 as itertools2

class _Bin():
    def __init__(self, capacity, f=lambda x:x):
        """a container with a limited capacity
        :param capacity: int,flot,tuple of whatever defines the capacity of the Bin
        :param f: function f(x) returning the capacity used by item x. Must return the empty capacity when f(0) is called
        """
        self._capacity=capacity
        self._f=f #functions that return the cached values of an item
        self._used=f(0)

    def __repr__(self):
        return '%s(%s/%s)'%(self.__class__.__name__,self._used,self._capacity)

    def _add(self,value,tot=None):
        """update sum when item is added"""
        if not tot : tot=self._used
        if isinstance(value,(int,float)):
            return tot+value
        if isinstance(value,set):
            return tot|value
        return list(map(self._add,list(zip(value,tot))))

    def _recalc(self):
        self._used=self._f(0)
        for item in self:
            self._used=self._add(self._f(item))

    def _sub(self,value,tot=None):
        """update sum AFTER item is removed"""
        if not tot : tot=self._used
        if isinstance(value,(int,float)):
            return tot-value
        if isinstance(value,set):
            if isinstance(tot,(int,float)):
                return tot-len(value)
            else:
                try:
                    i=self._used(tot) #index of set
                except:
                    i=None
                self._recalc()
                return self._used[i] if i is not None else self._used

        return list(map(self._sub,list(zip(value,tot))))

    def size(self):
        return self._sub(self._used,self._capacity)

    def _fits(self,value,cap=None):
        """compare value to capacity"""
        if cap is None : cap=self._capacity
        if isinstance(value,(int,float)):
            return value<=cap
        if isinstance(value,set):
            return len(value)<=cap
        return all(map(self._fits,list(zip(value,cap))))

    def fits(self,item):
        """:return: bool True if item fits in bin without exceeding capacity"""
        return self._fits(self._add(self._f(item)))

    def __iadd__(self,item):
        """addition of an element : MUST BE OVERLOADED by subclasses"""
        if not self.fits(item):
            raise OverflowError
        self._used=self._add(self._f(item))
        return self

    def __isub__(self,item):
        """removal of an element : MUST BE OVERLOADED by subclasses and called AFTER item is removed"""
        self._used=self._sub(self._f(item))
        return self

class BinDict(_Bin,dict):
    #http://docs.python.org/2/reference/datamodel.html#emulating-container-types

    def __isub__(self,key):
        item=self[key]
        super(BinDict,self).__delitem__(key) #must be first
        super(BinDict,self).__isub__(item) #must be next
        return self

    __delitem__=__isub__

    def __iadd__(self,key,item):
        if key in self:
            del self[key]
        super(BinDict,self).__iadd__(item) #raises OverflowError if full
        super(BinDict,self).__setitem__(key,item)
        return self

    __setitem__=__iadd__


class BinList(_Bin,list):
    #http://docs.python.org/2/reference/datamodel.html#emulating-container-types

    def __iadd__(self,item):
        super(BinList,self).__iadd__(item) #raises OverflowError if full
        super(BinList,self).append(item)
        return self

    append = __iadd__

    def insert(self,i,item):
        super(BinList,self).__iadd__(item) #raises OverflowError if full
        super(BinList,self).insert(i,item)
        return self

    def extend(self,more):
        for x in more:
            self.append(x)
        return self

    def __isub__(self,item):
        super(BinList,self).remove(item)
        super(BinList,self).__isub__(item)
        return self

    remove = __isub__

    def pop(self,i=-1):
        item=super(BinList,self).pop(i)
        super(BinList,self).__isub__(item)
        return item

    def __setitem__(self,i,item):
        """called when replacing a value in list"""
        old=self[i]
        super(BinList,self).__isub__(old)
        super(BinList,self).__iadd__(item) #raises OverflowError if full
        super(BinList,self).__setitem__(i,item)

#Bin packing algorithms
#see https://en.wikipedia.org/wiki/Bin_packing_problem

def first_fit_decreasing(items, bins, maxbins=0):
    """ fit items in bins using the "first fit decreasing" method
    :param items: iterable of items
    :param bins: iterable of Bin s. Must have at least one Bin
    :return: list of items that didn't fit. (bins are filled by side-effect)
    """

    nofit=[]
    items.sort(key=lambda x:bins[0]._f(x),reverse=True)
    for item in items:
        fit=False
        for b in bins:
            try:
                b+=item
                fit=True
                break
            except OverflowError:
                continue #next bin ?
        if not fit: # may we add a bin ?
            if len(bins)<maxbins: #yes
                b=type(bins[0])(bins[0]._capacity,bins[0]._f)
                try:
                    b+=item
                    bins.append(b)
                    continue #next item
                except OverflowError: #item too large for a bin
                    pass
            nofit.append(item)
    return nofit

def hillclimb(init_function,move_operator,objective_function,max_evaluations):
    '''
    hillclimb until either max_evaluations is reached or we are at a local optima
    '''
    best=init_function()
    best_score=objective_function(best)

    num_evaluations=1

    logging.info('hillclimb started: score=%f',best_score)

    while num_evaluations < max_evaluations:
        # examine moves around our current position
        move_made=False
        for m in move_operator(best):
            if num_evaluations >= max_evaluations:
                break

            # see if this move is better than the current
            next_score=objective_function(m)
            num_evaluations+=1
            if next_score > best_score:
                best=m
                best_score=next_score
                move_made=True
                break # depth first search

        if not move_made:
            break # we couldn't find a better move (must be at a local maximum)

    logging.info('hillclimb finished: num_evaluations=%d, best_score=%f',num_evaluations,best_score)
    return (num_evaluations,best_score,best)

def hillclimb_and_restart(init_function,move_operator,objective_function,max_evaluations):
    '''
    repeatedly hillclimb until max_evaluations is reached
    '''
    best=None
    best_score=0

    num_evaluations=0
    while num_evaluations < max_evaluations:
        remaining_evaluations=max_evaluations-num_evaluations

        logging.info('(re)starting hillclimb %d/%d remaining',remaining_evaluations,max_evaluations)
        evaluated,score,found=hillclimb(init_function,move_operator,objective_function,remaining_evaluations)

        num_evaluations+=evaluated
        if score > best_score or best is None:
            best_score=score
            best=found

    return (num_evaluations,best_score,best)

def P(prev_score,next_score,temperature):
    if next_score > prev_score:
        return 1.0
    else:
        return math.exp( -abs(next_score-prev_score)/temperature )

class ObjectiveFunction:
    '''class to wrap an objective function and
    keep track of the best solution evaluated'''
    def __init__(self,objective_function):
        self.objective_function=objective_function
        self.best=None
        self.best_score=None

    def __call__(self,solution):
        score=self.objective_function(solution)
        if self.best is None or score > self.best_score:
            self.best_score=score
            self.best=solution
            logging.info('new best score: %f',self.best_score)
        return score

def kirkpatrick_cooling(start_temp,alpha):
    T=start_temp
    while True:
        yield T
        T=alpha*T

def anneal(init_function,move_operator,objective_function,max_evaluations,start_temp,alpha):

    # wrap the objective function (so we record the best)
    objective_function=ObjectiveFunction(objective_function)

    current=init_function()
    current_score=objective_function(current)
    num_evaluations=1

    cooling_schedule=kirkpatrick_cooling(start_temp,alpha)

    logging.info('anneal started: score=%f',current_score)

    for temperature in cooling_schedule:
        done = False
        # examine moves around our current position
        for next in move_operator(current):
            if num_evaluations >= max_evaluations:
                done=True
                break

            next_score=objective_function(next)
            num_evaluations+=1

            # probablistically accept this solution
            # always accepting better solutions
            p=P(current_score,next_score,temperature)
            if random.random() < p:
                current=next
                current_score=next_score
                break
        # see if completely finished
        if done: break

    best_score=objective_function.best_score
    best=objective_function.best
    logging.info('final temperature: %f',temperature)
    logging.info('anneal finished: num_evaluations=%d, best_score=%f',num_evaluations,best_score)
    return (num_evaluations,best_score,best)


def reversed_sections(tour):
    '''generator to return all possible variations where the section between two cities are swapped'''
    for i,j in itertools2.all_pairs(len(tour)):
        if i and j and i < j:
            copy=tour[:]
            copy[i:j+1]=reversed(tour[i:j+1])
            yield copy

def swapped_cities(tour):
    '''generator to create all possible variations where two cities have been swapped'''
    for i,j in itertools2.all_pairs(len(tour)):
        if i < j:
            copy=tour[:]
            copy[i],copy[j]=tour[j],tour[i]
            yield copy

def tour_length(points,dist,tour=None):
    """generator of point-to-point distances along a tour"""
    if not tour:tour=list(range(len(points))) #will generate the closed tour length
    n=len(tour)
    for i in range(n):
        j=(i+1)%n
        yield dist(points[tour[i]],points[tour[j]])

def tsp(points,dist,max_iterations=100,start_temp=None,alpha=None,close=True,rand=True):
    """Travelling Salesman Problem
    @see http://en.wikipedia.org/wiki/Travelling_salesman_problem
    @param points : iterable containing all points
    @param dist : function returning the distance between 2 points : def dist(a,b):
    @param max_iterations :max number of optimization steps
    @param start_temp, alpha : params for the simulated annealing algorithm. if None, hill climbing is used
    @param close : computes closed TSP. if False, open TSP starting at points[0]
    @return iterations,score,best : number of iterations used, minimal length found, best path as list of indexes of points
    """
    n=len(points)
    def init_function():
        tour=list(range(1,n))
        if rand:
            random.shuffle(tour)
        return [0]+tour
    def objective_function(tour):
        """total up the total length of the tour based on the dist ance function"""
        return -sum(tour_length(points,dist,tour if close else tour[:-1]))
    if start_temp is None or alpha is None:
        iterations,score,best=hillclimb_and_restart(init_function,reversed_sections,objective_function,max_iterations)
    else:
        iterations,score,best=anneal(init_function,reversed_sections,objective_function,max_iterations,start_temp,alpha)
    return iterations,score,best
