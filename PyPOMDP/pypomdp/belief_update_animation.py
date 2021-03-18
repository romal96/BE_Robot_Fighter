#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 21:36:32 2020

@author: c.ponzoni
"""

# Import packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
#from matplotlib.widgets import Slider
from random import randrange, uniform
import time
# Change matplotlib backend
# %matplotlib notebook

# Create variable reference to plot
class AnimateBeliefPlot:
    def __init__(self, initial_belief, pomdp_act, exp_act):
        mpl.rcParams['font.size'] = 10
        mpl.rcParams['axes.linewidth'] = 2
        mpl.rcParams['axes.spines.top'] = False
        mpl.rcParams['axes.spines.right'] = False
        mpl.rcParams['xtick.major.size'] = 8
        mpl.rcParams['xtick.major.width'] = 2
        mpl.rcParams['ytick.major.size'] = 8
        mpl.rcParams['ytick.major.width'] = 2
        # Create the figure
        self.fig = plt.figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111)
        # Create variable reference to plot
        #self.f_d, = self.ax.plot([], [], linewidth=2.5) # Add text annotation and create variable reference
        #self.temp = self.ax.text(1, 1, '', ha='right', va='top', fontsize=24)
        # Get colors from coolwarm colormap
        self.colors = ['red', 'green', 'blue', 'orange']
        # Ensure the entire plot is visible
        self.fig.tight_layout()
        
        # Data
        self.beliefs = np.array([initial_belief])
        self.states_names = ['not engaged', 'engaged']
        #print(self.beliefs)
        #self.belief.append(initial_belief)
        self.states = [i for i in range(len(initial_belief))]
        #print(self.states)
        self.time_steps = [t for t in range(len(self.beliefs))]
        self.bars=[]
        floorbar = np.zeros(len(self.states))
        for state in self.states:
            #print(self.time_steps, self.beliefs[:,state])
            bar = plt.bar(self.time_steps, self.beliefs[:,state], bottom=floorbar, color=self.colors[state], edgecolor='white', width=1)
            self.bars.append(bar)
            floorbar = floorbar + self.beliefs[:,state]
        
        self.plt = plt
        self.plt.ion()
        self.plt.ylim(0,1.6)
        self.plt.xlim(-1,60)
        self.plt.title("Belief state progression")
        self.plt.text(40, 1.12, "POMDP action: "+pomdp_act, size=8, rotation=0., ha="center", va="center", 
                      bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
        self.plt.text(40, 1.22, "EXP action: "+exp_act, size=8, rotation=0., ha="center", va="center", 
                      bbox=dict(boxstyle="round", ec=(0., 0.5, 0.5), fc=(0., 0.8, 0.8)))
        #self.plt.text(40, 1.3, "Observation : oengaged", size=8, rotation=0., ha="center", va="center", 
        #              bbox=dict(boxstyle="round", ec=(0.5, 1., 0.5), fc=(0.8, 1.0, 0.8)))
        #self.plt.legend(self.states_names)
        self.plt.legend(self.states_names,loc='upper center', ncol=2, fancybox=True, shadow=True)
        self.plt.draw()
        self.plt.show()
        self.plt.pause(1)
        self.plt.clf()
        self.p_pomdp_act = pomdp_act
        self.p_exp_act = exp_act
        
    def update(self, newbel, pomdp_act, exp_act, obs):
        #np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
        self.beliefs = np.append(self.beliefs, [newbel], axis=0)
        #self.time_steps = [t for t in range(len(self.beliefs))]
        self.time_steps = np.arange(len(self.beliefs))
        #print(self.time_steps)
        #plt.clf()
        #plt.cla()
        for bar in self.ax.containers:
            bar.remove()
        
        floorbar = np.zeros(len(self.beliefs))
       
        for state in self.states:
            #print(self.time_steps, self.beliefs[:,state])
            self.plt.bar(self.time_steps, self.beliefs[:,state], bottom=floorbar, color=self.colors[state], edgecolor='white', width=1)
            #floorbar = floorbar + self.beliefs[:,state]
            floorbar = np.add(floorbar,self.beliefs[:,state])
            #print(floorbar)
        
        self.plt.ylim(0,1.6)
        self.plt.xlim(-1,60)
        self.plt.title("Belief state progression")
        self.plt.text(40, 1.12, "POMDP action: "+pomdp_act, size=8, rotation=0., ha="center", va="center", 
                      bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
        self.plt.text(40, 1.22, "EXP action: "+exp_act, size=8, rotation=0., ha="center", va="center", 
                      bbox=dict(boxstyle="round", ec=(0., 0.5, 0.5), fc=(0., 0.8, 0.8)))
        self.plt.text(40, 1.32, "Observation : "+obs, size=8, rotation=0., ha="center", va="center", 
                      bbox=dict(boxstyle="round", ec=(0.5, 1., 0.5), fc=(0.8, 1.0, 0.8)))
        
        self.plt.legend(self.states_names, loc='upper center', ncol=2, fancybox=True, shadow=True)
        if self.p_exp_act != exp_act :
            self.plt.vlines(self.time_steps[-2],0,1.0, color="black")
            self.plt.text(self.time_steps[-2], 1.025, "mode changed during experiment", size=8, color="black")
            
        self.plt.draw()
        self.plt.show()
        self.plt.pause(1)
        self.plt.clf()
        self.p_pomdp_act = pomdp_act
        self.p_exp_act = exp_act
        
    
    def destroy(self):
        self.plt.close()


if __name__ == '__main__':
    initial_belief = [0.5, 0.5]
    plotting = AnimateBeliefPlot(initial_belief)
    for t in range(10):
        frand = uniform(0, 1)
        new = [frand, 1.0-frand]
        plotting.update(new)
        

 