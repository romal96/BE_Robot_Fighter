#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:37:26 2020

@author: c.ponzoni
"""

import argparse
import os
import json
import multiprocessing
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb

def alphavecplot(alphavec):
    x = np.linspace(0, 1, 50)
    colors = {'listen': 'red', 'open-right': 'blue', 'open-left':'green'}
    for i in range(len(alphavec)):
        plt.plot(x, (alphavec[i]['v'][0]-alphavec[i]['v'][1])*x + alphavec[i]['v'][1], color=colors[alphavec[i]['action']])

def beliefplot(beliefs):
    for b in range(len(beliefs)):
        plt.axvline(beliefs[b][0], color='gray')


policy_file = "alphavecfile.policy"
with open(policy_file) as f:
    data = json.load(f)

sb.set_style("white")
alphavecplot(data['alphavec'])
beliefplot(data['beliefs'])
sb.despine()

#plt.show()
plt.text(1.08, -95, "b($s_0$)", ha='left')
plt.text(0.25, -45, "open-right", ha='left', color='blue', rotation=32)
plt.text(0.62, -46, "open-left", ha='left', color='green', rotation=-34)
plt.text(0.35, 24, "listen", ha='left', color='red')
plt.ylabel("V(b)")

plt.savefig('tiger_value_function.pdf')
plt.savefig('tiger_value_function.png')