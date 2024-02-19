#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:25:54 2024

@author: jeremy chen
"""

#for the first sub-task: supervised setting
#plot the figure of squared loss, logistic loss, hinge loss, and 0/1 error.

import numpy as np
import math
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
x = np.linspace(start=-2, stop=3,num =1001,dtype=np.float)
logi = np.log(1 + np.exp(-x))/math.log(2)
y_se = (x**2)

y_01 = x < 0
y_hinge = 1.0 - x
y_hinge[y_hinge < 0] = 0
plt.plot(x, logi, 'r-.', mec='k', label='Logistic Loss', lw=1)
plt.plot(x, y_01, 'g-.', mec='k', label='0/1 Loss', lw=1)
plt.plot(x, y_hinge, 'k-.',mec='k', label='Hinge Loss', lw=1)
plt.plot(x, y_se, 'b-.',mec='k', label='Squared Loss', lw=1)

plt.grid(True, ls='--')
plt.legend(loc='upper right')
plt.title('Loss function')
#plt.savefig('Task4_loss function', dpi=800)
plt.show()