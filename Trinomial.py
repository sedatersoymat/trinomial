# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 09:18:11 2022

Recombining Trinomial model with changing volatility

@author: Sedat Ersoy
"""

import math
import numpy as np
from scipy import interpolate

R0 = 0.025
rf = 0
T = 1 #in years
K = 0.026
is_european = True
is_call = True
num_in_year = 365
dt = 1/num_in_year             #given contract exercise dates, calculate time slices. maybe no need as we can do daily basis
total_slices = T*num_in_year
lambda_ = 1.12  # Haahtela(2010)
N = total_slices

Sigma_nodes = np.ones(total_slices)*0.10  # obtain vol for each time slices from vol surface.  
sigma = np.max(Sigma_nodes)
u = math.exp(rf*dt + math.sqrt(math.exp(lambda_**2*sigma**2*dt) - 1))
d = math.exp(rf*dt - math.sqrt(math.exp(lambda_**2*sigma**2*dt) - 1))
m = math.exp(rf*dt)


#initialize probability nodes
qu = m**2*(np.exp(Sigma_nodes**2*dt)-1)/(u**2+m*d-u*m-u*d)
qd = qu*(m-u)/(d-m)
qm = 1-qu-qd



#initialize security price tree
RTs = [np.array([R0])]

for i in range(N):
    prev_nodes =RTs[-1]
    RT = np.concatenate(
        (prev_nodes*u, [prev_nodes[-1]*m, prev_nodes[-1]*d]))
    RTs.append(RT)
    
#Intrinsic value of nodes
if is_call:
    temp_arr = np.subtract(RTs,K)
else:
    temp_arr = np.subtract(K, RTs)
            
intr_value_node = np.array( [np.maximum(values_node,0) for values_node in temp_arr] )


""" Traverse the tree backwards """
payoffs = intr_value_node[-1]
for i in reversed(range(N)):
    df_i = np.exp(np.multiply(RTs[i], -dt))   #discount rate for each scenario at node i+1.   
    #df_i = math.exp(-rf*dt)
    payoffs = (payoffs[:-2] * qu[i] + payoffs[1:-1] * qm[i] + payoffs[2:] * qd[i]) * df_i

    if not is_european:
        payoffs = np.maximum(intr_value_node[i], payoffs) 





x = np.arange(-5.01, 5.01, 0.25)
y = np.arange(-5.01, 5.01, 0.25)
xx, yy = np.meshgrid(x, y)
z = np.add(xx, yy)
f = interpolate.interp2d(x, y, z, kind='linear')

neww = f(-7,2)