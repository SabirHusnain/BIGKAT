# -*- coding: utf-8 -*-
"""

"""
from __future__ import division, print_function
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import pickle
import seaborn as sns

def create_wand(length = 130):    
    
    wand = np.empty((3,3))
    
    wand[1] = [0,0,0]
    wand[0] = [wand[1,0] - length/2, 0, 0]
    wand[2] = [wand[1,0] + length/2, 0, 0]
    
    
    return wand
    
def translation(p, T):
    
    T = np.array(T)
    
    return p + T
    
def rodrig(v, k):
    """Rotate vector v around vector k by a magnitude equal to the lenght of k"""
    
    theta = np.sqrt(np.dot(k,k))
    
    return v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * np.dot(k,v) * (1-np.cos(theta))   
    
        
wand = create_wand() 
wand = wand + np.array([0,0, 25])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(wand[:,0], wand[:,1], wand[:,2], s = 50, depthshade = False)
ax.plot(wand[:,0], wand[:,1], wand[:,2], color = 'k')
ax.view_init(elev=94., azim=-90)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim([-20, 500])
ax.set_ylim([-20, 500])
ax.set_zlim([-20, 500])
plt.gca().invert_yaxis()
plt.gca().invert_zaxis()
plt.title("Standard Wand")


##Move the wand
T = np.array([10, 5, 50])

wand_pos = wand
T_all = []
for i in range(25):
    
    wand_pos = wand_pos + T
    T_all.append(wand_pos)
    
T_all = np.array(T_all)

T_all_noise = T_all + np.random.normal(0, 10, size = T_all.shape)

dt = 1 /60
dt2 = dt

def time_stamp(dt, length = T_all.shape[0]):
    
    ts = np.empty(length)

    for i in range(length-1):
        ts[i+1] = ts[i] + dt
    
    return ts

ts = time_stamp(dt)




#pos3d = pos3d  

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(T_all[:,0,0],T_all[:,0,1], T_all[:,0,2], s =ts*5, color = 'r', depthshade = False)
ax.scatter(T_all[:,1,0],T_all[:,1,1], T_all[:,1,2], s = ts*5, color = 'b', depthshade = False)
ax.scatter(T_all[:,2,0],T_all[:,2,1], T_all[:,2,2], s = ts*5, color = 'r', depthshade = False)
ax.view_init(elev=94., azim=-90)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
#ax.set_xlim([-20, 500])
#ax.set_ylim([-20, 500])
#ax.set_zlim([-20, 500])
plt.gca().invert_yaxis()
plt.gca().invert_zaxis()
plt.title("Standard Wand")


#plt.close()
#
##A = np.array(   [[1, 0, 0, dt, 0, 0, 0.5 * dt**2, 0, 0],
##                [0, 1, 0, 0, dt, 0, 0, 0.5 * dt**2, 0],
##                [0, 0, 0, 0, 0, dt, 0, 0, 0.5 * dt**2],
##                [0, 0, 0, 1, 0, 0, dt, 0, 0],
##                [0, 0, 0, 0, 1, 0, 0, dt, 0],
##                [0, 0, 0, 0, 0, 1, 0, 0, dt],
##                [0, 0, 0, 0, 0, 0, 1, 0, 0],
##                [0, 0, 0, 0, 0, 0, 0, 1, 0],
##                [0, 0, 0, 0, 0, 0, 0, 0, 1] ])
#                
A = np.array([[1, 0, 0, dt, 0, 0 ],
              [0, 1, 0, 0, dt, 0],
              [0, 0, 1, 0, 0, dt],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])
              
A = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])
                
import pystan

code = """

data{

    int N; //Number of data points   

    vector[3] Y[N]; //Data
    
    real dt;
          
}

parameters{
   
    vector[K] X[N];
    
    real<lower = 0, upper = 50> Sigma;           
       
}




model{
    
    //Initial Prior
    X[1] ~ normal(500, 5000);
    
    //State model
    for (i in 2:N){
        X[i] ~ normal(X[i-1], 25);    
    }    
       
    //Observation model
    Y[1:N, 1] ~ normal(X[1:N, 1], Sigma);
    Y[1:N, 2] ~ normal(X[1:N, 2], Sigma);
    Y[1:N, 3] ~ normal(X[1:N, 3], Sigma);

}

"""


sm = pickle.load(open('osc_Kalman.pkl', 'rb'))
sm = pystan.StanModel(model_code=code) #Compile Stan Model
with open('osc_Kalman.pkl', 'wb') as f: #Save Stan model to file
    pickle.dump(sm, f)


stan_data = {'Y': T_all_noise[:,1], 'N': T_all.shape[0]}

fit = sm.sampling(data = stan_data, chains = 4, iter = 5000, refresh = 10, control = {'adapt_delta': 0.99})

sampler_params = fit.get_sampler_params(inc_warmup=False) 
n_divergent = reduce(lambda x, y: x + y['divergent__'].tolist(), sampler_params, []) 

if 1 in n_divergent: 
    print("N divergent = {}".format(sum(n_divergent)))   
#op = sm.optimizing(data = stan_data)

#vb_ = sm.vb(data = stan_data,iter=10000,  init='random',)
#
#
#with open('print_fit.txt', 'w') as f: #Save fit print output to file
#    print(fit, file=f)
#    
la = fit.extract()
