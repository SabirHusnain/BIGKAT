# -*- coding: utf-8 -*-
"""

"""
from __future__ import division, print_function
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import pickle
import seaborn as sns
import time

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

dt = 1.0 / 60
dt2 = dt
T = np.array([500, 700, 1300])

wand_pos = wand
T_all = []
for i in range(60*12):
    
    wand_pos = wand_pos + T * dt
    T_all.append(wand_pos)
    
T_all = np.array(T_all)

T_all_noise = T_all + np.random.normal(0, 2, size = T_all.shape)

dt = 1 / 60
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
    vector[3] Y1[N]; //Data left marker
    vector[3] Y2[N]; //Data right marker
    real<lower = 0> dt;
    real sig_V;
    real sig_P;    
    real D;
    
          
}

transformed data{

    vector[3] M;
    
    M[1] = D/2.0;
    M[2] = 0;
    M[3] = 0;

}

parameters{
   
    vector[6] X_raw[N];    
    real<lower = 0, upper = 10> Sigma;           
       
}

transformed parameters{

    vector[6] X[N];
    
    
    for (j in 1:3){    
        X[1,j] = 500 + X_raw[1, j] * 5000;    
        X[1,j+3] = 500 + X_raw[1, j] * 5000;       
    }
    
    
    for (i in 2:N){
    
        X[i, 1:3] = X[i-1, 1:3] + (X[i-1, 4:6] * dt) + X_raw[i-1, 1:3] * sig_P;    
        X[i, 4:6] = X[i-1, 4:6] + X_raw[i-1, 4:6] * sig_V;    
    
    }


}

model{
    
    
    //Initial Prior
    for (i in 1:N){
        X_raw[i] ~ normal(0, 1);   
    }
    
    

    for (i in 1:N){   
        //Observation model
        Y1[i, 1:3] ~ normal(X[i, 1:3] - M, Sigma);
  
        //Observation model
        Y2[i, 1:3] ~ normal(X[i, 1:3] + M, Sigma);
       
    }
}

generated quantities{

    vector[3] X1[N];
    vector[3] X2[N];
    
    
    for (i in 1:N){
    
        X1[i] = X[i, 1:3] - M;
        X2[i] = X[i, 1:3] + M;
    }

}

"""


#sm = pickle.load(open('osc_Kalman.pkl', 'rb'))
sm = pystan.StanModel(model_code=code) #Compile Stan Model
with open('osc_Kalman.pkl', 'wb') as f: #Save Stan model to file
    pickle.dump(sm, f)
#
#
stan_data = {'Y1': T_all_noise[:,0], 'Y2': T_all_noise[:,2], 'N': T_all.shape[0], 'dt': dt, 'sig_V': 5, 'sig_P': 8, 'D': 130.0}

fit = sm.sampling(data = stan_data, chains = 4, iter = 5000, refresh = 1, control = {'adapt_delta': 0.99})

sampler_params = fit.get_sampler_params(inc_warmup=False) 
n_divergent = reduce(lambda x, y: x + y['divergent__'].tolist(), sampler_params, []) 

if 1 in n_divergent: 
    print("N divergent = {}".format(sum(n_divergent)))   
    
    
    
##t0 = time.time()
##op = sm.optimizing(data = stan_data, tol_rel_grad = 1e2)
##t1 = time.time()
##
##print("Opt Time: {}".format(t1-t0))
##
##plt.plot(op['X'][:,0] - T_all[:,1,0])
##
##print(np.sum(np.abs(op['X'][:,0] - T_all[:,1,0])))
##
##print(np.sum(np.abs(T_all_noise[:,1,0] - T_all[:,1,0])))
##vb_ = sm.vb(data = stan_data,iter=10000,  init='random',sample_file = os.path.join(os.getcwd(), 'vb_results.csv'))
##
##
##with open('print_fit.txt', 'w') as f: #Save fit print output to file
##    print(fit, file=f)
##    
#la = fit.extract()
