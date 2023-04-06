# -*- coding: utf-8 -*-
"""

Camera model used for camera calibration. Aim is to make a Bayesian calibration model and to learn about the model

"""

from __future__ import print_function, division
import numpy as np
import itertools
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import cv2
import pickle

def create_chessboard(squares = (6,9), square_size = 24.5):    
    
    objp = np.zeros((np.prod(squares), 3), np.float32)
    
    p = itertools.product(np.arange(0,square_size*squares[0],square_size), np.arange(0,square_size*squares[1],square_size))
    objp[:,:2] = np.array([i for i in p])[:,::-1]
    
    
    
    return objp

def rotation(p, x, y, z):
    """Rotate point p by angle x, y, z"""
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                    [0, np.sin(x), np.cos(x)]])
                    
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1 , 0],
                    [-np.sin(y), 0 , np.cos(y)]])       
                    
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z) , 0],
                    [0, 0, 1]])
                    
    
                    
    R = np.dot(Rx, np.dot(Ry, Rz))
   
    return np.dot(R, p), R

def translation(p, x, y, z):
    
    T = np.array([x, y, z])
    
    return p + T
    
def get_RT_matrix(rotation, translation):
    """Return homogenous RT matrix where the 3rd collumn has been dropped"""
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rotation[0]), -np.sin(rotation[0])],
                    [0, np.sin(rotation[0]), np.cos(rotation[0])]])
                    
    Ry = np.array([[np.cos(rotation[1]), 0, np.sin(rotation[1])],
                   [0, 1 , 0],
                    [-np.sin(rotation[1]), 0 , np.cos(rotation[1])]])       
                    
    Rz = np.array([[np.cos(rotation[2]), -np.sin(rotation[2]), 0],
                   [np.sin(rotation[2]), np.cos(rotation[2]) , 0],
                    [0, 0, 1]])            
                        
    R = np.dot(Rx, np.dot(Ry, Rz))
    
    
    T = np.array(translation)
    
    RT = np.hstack((R, np.expand_dims(T, 1))) #Append T to last collumn of R
        
    
    return RT[:,[0,1,3]]

def get_camera_matrix(fx, fy, cx, cy, s):
    """Return a camera Matrix A
    input:
    fx: focal length x
    fy: focal length y
    cx: offset x
    cy: offset y
    s: skew parameter
    """
    
    return np.array([[fx, s, cx],
                     [0, fy, cy],
                     [0, 0, 1 ]])

def get_homography_matrix(M, RT):
    """Given the camera matrix M and the RT matrix (R1, R2, T) return the homography matrix H"""
    
    return np.dot(M, RT)
    
def mm_to_pixel(p_mm, k_x, k_y):
    
    c = np.array([k_x, k_y, 1])
  
    return p_mm * c

plot_example = False  


##Camera parameters
fx, fy = 2, 8 #Camera focal length
cx, cy = 0.5, 0.25 #Offsets of the camera center 
resolution = (1920, 1080) #Camera resolution
width, height = 5.0, 5.0 #sensor dimensions 
s = 0 #Skew parameter

kx, ky = resolution[0]/ width, resolution[1] / height #Scalling factor to convert screen coordinates in mm to pixels




#Create a chess board    
chessboard = create_chessboard()    
chessboardH = chessboard.copy()
chessboardH[:,-1] = 1

if plot_example:
    plt.scatter(chessboard[:,0], chessboard[:,1]) #Plot the chessboard
    plt.gca().invert_yaxis()

if plot_example:
    #Plot the chessboard in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(chessboard[:,0], chessboard[:,1], chessboard[:,2], depthshade = True)
    ax.view_init(elev=94., azim=-90)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([-20, 500])
    ax.set_ylim([-20, 500])
    ax.set_zlim([-20, 500])
    plt.gca().invert_yaxis()
    #plt.show()
    


M = get_camera_matrix(fx, fy, cx, cy, s) #Camera Intrinsic matrix   

#Image rotations and translations
#rot_trans =  [[(-np.pi/5, np.pi/4, 0), (200, 80, 400)],
#              [(np.pi/6, np.pi/8, 0.0), (50, 20, 300)],
#              [(-np.pi/3, -np.pi/4, 0.0), (60, 10, 355)],
#              [(np.pi/2.5, -np.pi/8, np.pi/3.5), (55, 20, 600)],
#              [(-np.pi/3.2, np.pi/6.7, 0.0), (45, 22, 400)],
#              [(np.pi/3.8, -np.pi/4.2, 0.0), (87, 46, 357)],
#              [(-np.pi/4, np.pi/8, 0.0), (465, 46, 438)],]
#rot_trans =  [[(-np.pi/5, np.pi/4, 0), (200, 80, 400)],
#              [(np.pi/6, np.pi/8, 0.0), (50, 20, 300)],]
#              [(-np.pi/3, -np.pi/4, 0.0), (60, 10, 355)]]
              
#              [(np.pi/2.5, -np.pi/8, np.pi/3.5), (55, 20, 600)],
#              [(-np.pi/3.2, np.pi/6.7, 0.0), (45, 22, 400)],
#              [(np.pi/3.8, -np.pi/4.2, 0.0), (87, 46, 357)],
#              [(-np.pi/4, np.pi/8, 0.0), (465, 46, 438)],]
              
rot_trans =  [[(-np.pi/5, np.pi/4, 0), (200, 80, 400)]]
              
all_imagepointspx = [] #Empty array to put all images points in

#Get all the images
for rt in rot_trans:
    
    chessboard_new = np.empty(chessboard.shape, dtype = np.float32)
    imgpoints_new = np.empty(chessboard.shape, dtype = np.float32)
    imgpointspx_new = np.empty(chessboard.shape, dtype = np.float32)

    RT = get_RT_matrix(rt[0], rt[1])    
    H = get_homography_matrix(M, RT)

    for c in range(len(chessboard)):   
        chessboard_new[c] = np.dot(RT, chessboardH[c])  #Chessboards position in space
        imgpoints_new[c] = np.dot(H, chessboardH[c])    #Chessoard mapped to image
        imgpoints_new[c] = imgpoints_new[c] / imgpoints_new[c][-1]  #Divide through by Z to make w = 1 and add some noise
        imgpointspx_new[c] = mm_to_pixel(imgpoints_new[c], kx, ky) + np.append(np.random.normal(0, 0.005, size = 2), 0) 
    
    all_imagepointspx.append(imgpointspx_new)        
    
    if plot_example:
          
        fig = plt.figure()
        ax2 = fig.add_subplot(111, projection='3d')
        ax2.scatter(chessboard_new[:,0], chessboard_new[:,1], chessboard_new[:,2], depthshade = True)
        ax2.text(chessboard_new[0,0],chessboard_new[0,1], chessboard_new[0,2],  '0', size=20, zorder=1,  
                color='k') 
        ax2.text(chessboard_new[-1,0],chessboard_new[-1,1], chessboard_new[-1,2],  '-1', size=20, zorder=1,  
                color='k') 
                
        ax2.view_init(elev=94., azim=-90)
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        ax2.set_xlim([-20, 500])
        ax2.set_ylim([-20, 500])
        ax2.set_zlim([-20, 500])
        plt.gca().invert_yaxis()
        plt.gca().invert_zaxis()
        
        #Plot the chessboards projection onto the camera
        fig2 = plt.figure()
        
        plt.scatter(imgpointspx_new[:,0], imgpointspx_new[:,1])
        plt.axvline(width/2)
        plt.axhline(height/2)
        plt.xlim([0,resolution[0]])
        plt.ylim([0, resolution[1]])
        plt.gca().invert_yaxis()

#############################
##############################
####Bayesian Analysis#########
##
import pystan

#all_images = np.expand_dims(imgpointspx, 0)

sm = pickle.load(open('bayesianCam_mult_stan.pkl', 'rb'))
sm = pystan.StanModel(file='bayesianCam_mult.stan') #Compile Stan Model
with open('bayesianCam_mult_stan.pkl', 'wb') as f: #Save Stan model to file
    pickle.dump(sm, f)

all_imagepointspx = np.array(all_imagepointspx)
stan_images = all_imagepointspx[:,:,:2].flatten()

stan_data = dict(images = stan_images, 
                 P = len(all_imagepointspx), 
                 K = 54, 
                 chessboardH = chessboardH, 
                 resolution = resolution,
                 k_vec = np.array([kx, ky, 1]))

fit = sm.sampling(data=stan_data, iter = 2000, chains=4, 
                      refresh = 5, init = "random", 
                      control = {'adapt_delta': 0.9, 'max_treedepth': 20}) #Fit model  



sampler_params = fit.get_sampler_params(inc_warmup=False) 
n_divergent = reduce(lambda x, y: x + y['n_divergent__'].tolist(), sampler_params, []) 

if 1 in n_divergent: 
    print("N divergent = {}".format(sum(n_divergent)))      
    with open("error logging.csv", 'a') as f:
        f.write("N divergent = {}\n".format(sum(n_divergent)))      
    
else:
    print("N divergent = {}".format(0))                                

la = fit.extract(permuted = True)

plt.plot(la['r1'])


plt.figure()
sns.distplot(la['fx'])
plt.title("fx: {}".format(fx))

plt.figure()
sns.distplot(la['fy'])
plt.title("fy: {}".format(fy))

plt.figure()
sns.distplot(la['cx'])
plt.title("cx: {}".format(cx))

plt.figure()
sns.distplot(la['cy'])
plt.title("cy: {}".format(cy))
#
plt.figure()
sns.distplot(la['r1'])
plt.title("r1: {}".format(rot_trans[0][0][0]))

#plt.figure()
#sns.distplot(la['r2'][:,0])
#plt.title("r2: {}".format(rot_trans[0][0][1]))
#
#plt.figure()
#sns.distplot(la['r3'][:,0])
#plt.title("r3: {}".format(rot_trans[0][0][2]))
