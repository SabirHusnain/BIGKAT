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
    
##Camera parameters
fx, fy = 2, 8 #Camera focal length in pixels
cx, cy = 0.5, 0.25 #Offsets of the camera center in pixels
resolution = (1920, 1080) #Camera resolution
width, height = 5.0,5.0 #sensor dimensions 
s = 0 #Skew parameter

kx, ky = resolution[0]/ width, resolution[1] / height #Scalling factor to convert screen coordinates in mm to pixels




#Create a chess board    
chessboard = create_chessboard()    
chessboardH = chessboard.copy()
chessboardH[:,-1] = 1
plt.scatter(chessboard[:,0], chessboard[:,1]) #Plot the chessboard
plt.gca().invert_yaxis()


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



#Place the chessboard somewhere in the world
##Rotate and translate the chessboard
chessboard2 = np.empty(chessboard.shape)
imgpoints2 = np.empty(chessboard.shape)
imgpointspx = np.empty(chessboard.shape)

RT = get_RT_matrix((np.pi/5, np.pi/4, 0), (100, 150, 300))
M = get_camera_matrix(fx, fy, cx, cy, s) #Camera Intrinsic matrix   
H = get_homography_matrix(M, RT)

for c in range(len(chessboard)):   
    chessboard2[c] = np.dot(RT, chessboardH[c])
    imgpoints2[c] = np.dot(H, chessboardH[c])
    imgpoints2[c] = imgpoints2[c] / imgpoints2[c][-1] #Divide through by Z to make w = 1
    imgpointspx[c] = mm_to_pixel(imgpoints2[c], kx, ky)
plot_example = True

if plot_example:
    #Plot the rotated and translated chessboard
    fig = plt.figure()
    ax2 = fig.add_subplot(111, projection='3d')
    ax2.scatter(chessboard2[:,0], chessboard2[:,1], chessboard2[:,2], depthshade = True)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_xlim([-20, 500])
    ax2.set_ylim([-20, 500])
    ax2.set_zlim([-20, 500])
    plt.gca().invert_yaxis()
#    plt.show()
    
    #Plot the chessboards projection onto the camera
    plt.figure()
    plt.scatter(imgpoints2[:,0], imgpoints2[:,1])
    plt.axvline(width/2)
    plt.axhline(height/2)
    plt.xlim([0,width])
    plt.ylim([0,height])
    plt.gca().invert_yaxis()
    
    #Plot the chessboards projection onto the camera in pixels
    plt.figure()
    plt.scatter(imgpointspx[:,0], imgpointspx[:,1])
    plt.axvline(resolution[0]/2)
    plt.axhline(resolution[1]/2)
    plt.xlim([0,resolution[0]])
    plt.ylim([0, resolution[1]])
    plt.gca().invert_yaxis()




############################
#############################
###Bayesian Analysis#########
#
import pystan

all_images = np.expand_dims(imgpointspx, 0)

sm = pickle.load(open('bayesianCam_stan.pkl', 'rb'))
sm = pystan.StanModel(file='bayesianCam.stan') #Compile Stan Model
with open('bayesianCam_stan.pkl', 'wb') as f: #Save Stan model to file
    pickle.dump(sm, f)

stan_data = dict(images = all_images, 
                 P = len(all_images), 
                 K = 54, 
                 chessboardH = chessboardH, 
                 resolution = resolution,
                 k_vec = np.array([kx, ky, 1]))

fit = sm.sampling(data=stan_data, iter = 1000, chains=1, 
                      refresh = 10, init = "random",
                      control = {'adapt_delta': 0.8, 'max_treedepth': 15}) #Fit model  


#
#sampler_params = fit.get_sampler_params(inc_warmup=False) 
#n_divergent = reduce(lambda x, y: x + y['n_divergent__'].tolist(), sampler_params, []) 
#
#if 1 in n_divergent: 
#    print("N divergent = {}".format(sum(n_divergent)))      
#    with open("error logging.csv", 'a') as f:
#        f.write("N divergent = {}\n".format(sum(n_divergent)))      
#    
#else:
#    print("N divergent = {}".format(0))                                
#
la = fit.extract(permuted = True)

plt.figure()
sns.distplot(la['fx'])
plt.title("fx")

plt.figure()
sns.distplot(la['fy'])
plt.title("fy")

plt.figure()
sns.distplot(la['cx'])
plt.title("cx")

plt.figure()
sns.distplot(la['cy'])
plt.title("cz")
#
# 
# 
## 
######Cv2 Analysis
###
##import pickle
##
###
##imp = pickle.load(open("imgpoints.pkl", 'rb'))
##obj = pickle.load(open("objpoints.pkl", 'rb'))
###
##imp = [i.squeeze() for i in imp]
##imp[0][:] = 0
###imgpoints2 = all_imagepointspx[:,:,:2]
###imgpoints2 = [imgpoints2[0] for i in range(imgpoints2.shape[0])]
###
##objpoints2 = [chessboard for ch in range(len(all_imagepointspx))]
##
##all_imagepointspx2 = [i.astype('float32') for i in all_imagepointspx] #For some reason this has to be done like this for the calibration to work. Very bizaar
###calib_params = cv2.calibrateCamera(obj, imp, resolution, None, None)
##
##M2 = M.copy()
##M2[0,0] = M2[0,0] + 10
##M2[1,1] = M2[1,1] + 100
##M2[0,2] = 0
##M2[1,2] = 0
##
##
##rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(objpoints2, all_imagepointspx2, resolution, None, None)
##print(camera_matrix)
##
###calib_params = cv2.calibrateCamera(objpoints2[:2], imp[:2], resolution, None, None)
##
